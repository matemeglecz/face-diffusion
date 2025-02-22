import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.text_utils import *
from utils.config_utils import *
from utils.diffusion_utils import *
import wandb
import os
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import socket
from mpi4py import MPI

def setup(rank, world_size):
    print('Setting up process group')
    #torch.cuda.set_device(rank)
    # Initialize the process group for distributed training
    dist.init_process_group(
        backend='nccl',     # Use 'gloo' for CPU and 'nccl' for GPU
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )

    print("setup finished")

def cleanup():
    dist.destroy_process_group()

def setup_wandb(config):
    wandb.init(project="Face-diffusion", entity="megleczmate", sync_tensorboard=True, tags=["ddpm"])
    # get current time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    wandb.run.name = config['task_name']

    if config['continue']:
    	wandb.run.name = wandb.run.name + '_continue' 
  
    wandb.run.name = wandb.run.name + '_' + current_time

    wandb.run.save()
    wandb.config.update(config)
    return wandb


def train(rank, world_size, args):
    print(f"Running DDP on rank {rank}.")
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    if rank == 0:
        wandb = setup_wandb(config)

    setup(rank, world_size)

    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']


    #############################

    if train_config['distributed_data_paralell']:
        device = rank
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    # Instantiate Condition related components
    text_tokenizer = None
    text_model = None
    empty_text_embed = None
    condition_types = []
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']
        if 'text' in condition_types:
            validate_text_config(condition_config)
            with torch.no_grad():
                # Load tokenizer and text model based on config
                # Also get empty text representation
                text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']
                                                                     ['text_embed_model'], device=device)
                empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)
            
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                use_latents=True,
                                latent_path=os.path.join(train_config['task_name'],
                                                         train_config['vqvae_latent_dir_name']),
                                condition_config=condition_config,
                                )
    
    sampler = DistributedSampler(im_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=False,
                             sampler=sampler,)
                             #num_workers=4)
    
    # Instantiate the unet model
    model_base = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    
    if train_config['distributed_data_paralell']:
        model = DDP(
                model_base,
                device_ids=[rank],
            )
    else:
        model = model_base
    
    if config['continue']:
        #ckpt name
        name = train_config['ldm_ckpt_name']

        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                        name)))
    
    model.train()
    
    vae = None
    # Load VAE ONLY if latents are not to be saved or some are missing
    if not im_dataset.use_latents:
        print('Loading vqvae model as latents not present')
        vae_base = VQVAE(im_channels=dataset_config['im_channels'],
                    model_config=autoencoder_model_config).to(device)
        
        if train_config['distributed_data_paralell']:
            vae = DDP(
                vae_base,
                device_ids=[rank],
            )
        else:
            vae = vae_base

        vae.eval()
        # Load vae if found
        if os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['vqvae_autoencoder_ckpt_name'])):
            print('Loaded vae checkpoint')
            vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                        train_config['vqvae_autoencoder_ckpt_name']),
                                           map_location=device))
        else:
            raise Exception('VAE checkpoint not found and use_latents was disabled')
    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    
    # Load vae and freeze parameters ONLY if latents already not saved
    if not im_dataset.use_latents:
        assert vae is not None
        for param in vae.parameters():
            param.requires_grad = False

    log_step_steps_wandb = train_config['ddpm_log_step_steps_wandb']
    wandb_image_save_steps = train_config['ddpm_img_save_steps_wandb']
    
    step = 0

    if config['continue']:
        step = config['last_step'] + 1

    start_epoch = 0

    if config['continue']:
        start_epoch = config['last_epoch'] + 1

    # Run training
    for epoch_idx in range(start_epoch, num_epochs):
        losses = []
        for data in tqdm(data_loader, ascii=True):
            cond_input = None
            if condition_config is not None:
                im, cond_input = data
            else:
                im = data
            optimizer.zero_grad()
            im = im.float().to(device)
            if not im_dataset.use_latents:
                with torch.no_grad():
                    im, _ = vae.encode(im)
                    
            ########### Handling Conditional Input ###########
            if 'text' in condition_types:
                with torch.no_grad():
                    assert 'text' in cond_input, 'Conditioning Type Text but no text conditioning input present'
                    validate_text_config(condition_config)
                    text_condition = get_text_representation(cond_input['text'],
                                                                 text_tokenizer,
                                                                 text_model,
                                                                 device)
                    text_drop_prob = get_config_value(condition_config['text_condition_config'],
                                                      'cond_drop_prob', 0.)
                    text_condition = drop_text_condition(text_condition, im, empty_text_embed, text_drop_prob)
                    cond_input['text'] = text_condition
            if 'image' in condition_types:
                assert 'image' in cond_input, 'Conditioning Type Image but no image conditioning input present'
                validate_image_config(condition_config)
                cond_input_image = cond_input['image'].to(device)
                # Drop condition
                im_drop_prob = get_config_value(condition_config['image_condition_config'],
                                                      'cond_drop_prob', 0.)
                cond_input['image'] = drop_image_condition(cond_input_image, im, im_drop_prob)
            if 'class' in condition_types:
                assert 'class' in cond_input, 'Conditioning Type Class but no class conditioning input present'
                validate_class_config(condition_config)
                class_condition = torch.nn.functional.one_hot(
                    cond_input['class'],
                    condition_config['class_condition_config']['num_classes']).to(device)
                class_drop_prob = get_config_value(condition_config['class_condition_config'],
                                                   'cond_drop_prob', 0.)
                # Drop condition
                cond_input['class'] = drop_class_condition(class_condition, class_drop_prob, im)
            if 'attribute' in condition_types:
                assert 'attribute' in cond_input, 'Conditioning Type Attribute but no attribute conditioning input present'
                validate_attribute_config(condition_config)
                attribute_condition = cond_input['attribute'].to(device)
                attribute_drop_prob = get_config_value(condition_config['attribute_condition_config'],
                                                   'cond_drop_prob', 0.)
                # Drop condition
                # we dont drop attributes
                #cond_input['attribute'] = drop_attribute_condition(attribute_condition, attribute_drop_prob, im)

            ################################################
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, cond_input=cond_input)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            step += 1


            if (step % log_step_steps_wandb == 0 or step == 1) and rank == 0:
                wandb.log({'loss': loss.item(), 'epoch': epoch_idx, 'step': step})

            if (step % wandb_image_save_steps == 0 or step == 1) and rank == 0:
                with torch.no_grad():
                    im_save = scheduler.add_noise(im, noise, t)
                    noisy_im_save = scheduler.add_noise(im, noise, t)
                    noise_pred_save = model(noisy_im_save, t, cond_input=cond_input)

                    # transpose to NHWC
                    im_save = im_save.permute(0, 2, 3, 1)
                    noisy_im_save = noisy_im_save.permute(0, 2, 3, 1)
                    noise_pred_save = noise_pred_save.permute(0, 2, 3, 1)

                    wandb.log({'original_image': [wandb.Image(im_save[0].cpu().detach().numpy())],
                               'noisy_image': [wandb.Image(noisy_im_save[0].cpu().detach().numpy())],
                               'reconstructed_image': [wandb.Image(noise_pred_save[0].cpu().detach().numpy())]})


        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        
        if rank == 0:
            # copy previous save to backup if exists
            if os.path.exists(os.path.join(train_config['task_name'],
                                           train_config['ldm_ckpt_name'])):
                os.system('cp {} {}'.format(os.path.join(train_config['task_name'],
                                                        train_config['ldm_ckpt_name']),
                                            os.path.join(train_config['task_name'],
                                                        'backup_' + train_config['ldm_ckpt_name'])))


            torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ldm_ckpt_name']))
    
    cleanup()
    print('Done Training ...')

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

if __name__ == '__main__':
    os.environ["NCCL_DEBUG"] = "INFO"
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/celebhq_attribute_cond.yaml', type=str)
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD

    hostname = socket.gethostbyname(socket.getfqdn())

    world_size = torch.cuda.device_count()  # Number of GPUs available

    print(world_size)

    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)

    print(hostname, port)
    
    # Launch one process per GPU
    torch.multiprocessing.spawn(train, args=(world_size,args), nprocs=world_size, join=True)
