import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm
from models.vqvae import VQVAE
from models.lpips import LPIPS
from models.discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from torch.optim import Adam
from torchvision.utils import make_grid
import wandb
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
    wandb.init(project="Face-diffusion", entity="megleczmate", sync_tensorboard=True, tags=["vqvae"])
    # get current time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    wandb.run.name = config['task_name'] + '_' + current_time
    wandb.run.save()
    wandb.config.update(config)
    return wandb

def train(rank, world_size, args):
    print(f"Running basic DDP example on rank {rank}.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################

    if train_config['distributed_data_paralell']:
        device = rank

    # Create the model and dataset #
    model_base = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    
    if train_config['distributed_data_paralell']:
        model = DDP(
                model_base,
                device_ids=[rank],
            )
    else:
        model = model_base
        

    # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    sampler = DistributedSampler(im_dataset, num_replicas=world_size, rank=rank)

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=False,
                             sampler=sampler)
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.MSELoss()
    
    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS(device='cuda').eval().to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
    
    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    
    disc_step_start = train_config['disc_start']
    step_count = 0
    
    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    wandb_image_save_steps = train_config['autoencoder_img_save_steps_wandb']
    log_step_steps_wandb = train_config['autoencoder_log_step_steps_wandb']
    img_save_count = 0
    
    for epoch_idx in range(num_epochs):
        recon_losses = []
        codebook_losses = []
        #commitment_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        
        for im in tqdm(data_loader, ascii=True):
            step_count += 1
            im = im.float().to(device)
            
            # Fetch autoencoders output(reconstructions)
            model_output = model(im)
            output, z, quantize_losses = model_output
            
            # Image Saving Logic
            if step_count % image_save_steps == 0 or step_count == 1:
                sample_size = min(8, im.shape[0])
                save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                
                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                if not os.path.exists(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples')):
                    os.mkdir(os.path.join(train_config['task_name'], 'vqvae_autoencoder_samples'))
                img.save(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples',
                                      'current_autoencoder_sample_{}.png'.format(img_save_count)))
                img_save_count += 1
                img.close()

            # Save images to wandb
            if (step_count % wandb_image_save_steps == 0 or step_count == 1) and rank == 0:
                sample_size = min(8, im.shape[0])
                save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                
                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                wandb.log({"Autoencoder Samples": [wandb.Image(grid)]})
            
            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, im) 
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            g_loss = (recon_loss +
                      (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps) +
                      (train_config['commitment_beta'] * quantize_losses['commitment_loss'] / acc_steps))
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())
            # Adversarial loss only if disc_step_start steps passed
            if step_count > disc_step_start:
                disc_fake_pred = discriminator(model_output[0])
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                           device=disc_fake_pred.device))
                gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps
            lpips_loss = torch.mean(lpips_model(output, im)) / acc_steps
            perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
            g_loss += train_config['perceptual_weight']*lpips_loss / acc_steps
            losses.append(g_loss.item())
            g_loss.backward()

            # log the losses in every 500 steps
            if (step_count % log_step_steps_wandb == 0 or step_count == 1) and rank == 0:
                wandb.log({ "Step": step_count,
                            "Recon Loss": np.mean(recon_losses),
                            "Perceptual Loss": np.mean(perceptual_losses),
                            "Codebook Loss": np.mean(codebook_losses),
                            "Generator Loss": np.mean(gen_losses)})

            #####################################
            
            ######### Optimize Discriminator #######
            if step_count > disc_step_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.zeros(disc_fake_pred.shape,
                                                            device=disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred,
                                                torch.ones(disc_real_pred.shape,
                                                           device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                disc_loss.backward()
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            #####################################
            
            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        if rank == 0:
            if len(disc_losses) > 0:
                print(
                    'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                    'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
                    format(epoch_idx + 1,
                        np.mean(recon_losses),
                        np.mean(perceptual_losses),
                        np.mean(codebook_losses),
                        np.mean(gen_losses),
                        np.mean(disc_losses)))
                wandb.log({ "Epoch": epoch_idx + 1,
                            "Recon Loss": np.mean(recon_losses),
                            "Perceptual Loss": np.mean(perceptual_losses),
                            "Codebook Loss": np.mean(codebook_losses),
                            "Generator Loss": np.mean(gen_losses),
                            "Discriminator Loss": np.mean(disc_losses)})
            else:
                print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}'.
                    format(epoch_idx + 1,
                            np.mean(recon_losses),
                            np.mean(perceptual_losses),
                            np.mean(codebook_losses)))
                wandb.log({ "Epoch": epoch_idx + 1,
                            "Recon Loss": np.mean(recon_losses),
                                "Perceptual Loss": np.mean(perceptual_losses),
                                "Codebook Loss": np.mean(codebook_losses)})    
        

        if rank == 0:
            torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                        train_config['vqvae_autoencoder_ckpt_name']))
            torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                                train_config['vqvae_discriminator_ckpt_name']))
        
    cleanup()
    print('Done Training...')


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
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
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
