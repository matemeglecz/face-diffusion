import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from scheduler.linear_noise_scheduler_ddim import LinearNoiseSchedulerDDIM
from utils.config_utils import *
from collections import OrderedDict
from datetime import datetime
from tools.sample_ddpm_attr_cond import ddim_inversion
from torchvision.transforms import Compose, Normalize
from dataset.celeb_dataset import CelebDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# free gpu memory
torch.cuda.empty_cache()


def glasses_loss(x, classifier_model, device='cuda'):
    # Create a resnet-18 model
    
    classifier_model.train()  # Ensure the model is in training mode

    # Move the input tensor `x` to the correct device
    x = x.to(device)

    transforms = Compose([
            #Normalize(mean=[-0.5047, -0.2201,  0.0777], std=[1.0066, 0.8887, 0.6669])
            #Normalize(mean=[-0.4908,  0.0627,  0.1011], std=[0.5076, 0.4108, 0.4806])
            #Normalize(mean=[-0.5099,  0.0534,  0.0902], std=[0.4977, 0.4097, 0.4811]) # smiles
            Normalize(mean=[-0.4932,  0.0717,  0.0965], std=[0.5124, 0.4079, 0.4835]) # chubby, mouth
        ])
    x = transforms(x)

    # Predict the glasses attribute
    pred = classifier_model(x)

    # Generate a target tensor with the same batch size as the input (assuming a binary classification task)
    target = torch.ones(pred.size(0), 1).to(device)  # Assuming all targets are 1 (glasses present)

    # Calculate the loss using Binary Cross Entropy
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(pred, target)

    # Return the loss with gradients enabled
    return loss


def sample(model, classifier_model, cond, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, use_ddim=False, start_step=0, num_steps=1000, noise_input=None, dir=''):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """    

    # seed random for reproducibility
    #torch.manual_seed(9)

    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    
    ########### Sample random noise latent ##########
    if noise_input is not None:
        xt = noise_input.to(device)
    else:
        xt = torch.randn((train_config['num_samples'],
                        autoencoder_model_config['z_channels'],
                        im_size,
                        im_size)).to(device)
    ###############################################


    ############# Validate the config #################
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, ("This sampling script is for class conditional "
                                          "but no conditioning config found")
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'attribute' in condition_types, ("This sampling script is for attribute conditional "
                                          "but no class condition found in config")
    #validate_class_config(condition_config)
    ###############################################
    
    ############ Create Conditional input ###############
    num_classes = condition_config['attribute_condition_config']['attribute_condition_num']
    #sample_classes = torch.randint(0, num_classes, (train_config['num_samples'], ))
    #print('Generating images for {}'.format(list(sample_classes.numpy())))
    cond_input = {
        # 'class': torch.nn.functional.one_hot(sample_classes, num_classes).to(device)
        #  ['Male', 'Young', 'Bald', 'Bangs', 'Receding_Hairline', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Straight_Hair', 'Wavy_Hair', 'No_Beard', 'Goatee', 'Mustache', 'Sideburns', 'Narrow_Eyes', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose']
        'attribute': cond

    }
    # Unconditional input for classifier free guidance
    uncond_input = {
        'attribute': cond_input['attribute'] * 0
    }
    ###############################################
    
    # By default classifier free guidance is disabled
    # Change value in config or change default value here to enable it
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)
    
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    if not use_ddim:
        num_steps = diffusion_config['num_timesteps']
    
    ################# Sampling Loop ########################
    for i in tqdm(reversed(range(num_steps - start_step)), total=num_steps):
        torch.set_grad_enabled(True)
        xt_in = xt.clone()

        # activate gradient for xt
        xt.requires_grad_(True)
        
        timestep = ((i-1) * (1000 // num_steps)) + 1
        #print(timestep)
        
        # Get prediction of noise
        t = (torch.ones((xt.shape[0],))*timestep).long().to(device)
        
        with torch.no_grad():
            noise_pred_cond = model(xt_in, t, cond_input)
        
        if cf_guidance_scale > 1:
            noise_pred_uncond = model(xt, t, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale*(noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
        
        # If DDIM is enabled, we need to also compute t_prev for the DDIM reverse process
        
        if use_ddim:
            t_prev = (torch.ones((xt.shape[0],)).to(device) * max(t - (1000 // num_steps), 1)).long().to(device)
            xt_new, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, t, t_prev)  # Use DDIM sampling
        else:
            xt_new, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))  # Use DDPM sampling
        

        #loss = color_loss(x0_pred) * 2
        # if not first step, use glasses loss

        loss = glasses_loss(x0_pred, classifier_model) * 0.02

        # set the loss to require grad
        
        #if i % 10 == 0:
        #    print(i, "loss:", loss.item())

        cond_grad = -torch.autograd.grad(loss, xt, retain_graph=True)[0] 

        # momentum
        if i < 950 and i > 200:
            cond_grad = cond_grad + 0.9 * cond_grad_prev        
        if i < 951:
            cond_grad_prev = cond_grad

        

        # apply gradient clipping
        cond_grad = torch.clamp(cond_grad, -0.05, 0.05)

        
        xt = xt + cond_grad
        #print(cond_grad.max(), cond_grad.min())

        if use_ddim:
            t_prev = (torch.ones((xt.shape[0],)).to(device) * max(t - (1000 // num_steps), 1)).long().to(device)
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, t, t_prev)  # Use DDIM sampling
        else:
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))  # Use DDPM sampling

        if i == 0:
            # Decode ONLY the final image to save time
            ims = vae.decode(xt)
        else:
            ims = x0_pred
        
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        

        # save latent to pt        
        #torch.save(xt, os.path.join(train_config['task_name'], 'cond_attr_samples', dir, current_time, 'xt_{}.pt'.format(i)))
    ##############################################################

    return ims, cond_input


# Read the config file #
with open('celebhq-512-64-train-komondor_b/celeba_komondor_512_b.yaml', 'r') as file:
#with open('celebhq-1024-64-16k-komondor/celeba_komondor_16k.yaml', 'r') as file:
#with open('celebhq-512-64/celeba_komondor_512.yaml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
print(config)
########################

diffusion_config = config['diffusion_params']
dataset_config = config['dataset_params']
diffusion_model_config = config['ldm_params']
autoencoder_model_config = config['autoencoder_params']
train_config = config['train_params']
sample_config = config['sample_params']

########## Create the noise scheduler #############

if sample_config['use_ddim']:
    print('Using DDIM')
    scheduler = LinearNoiseSchedulerDDIM(num_timesteps=diffusion_config['num_timesteps'],
                                            beta_start=diffusion_config['beta_start'],
                                            beta_end=diffusion_config['beta_end'])
else:
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                    beta_start=diffusion_config['beta_start'],
                                    beta_end=diffusion_config['beta_end'])
###############################################

########## Load Unet #############
model = Unet(im_channels=autoencoder_model_config['z_channels'],
                model_config=diffusion_model_config).to(device)

if os.path.exists(os.path.join(train_config['task_name'],
                                train_config['ldm_ckpt_name'])):
    

    ddp_state_dict = torch.load(os.path.join(train_config['task_name'],
                                                    train_config['ldm_ckpt_name']),
                                        map_location=device)
    new_state_dict = OrderedDict()
    for k, v in ddp_state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    ddp_state_dict = new_state_dict
    print('Loaded unet checkpoint')
    model.load_state_dict(ddp_state_dict)
else:
    raise Exception('Model checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                                        train_config['ldm_ckpt_name'])))

model.train()
#####################################

# Create output directories
if not os.path.exists(train_config['task_name']):
    os.mkdir(train_config['task_name'])

########## Load VQVAE #############
vae = VQVAE(im_channels=dataset_config['im_channels'],
            model_config=autoencoder_model_config).to(device)
vae.eval()

# Load vae if found
if os.path.exists(os.path.join(train_config['task_name'],
                                train_config['vqvae_autoencoder_ckpt_name'])):
    print('Loaded vae checkpoint')

    vae_state_dict = torch.load(os.path.join(train_config['task_name'],
                                                train_config['vqvae_autoencoder_ckpt_name']),
                                    map_location=device)
    
    
    new_state_dict = OrderedDict()

    for k, v in vae_state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        new_state_dict[name] = v   

    #new_state_dict = vae_state_dict     
    
    vae.load_state_dict(new_state_dict, strict=True)
else:
    raise Exception('VAE checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                train_config['vqvae_autoencoder_ckpt_name'])))
#####################################
'''
classifier_model = torchvision.models.resnet18(pretrained=False)

num_ftrs = classifier_model.fc.in_features

# Modify the last fully connected layer for binary classification (1 output)
classifier_model.fc = torch.nn.Linear(num_ftrs, 1)

# Load weights from 'celeba_resnet18_latent_glasses_classifier_1.pth'
state = torch.load('celeba_resnet18_latent_glasses_classifier_1.pth', map_location=device)

new_state_dict = OrderedDict()
for k, v in state.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v

classifier_model.load_state_dict(new_state_dict)
'''
from models.simple_cnn import SimpleCNN

classifier_model = SimpleCNN()
#classifier_model.load_state_dict(torch.load('celeba_cnn_latent_glasses_classifier_0.pth', map_location=device))
classifier_model.load_state_dict(torch.load('celeba_cnn_latent_mouth_classifier_0_700_ddpm.pth', map_location=device))



classifier_model.to(device)



im_dataset_val = CelebDataset(split='val',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                use_latents=False,
                                latent_path=os.path.join(train_config['task_name'],
                                                         train_config['vqvae_latent_dir_name']),
                                condition_config=diffusion_model_config['condition_config'],
                                )


torch.manual_seed(4)


# dataloader
val_loader = torch.utils.data.DataLoader(im_dataset_val,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=4,)

# sample

save_dir = '/mnt/g/data/mouth_val_0.02_warm_up_mid_shorter_2'

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

import numpy as np

image_num = 0

for i, (im_real, cond) in tqdm(enumerate(val_loader), total=len(val_loader)):
    #if i in [3, 5, 6, 8, 14, 17, 20, 23, 25, 29, 30]:
    #    image_num += 1
    #    continue

    torch.cuda.empty_cache()
    im_real = im_real.to(device)
    attr = cond['attribute'].clone()    
    
    cond['attribute'] = cond['attribute'].to(device)

    cond = torch.tensor(cond['attribute']).to(device)


    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])

    noise_input = torch.randn((train_config['num_samples'],
                            autoencoder_model_config['z_channels'],
                            im_size,
                            im_size)).to(device)

    ims, cond_trans = sample(model, classifier_model, cond, scheduler, train_config, diffusion_model_config,
                autoencoder_model_config, diffusion_config, dataset_config, vae, use_ddim=True, dir='', noise_input=noise_input, num_steps=1000, start_step=0)
    

    for j in range(ims.shape[0]):
        img_j = ims[j].permute(1, 2, 0).cpu().numpy()
        img_j = (img_j * 255).astype(np.uint8)
        img_j = Image.fromarray(img_j)
        img_j.save(os.path.join(save_dir, f'{image_num}.png'))

        #write cond to file txt
        with open(os.path.join(save_dir, f'{image_num}.txt'), 'w') as f:
            f.write(str(attr[j].cpu().numpy()))

        image_num += 1

        

