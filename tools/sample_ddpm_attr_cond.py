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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, cond, scheduler, train_config, diffusion_model_config,
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
        timestep = ((i-1) * (1000 // num_steps)) + 1
        #print(timestep)
        
        # Get prediction of noise
        t = (torch.ones((xt.shape[0],))*timestep).long().to(device)
        noise_pred_cond = model(xt, t, cond_input)
        
        if cf_guidance_scale > 1:
            noise_pred_uncond = model(xt, t, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale*(noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
        
        # If DDIM is enabled, we need to also compute t_prev for the DDIM reverse process
        if use_ddim:
            timestep_prev = max(timestep - (1000 // num_steps), 1)
            t_prev = (torch.ones((xt.shape[0],)).to(device) * max(timestep - (1000 // num_steps), 1)).long().to(device)
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

        if dir != '':
            grid = make_grid(ims, nrow=1)
            img = torchvision.transforms.ToPILImage()(grid)


            if not os.path.exists(os.path.join(train_config['task_name'], 'cond_attr_samples', dir, current_time)):
                os.makedirs(os.path.join(train_config['task_name'], 'cond_attr_samples', dir, current_time), exist_ok=True)
            img.save(os.path.join(train_config['task_name'], 'cond_attr_samples', dir, current_time, 'x0_{}.png'.format(i)))
            img.close()

        # save latent to pt        
        #torch.save(xt, os.path.join(train_config['task_name'], 'cond_attr_samples', dir, current_time, 'xt_{}.pt'.format(i)))
    ##############################################################

    return ims, cond_input

def ddim_inversion(scheduler, vae, xt, diffusion_config, condition_input, model, train_config, num_inference_steps=None, dir='', save_img=True):
    r"""
    Reverse the process by diffusing the image forward in time.
    :param scheduler: the noise scheduler used (e.g., LinearNoiseSchedulerDDIM)
    :param vae: the variational autoencoder (VAE) to encode and decode images
    :param xt: image tensor that will be diffused forward
    :param diffusion_config: configuration for the diffusion process
    :param condition_input: the conditioning input for the image
    :param model: the diffusion model (e.g., Unet)
    :param train_config: the training configuration
    """

    xt = xt.to(device)  # Ensure image is on the correct device
    xt = (xt * 2) - 1  # Rescale from [0, 1] to [-1, 1] to match the model's input range

    # First, encode the image into latent space using the VAE
    z, _ = vae.encode(xt)

    all_timesteps = diffusion_config['num_timesteps']

    # If the number of inference steps is not provided, use all timesteps
    if num_inference_steps is None:
        num_timesteps = all_timesteps

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    
    intermediate_latents = []
    # Move forward in time by applying noise progressively
    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps):
        t = (torch.ones((z.shape[0],)) * (i * (all_timesteps // num_inference_steps) + 1)).long().to(z.device)

        if i >= num_inference_steps - 1: continue

        # Predict noise based on current step and conditions
        noise_pred = model(z, t, condition_input)
        
        next_timestep = t
        current_timestep = max(0, t - (all_timesteps // num_inference_steps))

        # Use the noise prediction to forward-sample to the next timestep using DDIM forward equation
        # Reverse the reverse process from sample_prev_timestep
        alpha_t = scheduler.alpha_cum_prod.to(z.device)[current_timestep]
        alpha_t_next = scheduler.alpha_cum_prod.to(z.device)[next_timestep]
        
        '''
        z_next = (
            torch.sqrt(alpha_t_next) * z +
            torch.sqrt(1 - alpha_t_next) * noise_pred
        )
        '''

        z_next = (z - torch.sqrt(1 - alpha_t) * noise_pred) * (torch.sqrt(alpha_t_next) / torch.sqrt(alpha_t)) + torch.sqrt(1 - alpha_t_next) * noise_pred
        
        # Optionally, if stochasticity is involved (if ddim_eta > 0), add noise at each step
        if scheduler.ddim_eta > 0:
            variance = (1 - alpha_t_next) / (1 - alpha_t) * scheduler.betas.to(z.device)[t]
            sigma = scheduler.ddim_eta * torch.sqrt(variance)
            z_next = z_next + sigma * torch.randn_like(z_next)
        
        z = z_next  # Move to the next time step

        intermediate_latents.append(z)

        if save_img:
            ims_clamped = torch.clamp(z, -1., 1.).detach().cpu()
            ims_clamped = (ims_clamped + 1) / 2  # Rescale to [0, 1]
            
            # Convert to image and save
            grid = make_grid(ims_clamped, nrow=1)
            img = torchvision.transforms.ToPILImage()(grid)
            
            # Save images at each step for visualization
            save_dir = os.path.join(train_config['task_name'], 'cond_attr_samples', dir, current_time)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            
            # Save the image corresponding to the current timestep
            img.save(os.path.join(save_dir, 'x0_{}.png'.format(i)))
            img.close()

    # convert to torch tensor
    intermediate_latents = torch.stack(intermediate_latents, dim=0)

    # Return the final noisy latent z and the predicted noise used for the inversion
    return intermediate_latents


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
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
    model.eval()
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
        '''
        new_state_dict = OrderedDict()

        for k, v in vae_state_dict.items():
            if k.startswith('module.'):
                name = k[7:]
            new_state_dict[name] = v        
        '''
        new_state_dict = vae_state_dict
        vae.load_state_dict(new_state_dict, strict=True)
    else:
        raise Exception('VAE checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name'])))
    #####################################
    
    with torch.no_grad():
        ims, cond = sample(model, scheduler, train_config, diffusion_model_config,
                        autoencoder_model_config, diffusion_config, dataset_config, vae, use_ddim=sample_config['use_ddim'])

        ims_raw = ddim_inversion(scheduler, vae, ims, diffusion_config, cond, model, train_config, 500)
        
        sample(model, scheduler, train_config, diffusion_model_config, autoencoder_model_config, 
               diffusion_config, dataset_config, vae, use_ddim=sample_config['use_ddim'], noise_input=ims_raw)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation for class conditional '
                                                 'Mnist generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist_class_cond.yaml', type=str)
    args = parser.parse_args()
    infer(args)
