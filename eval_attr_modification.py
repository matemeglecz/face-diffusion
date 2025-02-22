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
from dataset.celeb_dataset import CelebDataset
from torch.utils.data import DataLoader
import numpy as np

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
    for i in reversed(range(num_steps - start_step)):
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
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, timestep, timestep_prev)  # Use DDIM sampling
        else:
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))  # Use DDPM sampling
       
        if i == 0:
            # Decode ONLY the final image to save time
            ims = vae.decode(xt)
        else:
            ims = x0_pred
        
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        
    return ims, cond_input


#with open('celebhq-512-64-train-komondor_b/celeba_komondor_512_b.yaml', 'r') as file:
with open('celebhq-512-64-train-komondor_b_2/celeba_komondor_512_b_2.yaml', 'r') as file:
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

sample_config['use_ddim'] = True

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
    
    
    new_state_dict = OrderedDict()

    for k, v in vae_state_dict.items():
        if k.startswith('module.'):
            name = k[7:]        
            new_state_dict[name] = v   

    #new_state_dict = vae_state_dict     
    
    vae.load_state_dict(new_state_dict, strict=False)
else:
    raise Exception('VAE checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                train_config['vqvae_autoencoder_ckpt_name'])))
#####################################

im_dataset_train = CelebDataset(split='test',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                use_latents=False,
                                latent_path=os.path.join(train_config['task_name'],
                                                         train_config['vqvae_latent_dir_name']),
                                condition_config=diffusion_model_config['condition_config'],
                                )




dataloader = DataLoader(im_dataset_train, batch_size=4, shuffle=False)


['Male', 'Young', 'Bald', 'Bangs', 'Receding_Hairline', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Straight_Hair', 'Wavy_Hair', 'No_Beard', 'Goatee', 'Mustache', 'Sideburns', 'Narrow_Eyes', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose']

def randomly_modify_attrubute(cond):
    # randomly modify the attributes
    num_attributes = cond.shape[1]


    hair_types = ['Bald', 'Straight_Hair', 'Wavy_Hair']
    hair_types_idx = [2, 9, 10]

    facial_hair = ['No_Beard', 'Goatee', 'Mustache', 'Sideburns']
    facial_hair_idx = [11, 12, 13, 14]

    hair_specific = ['Receding_Hairline', 'Bangs']
    hair_specific_idx = [4, 3]

    facial_features = ['Narrow_Eyes', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose']
    facial_features_idx = [15, 16, 17, 18]

    hair_color = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
    hair_color_idx = [5, 6, 7, 8]

    # choose a random attribute to modify
    attribute_idx = np.random.randint(0, num_attributes)

    cond[0, attribute_idx] = 1 - cond[0, attribute_idx]

    if attribute_idx == 0 and cond[0, attribute_idx] == 0:
        cond[0, 11] = 1
        cond[0, 12] = 0
        cond[0, 13] = 0
        cond[0, 14] = 0
    elif attribute_idx == 2 and cond[0, attribute_idx] == 1:
        cond[0, 9] = 0
        cond[0, 10] = 0
        for i in hair_color_idx:
            cond[0, i] = 0
        cond[0, 3] = 0
    
    elif attribute_idx in hair_types_idx:
        # make sure that one of the hair types is set to 1
        if cond[0, 2] == 0 and cond[0, 9] == 0 and cond[0, 10] == 0:
            # randomly choose one of the hair types
            hair_type = np.random.randint(0, 2) + 1
            cond[0, hair_types_idx[hair_type]] = 1
    elif attribute_idx in facial_hair_idx:
        # make sure that one of the facial hair is set to 1
        if cond[0, 11] == 0 and cond[0, 12] == 0 and cond[0, 13] == 0 and cond[0, 14] == 0:
            # randomly choose one of the facial hair
            facial_hair = np.random.randint(0, 3) + 1
            cond[0, facial_hair_idx[facial_hair]] = 1
        if cond[0, 11] == 1:
            cond[0, 12] = 0
            cond[0, 13] = 0
            cond[0, 14] = 0
    elif attribute_idx in hair_specific_idx:
        # make sure that maximum one of the hair specific is set to 1
        if cond[0, 3] == 1 and cond[0, 4] == 1:
            if attribute_idx == 3:
                cond[0, 4] = 0
            else:
                cond[0, 3] = 0
    elif attribute_idx in hair_color:
        # check if more than one hair color is set to 1
        hair_color_sum = torch.sum(cond[0, hair_color_idx])
        if hair_color_sum > 1 or hair_color_sum == 0:
            # randomly choose one of the hair colors to set to 1 and the rest to 0
            hair_color = np.random.randint(0, 3)
            for i in hair_color_idx:
                if i == hair_color_idx[hair_color]:
                    cond[0, i] = 1
                else:
                    cond[0, i] = 0

    return cond
            


save_dir = '/mnt/g/data/condition_edit_test/'

image_num = 0

images_all_syn = torch.tensor([])
images_all_real = torch.tensor([])
attribute_desc = torch.tensor([])

# seed random for reproducibility
torch.manual_seed(4)


# for each image in the training set create a ddim inversion
for i, (im_real, cond) in tqdm(enumerate(dataloader), total=len(dataloader)):

    # torch clear cache
    torch.cuda.empty_cache()
    im_real = im_real.to(device)
    attr = cond['attribute'].clone()    
    
    cond = cond
    cond['attribute'] = cond['attribute'].to(device)

    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    
    ########### Sample random noise latent ##########

    xt = torch.randn((1,
                    autoencoder_model_config['z_channels'],
                    im_size,
                    im_size)).to(device)    
    
    # repeat xt for the number of samples
    xt = xt.repeat(4, 1, 1, 1)


    with torch.no_grad():

        cond = torch.tensor(cond['attribute']).to(device)
        
        
        img, cond_input = sample(model, cond, scheduler, train_config, diffusion_model_config,
                                autoencoder_model_config, diffusion_config, dataset_config, vae, use_ddim=True, noise_input=xt)
        
        attribute_desc = torch.cat((attribute_desc, attr), 0)
        # save the images
        for j in range(img.shape[0]):
            # concatenate the images
            images_all_syn = torch.cat((images_all_syn, img[j].unsqueeze(0)), dim=0)

            im_real = im_real.detach().cpu()

            images_all_real = torch.cat((images_all_real, im_real[j].unsqueeze(0)), dim=0)

            img_j = img[j].permute(1, 2, 0).cpu().numpy()
            img_j = (img_j * 255).astype(np.uint8)
            img_j = Image.fromarray(img_j)
            img_j.save(os.path.join(save_dir, f'{image_num}.png'))

            # save attribute description
            attr = attr.cpu().numpy()
            with open(os.path.join(save_dir, f'{image_num}.txt'), 'w') as f:
                f.write(str(attr))


            image_num += 1
                
        



    