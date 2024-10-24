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
from tools.sample_ddpm_attr_cond import sample, ddim_inversion
from torchvision.transforms import Compose, Normalize
from dataset.celeb_dataset import CelebDataset
from torchvision import models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# free gpu memory
torch.cuda.empty_cache()

import time
import numpy as np
import sklearn.linear_model as linear_model


def find_feature_axis(z, y, method='linear', **kwargs_model):
    """
    function to find axis in the latent space that is predictive of feature vectors

    :param z: vectors in the latent space, shape=(num_samples, num_latent_vector_dimension)
    :param y: feature vectors, shape=(num_samples, num_features)
    :param method: one of ['linear', 'logistic'], or a sklearn.linear_model object, (eg. sklearn.linear_model.ElasticNet)
    :param kwargs_model: parameters specific to a sklearn.linear_model object, (eg., penalty=’l2’)
    :return: feature vectors, shape = (num_latent_vector_dimension, num_features)
    """

    if method == 'linear':
        model = linear_model.LinearRegression(**kwargs_model)
        model.fit(z, y)
    elif method == 'tanh':
        def arctanh_clip(y):
            return np.arctanh(np.clip(y, np.tanh(-3), np.tanh(3)))

        model = linear_model.LinearRegression(**kwargs_model)

        model.fit(z, arctanh_clip(y))
    else:
        raise Exception('method has to be one of ["linear", "tanh"]')

    return model.coef_.transpose()


def normalize_feature_axis(feature_slope):
    """
    function to normalize the slope of features axis so that they have the same length

    :param feature_slope: array of feature axis, shape = (num_latent_vector_dimension, num_features)
    :return: same shape of input
    """

    feature_direction = feature_slope / np.linalg.norm(feature_slope, ord=2, axis=0, keepdims=True)
    return feature_direction


def disentangle_feature_axis(feature_axis_target, feature_axis_base, yn_base_orthogonalized=False):
    """
    make feature_axis_target orthogonal to feature_axis_base

    :param feature_axis_target: features axes to decorrerelate, shape = (num_dim, num_feature_0)
    :param feature_axis_base: features axes to decorrerelate, shape = (num_dim, num_feature_1))
    :param yn_base_orthogonalized: True/False whether the feature_axis_base is already othogonalized
    :return: feature_axis_decorrelated, shape = shape = (num_dim, num_feature_0)
    """

    # make sure this funciton works to 1D vector
    if len(feature_axis_target.shape) == 0:
        yn_single_vector_in = True
        feature_axis_target = feature_axis_target[:, None]
    else:
        yn_single_vector_in = False

    # if already othogonalized, skip this step
    if yn_base_orthogonalized:
        feature_axis_base_orthononal = orthogonalize_vectors(feature_axis_base)
    else:
        feature_axis_base_orthononal = feature_axis_base

    # orthogonalize every vector
    feature_axis_decorrelated = feature_axis_target + 0
    num_dim, num_feature_0 = feature_axis_target.shape
    num_dim, num_feature_1 = feature_axis_base_orthononal.shape
    for i in range(num_feature_0):
        for j in range(num_feature_1):
            feature_axis_decorrelated[:, i] = orthogonalize_one_vector(feature_axis_decorrelated[:, i],
                                                                       feature_axis_base_orthononal[:, j])

    # make sure this funciton works to 1D vector
    if yn_single_vector_in:
        result = feature_axis_decorrelated[:, 0]
    else:
        result = feature_axis_decorrelated

    return result


def disentangle_feature_axis_by_idx(feature_axis, idx_base=None, idx_target=None, yn_normalize=True):
    """
    disentangle correlated feature axis, make the features with index idx_target orthogonal to
    those with index idx_target, wrapper of function disentangle_feature_axis()

    :param feature_axis:       all features axis, shape = (num_dim, num_feature)
    :param idx_base:           index of base features (1D numpy array), to which the other features will be orthogonal
    :param idx_target: index of features to disentangle (1D numpy array), which will be disentangled from
                                    base features, default to all remaining features
    :param yn_normalize:       True/False to normalize the results
    :return:                   disentangled features, shape = feature_axis
    """

    (num_dim, num_feature) = feature_axis.shape

    # process default input
    if idx_base is None or len(idx_base) == 0:    # if None or empty, do nothing
        feature_axis_disentangled = feature_axis
    else:                                         # otherwise, disentangle features
        if idx_target is None:                # if None, use all remaining features
            idx_target = np.setdiff1d(np.arange(num_feature), idx_base)

        feature_axis_target = feature_axis[:, idx_target] + 0
        feature_axis_base = feature_axis[:, idx_base] + 0
        feature_axis_base_orthogonalized = orthogonalize_vectors(feature_axis_base)
        feature_axis_target_orthogonalized = disentangle_feature_axis(
            feature_axis_target, feature_axis_base_orthogonalized, yn_base_orthogonalized=True)

        feature_axis_disentangled = feature_axis + 0  # holder of results
        feature_axis_disentangled[:, idx_target] = feature_axis_target_orthogonalized
        feature_axis_disentangled[:, idx_base] = feature_axis_base_orthogonalized

    # normalize output
    if yn_normalize:
        feature_axis_out = normalize_feature_axis(feature_axis_disentangled)
    else:
        feature_axis_out = feature_axis_disentangled
    return feature_axis_out


def orthogonalize_one_vector(vector, vector_base):
    """
    tool function, adjust vector so that it is orthogonal to vector_base (i.e., vector - its_projection_on_vector_base )

    :param vector0: 1D array
    :param vector1: 1D array
    :return: adjusted vector1
    """
    return vector - np.dot(vector, vector_base) / np.dot(vector_base, vector_base) * vector_base


def orthogonalize_vectors(vectors):
    """
    tool function, adjust vectors so that they are orthogonal to each other, takes O(num_vector^2) time

    :param vectors: vectors, shape = (num_dimension, num_vector)
    :return: orthorgonal vectors, shape = (num_dimension, num_vector)
    """
    vectors_orthogonal = vectors + 0
    num_dimension, num_vector = vectors.shape
    for i in range(num_vector):
        for j in range(i):
            vectors_orthogonal[:, i] = orthogonalize_one_vector(vectors_orthogonal[:, i], vectors_orthogonal[:, j])
    return vectors_orthogonal


def plot_feature_correlation(feature_direction, feature_name=None):
    import matplotlib.pyplot as plt

    len_z, len_y = feature_direction.shape
    if feature_name is None:
        feature_name = range(len_y)

    feature_correlation = np.corrcoef(feature_direction.transpose())

    c_lim_abs = np.max(np.abs(feature_correlation))

    plt.pcolormesh(np.arange(len_y+1), np.arange(len_y+1), feature_correlation,
                   cmap='coolwarm', vmin=-c_lim_abs, vmax=+c_lim_abs)
    plt.gca().invert_yaxis()
    plt.colorbar()
    # plt.axis('square')
    plt.xticks(np.arange(len_y) + 0.5, feature_name, fontsize='x-small', rotation='vertical')
    plt.yticks(np.arange(len_y) + 0.5, feature_name, fontsize='x-small')
    plt.show()


def plot_feature_cos_sim(feature_direction, feature_name=None):
    """
    plot cosine similarity measure of vectors

    :param feature_direction: vectors, shape = (num_dimension, num_vector)
    :param feature_name:      list of names of features
    :return:                  cosines similarity matrix, shape = (num_vector, num_vector)
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity

    len_z, len_y = feature_direction.shape
    if feature_name is None:
        feature_name = range(len_y)

    feature_cos_sim = cosine_similarity(feature_direction.transpose())

    c_lim_abs = np.max(np.abs(feature_cos_sim))

    plt.pcolormesh(np.arange(len_y+1), np.arange(len_y+1), feature_cos_sim,
                   vmin=-c_lim_abs, vmax=+c_lim_abs, cmap='coolwarm')
    plt.gca().invert_yaxis()
    plt.colorbar()
    # plt.axis('square')
    plt.xticks(np.arange(len_y) + 0.5, feature_name, fontsize='x-small', rotation='vertical')
    plt.yticks(np.arange(len_y) + 0.5, feature_name, fontsize='x-small')
    plt.show()
    return feature_cos_sim

def flatten_latents(latents):
    # get shape of latents
    shape = latents.shape
    
    flattened_latents = torch.zeros(shape[0], shape[1] * shape[2] * shape[3])

    for i in range(shape[0]):
        flattened_latents[i] = latents[i].flatten()

    return flattened_latents

def reverse_flatten_latents(flattened_latents, shape):
    # get shape of latents
    shape = shape

    latents = torch.zeros(shape[0], shape[1], shape[2], shape[3])

    for i in range(shape[0]):
        latents[i] = flattened_latents[i].reshape(shape[1], shape[2], shape[3])

    return latents


def glasses_loss(x, classifier_model, device='cuda', target_label=1):
    # Create a resnet-18 model
    
    classifier_model.train()  # Ensure the model is in training mode

    # Move the input tensor `x` to the correct device
    x = x.to(device)

    transforms = Compose([
            #Normalize(mean=[-0.5047, -0.2201,  0.0777], std=[1.0066, 0.8887, 0.6669])
            #Normalize(mean=[-0.4908,  0.0627,  0.1011], std=[0.5076, 0.4108, 0.4806])
            Normalize(mean=[-0.5099,  0.0534,  0.0902], std=[0.4977, 0.4097, 0.4811]) # smiles
            #Normalize(mean=[-0.4932,  0.0717,  0.0965], std=[0.5124, 0.4079, 0.4835]) # chubby, mouth
        ])
    x = transforms(x)

    # Predict the glasses attribute
    pred = classifier_model(x)

    # Generate a target tensor with the same batch size as the input (assuming a binary classification task)
    if target_label == 0:
        target = torch.zeros(pred.size(0), 1).to(device)
    else:
        target = torch.ones(pred.size(0), 1).to(device)  # Assuming all targets are 1 (glasses present)

    # Calculate the loss using Binary Cross Entropy
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(pred, target)

    # Return the loss with gradients enabled
    return loss


def sample_guidance(model, classifier_model, cond, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, use_ddim=False, start_step=0, num_steps=1000, noise_input=None, dir='', target_label=1):
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

        loss = glasses_loss(x0_pred, classifier_model, target_label=target_label) * 0.05

        # set the loss to require grad
        
        #if i % 10 == 0:
        #    print(i, "loss:", loss.item())

        cond_grad = -torch.autograd.grad(loss, xt, retain_graph=True)[0] 

        # momentum
        if i < 770 and i > 200:
            cond_grad = cond_grad + 0.9 * cond_grad_prev        
        if i < 771:
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

oracle_config = {'task_name': 'celeba_attribute_classifier_resnet50_smile_attributes',
                'condition_types': [ 'attribute' ],
                'attribute_condition_config': {
                    'attribute_condition_num': 1,
                    'attribute_condition_selected_attrs': ['Smiling']
                    }
                }



oracle_classifier = models.resnet50(pretrained=False)

num_ftrs = oracle_classifier.fc.in_features

oracle_classifier.fc = torch.nn.Linear(num_ftrs, oracle_config['attribute_condition_config']['attribute_condition_num'])

# load checkpoint
checkpoint = torch.load('celeba_resnet50_smiling_classifier_1_oracle.pth')

new_state_dict = OrderedDict()

for k, v in checkpoint.items():
    if k.startswith('module.'):
        name = k[7:]
    new_state_dict[name] = v  

oracle_classifier.load_state_dict(new_state_dict)

oracle_classifier = oracle_classifier.cuda()

oracle_classifier.eval()


from models.simple_cnn import SimpleCNN

classifier_model = SimpleCNN()
#classifier_model.load_state_dict(torch.load('celeba_cnn_latent_glasses_classifier_0.pth', map_location=device))
classifier_model.load_state_dict(torch.load('celeba_cnn_latent_smile_classifier_0_700_ddpm.pth', map_location=device))



classifier_model.to(device)



im_dataset_val = CelebDataset(split='test',
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


dict = torch.load('/mnt/g/data/latents/700/raw/smiling/start_latents_attributes.pt')
latents = dict['latents']
attribute_desc = dict['attributes']

flattened_latents = flatten_latents(latents)

attribute_np = attribute_desc.cpu().numpy()


feature_slope = find_feature_axis(flattened_latents, attribute_np, method='tanh')

""" normalize the feature vectors """
yn_normalize_feature_direction = True
if yn_normalize_feature_direction:
    feature_direction = normalize_feature_axis(feature_slope)
else:
    feature_direction = feature_slope


len_z, len_y = feature_direction.shape

feature_direction_disentangled = disentangle_feature_axis_by_idx(
    feature_direction, idx_base=range(len_y - 1), idx_target=None, yn_normalize=True)

# swap the axes
feature_direction_disentangled = feature_direction_disentangled.T

# reshape the feature_direction_disentangled
feature_direction_disentangled = reverse_flatten_latents(torch.tensor(feature_direction_disentangled).to(device), (20, 3, 64, 64))


feature_direction_disentangled = torch.tensor(feature_direction_disentangled)[19]




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
    
    with torch.no_grad():
        ims_orig, cond_trans = sample(model, cond, scheduler, train_config, diffusion_model_config,
                    autoencoder_model_config, diffusion_config, dataset_config, vae, use_ddim=True, dir='', noise_input=noise_input, num_steps=1000, start_step=0)
        
        im_tensor = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])(ims_orig)
        
        im_tensor = (2 * im_tensor) - 1

        prediction = oracle_classifier(im_tensor.unsqueeze(0).to(device))

        prediction = torch.sigmoid(prediction)

        target_label = 1
        if prediction > 0.5:
            target_label = 0
        else:
            target_label = 1

        if target_label == 1:
            noise_input += feature_direction_disentangled.to(device) * 15
        else:
            noise_input -= feature_direction_disentangled.to(device) * 15

        ims_noise_edit, cond_trans = sample(model, cond, scheduler, train_config, diffusion_model_config,
                    autoencoder_model_config, diffusion_config, dataset_config, vae, use_ddim=True, dir='', noise_input=noise_input, num_steps=1000, start_step=0)
        


    ims_guidance, cond_trans = sample_guidance(model, classifier_model, cond, scheduler, train_config, diffusion_model_config,
                autoencoder_model_config, diffusion_config, dataset_config, vae, use_ddim=True, dir='', noise_input=noise_input, num_steps=1000, start_step=0, target_label=target_label)
    



    

    
    

    for j in range(ims_orig.shape[0]):
        img_j = ims_orig[j].permute(1, 2, 0).cpu().numpy()
        img_j = (img_j * 255).astype(np.uint8)
        img_j = Image.fromarray(img_j)
        img_j.save(os.path.join(save_dir, f'{image_num}_orig.png'))

        img_j = ims_noise_edit[j].permute(1, 2, 0).cpu().numpy()
        img_j = (img_j * 255).astype(np.uint8)
        img_j = Image.fromarray(img_j)
        img_j.save(os.path.join(save_dir, f'{image_num}_noise_edit.png'))

        img_j = ims_guidance[j].permute(1, 2, 0).cpu().numpy()
        img_j = (img_j * 255).astype(np.uint8)
        img_j = Image.fromarray(img_j)
        img_j.save(os.path.join(save_dir, f'{image_num}_guidance.png'))

        #write cond to file txt
        with open(os.path.join(save_dir, f'{image_num}.txt'), 'w') as f:
            f.write(str(attr[j].cpu().numpy()))

        image_num += 1

        

