import glob
import os
import random
import torch
import torchvision
import numpy as np
from PIL import Image
from utils.diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import pandas as pd


class CelebDataset(Dataset):
    r"""
    Celeb dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """
    
    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='jpg',
                 use_latents=False, latent_path=None, condition_config=None, classification=False):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False
        self.classification = classification
        
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        
        self.idx_to_cls_map = {}
        self.cls_to_idx_map ={}
        
        if 'image' in self.condition_types:
            self.mask_channels = condition_config['image_condition_config']['image_condition_input_channels']
            self.mask_h = condition_config['image_condition_config']['image_condition_h']
            self.mask_w = condition_config['image_condition_config']['image_condition_w']

        if 'attribute' in self.condition_types:
            self.attribute_num = condition_config['attribute_condition_config']['attribute_condition_num']
            self.selected_attrs = condition_config['attribute_condition_config']['attribute_condition_selected_attrs']

            # load the attribute file
            attr_path = os.path.join(im_path, 'CelebAMask-HQ-attribute-anno.txt')
            attr_df = pd.read_csv(attr_path, sep='\s+', header=1)
            # replace -1 with 0
            attr_df = attr_df.replace(-1, 0)
            self.attr_df = attr_df[self.selected_attrs]
            
        self.images, self.texts, self.masks, self.attributes = self.load_images(im_path)
        
        # Whether to load images or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')
    
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        fnames = glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('png')))
        fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpg')))
        fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpeg')))
        texts = []
        masks = []
        attributes = []
        
        if 'image' in self.condition_types:
            label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                          'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
            self.idx_to_cls_map = {idx: label_list[idx] for idx in range(len(label_list))}
            self.cls_to_idx_map = {label_list[idx]: idx for idx in range(len(label_list))}
        
        # sort the files to get the same order
        fnames = sorted(fnames)

        for fname in tqdm(fnames):
            ims.append(fname)
            
            if 'text' in self.condition_types:
                im_name = os.path.split(fname)[1].split('.')[0]
                captions_im = []
                with open(os.path.join(im_path, 'celeba-caption/{}.txt'.format(im_name))) as f:
                    for line in f.readlines():
                        captions_im.append(line.strip())
                texts.append(captions_im)
                
            if 'image' in self.condition_types:
                im_name = int(os.path.split(fname)[1].split('.')[0])
                masks.append(os.path.join(im_path, 'CelebAMask-HQ-mask', '{}.png'.format(im_name)))

            if 'attribute' in self.condition_types:
                im_name = int(os.path.split(fname)[1].split('.')[0])
                attr = self.attr_df.iloc[im_name]
                attr = attr.to_numpy()
                attributes.append(attr)

        if 'text' in self.condition_types:
            assert len(texts) == len(ims), "Condition Type Text but could not find captions for all images"
        if 'image' in self.condition_types:
            assert len(masks) == len(ims), "Condition Type Image but could not find masks for all images"
        if 'attribute' in self.condition_types:
            assert len(attributes) == len(ims), "Condition Type Attribute but could not find attributes for all images"

        print('Found {} images'.format(len(ims)))
        print('Found {} masks'.format(len(masks)))
        print('Found {} captions'.format(len(texts)))
        print('Found {} attributes'.format(len(attributes)))

        # split the data into train, test and validation
        if self.split == 'train':
            ims = ims[:int(0.9*len(ims))]
            texts = texts[:int(0.9*len(texts))]
            masks = masks[:int(0.9*len(masks))]
            attributes = attributes[:int(0.9*len(attributes))]
        elif self.split == 'test':
            ims = ims[int(0.9*len(ims)):int(0.98*len(ims))]
            texts = texts[int(0.9*len(texts)):int(0.98*len(ims))]
            masks = masks[int(0.9*len(masks)):int(0.98*len(ims))]
            attributes = attributes[int(0.9*len(attributes)):int(0.98*len(attributes))]
        else:
            ims = ims[int(0.98*len(ims)):]
            texts = texts[int(0.98*len(texts)):]
            masks = masks[int(0.98*len(masks)):]
            attributes = attributes[int(0.98*len(attributes)):]


        return ims, texts, masks, attributes

    def get_mask(self, index):
        r"""
        Method to get the mask of WxH
        for given index and convert it into
        Classes x W x H mask image
        :param index:
        :return:
        """
        mask_im = Image.open(self.masks[index])
        mask_im = np.array(mask_im)
        im_base = np.zeros((self.mask_h, self.mask_w, self.mask_channels))
        for orig_idx in range(len(self.idx_to_cls_map)):
            im_base[mask_im == (orig_idx+1), orig_idx] = 1
        mask = torch.from_numpy(im_base).permute(2, 0, 1).float()
        return mask
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if 'text' in self.condition_types:
            cond_inputs['text'] = random.sample(self.texts[index], k=1)[0]
        if 'image' in self.condition_types:
            mask = self.get_mask(index)
            cond_inputs['image'] = mask
        if 'attribute' in self.condition_types:
            cond_inputs['attribute'] = self.attributes[index]

        #######################################
        
        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs    
        else:           
            im = Image.open(self.images[index])

            if not self.classification:
                im_tensor = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(self.im_size),
                    torchvision.transforms.CenterCrop(self.im_size),
                    torchvision.transforms.ToTensor(),
                ])(im)
            else:
                im_tensor = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])(im)

            im.close()
        
            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs
