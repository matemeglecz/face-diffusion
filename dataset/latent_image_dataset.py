import os
import pickle
import glob
from torch.utils.data.dataset import Dataset
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize

#val_split_origins = ['latents_glasses_intermediates_39.pt', 'latents_glasses_intermediates_259.pt']

# 300 glasses
'''
val_split_origins = ['latents_glasses_intermediates_1204.pt', 
                     'latents_glasses_intermediates_1209.pt',                     
                     'latents_glasses_intermediates_1274.pt',
                     'latents_glasses_intermediates_1279.pt',]
'''

# 700 glasses
'''
val_split_origins = ['latents_glasses_intermediates_4.pt',
                        'latents_glasses_intermediates_9.pt',
                        'latents_glasses_intermediates_194.pt',
                        'latents_glasses_intermediates_199.pt']
'''
'''
val_split_origins = ['latents_mouth_intermediates_4.pt',
                        'latents_mouth_intermediates_4_2.pt',
                        'latents_mouth_intermediates_9.pt',
                        'latents_mouth_intermediates_9_2.pt',
                        'latents_mouth_intermediates_194.pt',
                        'latents_mouth_intermediates_194_2.pt',
                        'latents_mouth_intermediates_199.pt',
                        'latents_mouth_intermediates_199_2.pt']
'''
'''
val_split_origins = ['latents_chubby_intermediates_4.pt',
                        'latents_chubby_intermediates_9.pt',
                        'latents_chubby_intermediates_194.pt',
                        'latents_chubby_intermediates_199.pt']
'''
val_split_origins = ['latents_nose_intermediates_4.pt',
                     'latents_nose_intermediates_9.pt',
                        'latents_nose_intermediates_194.pt',
                        'latents_nose_intermediates_199.pt']

class LatentImageDataset(Dataset):
    r"""
    Latent dataset will load the latents from the latent_path
    """
    def __init__(self, latent_path, split='all', target_attributes=None):
        self.latent_path = latent_path
        self.split = split


        self.latents, self.attributes_df = self.load_latents()


        if target_attributes is not None:
            self.attributes_df = self.attributes_df[['file_name'] + target_attributes]


        self.attributes = self.attributes_df.drop(columns=['file_name']).values.astype(np.float32)        
        
                                            

    def load_latents(self):
        r"""
        Load the latents from the latent path
        :return:
        """

        assert os.path.exists(self.latent_path), "images path {} does not exist".format(self.latent_path)

        attributes_df = pd.read_csv(os.path.join(self.latent_path, 'attributes.csv'))

        attributes_df = attributes_df[attributes_df['file_name'] != 'file_name']

        attributes_df['origin_name'] = attributes_df['origin_name'].apply(lambda x: x.split('/')[-1])

        fnames = []

        if self.split == 'val':
            attributes_df = attributes_df[attributes_df['origin_name'].isin(val_split_origins)]
            for idx, row in attributes_df.iterrows():
                fnames.append(row['file_name'])
        elif self.split == 'train':
            attributes_df = attributes_df[~attributes_df['origin_name'].isin(val_split_origins)]
            for idx, row in attributes_df.iterrows():
                #if idx % 4 == 0:
                fnames.append(row['file_name'])
        else:
            for idx, row in attributes_df.iterrows():
                fnames.append(row['file_name'])
        

        attributes_df = attributes_df.drop(columns=['origin_name'])

        return fnames, attributes_df


    def __getitem__(self, index):
        latent = self.latents[index]                

        # load the image
        latent = np.load(os.path.join(self.latent_path, latent))

        '''
        attribute = self.attributes[self.attributes['file_name'] == latent]

        attribute = attribute.drop(columns=['file_name']).values

        # change the type to float        
        attribute = attribute.astype(np.float32)        
        '''
        attribute = self.attributes[index]


        latent = latent.astype(np.float32)

        # to tensor
        latent = torch.from_numpy(latent).float()
        attribute = torch.from_numpy(attribute)

        
        # normalize the latent with these means: tensor([-0.5047, -0.2201,  0.0777]) and stds: tensor([1.0066, 0.8887, 0.6669])
        transforms = Compose([
            #Normalize(mean=[-0.5047, -0.2201, 0.0777], std=[1.0066, 0.8887, 0.6669])
            #Normalize(mean=[-0.5047, -0.2201,  0.0777], std=[1.0066, 0.8887, 0.6669])
            #tensor([-0.4908,  0.0627,  0.1011]) tensor([0.5076, 0.4108, 0.4806])
            #Normalize(mean=[-0.4908,  0.0627,  0.1011], std=[0.5076, 0.4108, 0.4806]) # glasses
            #Normalize(mean=[-0.5099,  0.0534,  0.0902], std=[0.4977, 0.4097, 0.4811]) # smiles
            #Normalize(mean=[-0.4932,  0.0717,  0.0965], std=[0.5124, 0.4079, 0.4835]) # chubby            
            Normalize(mean=[-0.5027,  0.0582,  0.0977], std=[0.4988, 0.4093, 0.4814]) # nose
        ])
        latent = transforms(latent)
        

        attribute = attribute.float()

        #attribute = attribute.squeeze(0)


        return latent, attribute

    def __len__(self):
        return len(self.latents)