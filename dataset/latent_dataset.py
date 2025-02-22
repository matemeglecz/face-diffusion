import os
import pickle
import glob
from torch.utils.data.dataset import Dataset
import torch


class LatentDataset(Dataset):
    r"""
    Latent dataset will load the latents from the latent_path
    """
    def __init__(self, latent_path, split='all', target_attribute_idx=None):
        self.latent_path = latent_path
        self.latents, self.attributes = self.load_latents()

        # seed random
        torch.manual_seed(9)

        # shuffle the latents
        #idx = torch.randperm(len(self.latents))
        #self.latents = self.latents[idx]
        #self.attributes = self.attributes[idx]

        if split == 'train':
            self.latents = self.latents[:int(0.8 * len(self.latents))]
            self.attributes = self.attributes[:int(0.8 * len(self.attributes))]
        elif split == 'val':
            self.latents = self.latents[int(0.8 * len(self.latents)):int(0.9 * len(self.latents))]
            self.attributes = self.attributes[int(0.8 * len(self.attributes)):int(0.9 * len(self.attributes))]
        elif split == 'test':
            self.latents = self.latents[int(0.9 * len(self.latents)):]
            self.attributes = self.attributes[int(0.9 * len(self.attributes)):]
        else:
            pass

        if target_attribute_idx is not None:
            self.attributes = self.attributes[:, target_attribute_idx]

        # add a channel to the attributes
        self.attributes = torch.unsqueeze(self.attributes, 1)

    def load_latents(self):
        r"""
        Load the latents from the latent path
        :return:
        """

        latents, attributes = torch.tensor([]), torch.tensor([])
        
        # check if the latent path is a directory
        if os.path.isdir(self.latent_path):
            for fname in glob.glob(os.path.join(self.latent_path, '*.pt')):
                print('Loading latent from {}'.format(fname))
                s = torch.load(open(fname, 'rb'))
                for k, v in s.items():
                    if k == 'x0':
                        latents = torch.cat((latents, v), dim=1)
                    elif k == 'attribute_desc':
                        attributes = torch.cat((attributes, v), dim=0)

        else:
            s = torch.load(open(self.latent_path, 'rb'))
            for k, v in s.items():
                if k == 'x0':
                    latents = v
                elif k == 'attribute_desc':
                    attributes = v

  

        # swap the dimensions 0 and 1
        latents = latents.permute(1, 0, 2, 3, 4)

        unstacked_latents = torch.tensor([])
        unstacked_attributes = torch.tensor([])
        for i in range(latents.shape[0]):
            # extend the latents
            unstacked_latents = torch.cat((unstacked_latents, latents[i]), dim=0)
            # extend the attributes
            # add shape[1] number of attributes
            temp = torch.unsqueeze(attributes[i], 0).repeat(latents.shape[1], 1)
            unstacked_attributes = torch.cat((unstacked_attributes, temp), dim=0)


        print(unstacked_latents.shape, unstacked_attributes.shape)
        
        return unstacked_latents, unstacked_attributes
    



    def __getitem__(self, index):
        return self.latents[index], self.attributes[index]

    def __len__(self):
        return len(self.latents)