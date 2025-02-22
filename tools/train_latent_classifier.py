import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from dataset.latent_dataset import LatentDataset
from dataset.latent_image_dataset import LatentImageDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
import os
from tqdm import tqdm
import wandb
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import socket
from mpi4py import MPI
import pickle

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
    wandb.init(project="Face-diffusion", entity="megleczmate", sync_tensorboard=True, tags=["latent_classifier"])
    # get current time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    wandb.run.name = config['task_name'] + current_time
    wandb.run.save()
    wandb.config.update(config)
    return wandb

# Training function
def train_model(model, criterion, optimizer, condition_config, world_size, rank, num_epochs=25, wandb=None):
    # Path to CelebA dataset
    data_dir = '/mnt/c/latents/250_raw'

    if rank == 0:
        wandb = setup_wandb(condition_config)

    # Load CelebA dataset with attribute labels
    image_datasets = {
        'train': LatentImageDataset(data_dir,
                                split='train',
                                target_attributes=['Eyeglasses']),
        'val': LatentImageDataset(data_dir,
                                split='val',
                                target_attributes=['Eyeglasses']),
    }
    
    samplers = {
        'train': DistributedSampler(image_datasets['train'], num_replicas=world_size, rank=rank, shuffle=True),
        'val': DistributedSampler(image_datasets['val'], num_replicas=world_size, rank=rank, shuffle=False)
    }

    # Data loaders
    batch_size = 1024
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, sampler=samplers['train'], num_workers=8, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, sampler=samplers['val'], num_workers=8, pin_memory=True)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print(f'Training dataset size: {dataset_sizes["train"]}')
    print(f'Validation dataset size: {dataset_sizes["val"]}')
    
    
    step_count = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        val_accuracies_per_attribute = {attr: 0.0 for attr in condition_config['attribute_condition_config']['attribute_condition_selected_attrs']}
        val_tps_per_attribute = {attr: 0.0 for attr in condition_config['attribute_condition_config']['attribute_condition_selected_attrs']}
        val_fps_per_attribute = {attr: 0.0 for attr in condition_config['attribute_condition_config']['attribute_condition_selected_attrs']}
        val_fns_per_attribute = {attr: 0.0 for attr in condition_config['attribute_condition_config']['attribute_condition_selected_attrs']}
        val_tns_per_attribute = {attr: 0.0 for attr in condition_config['attribute_condition_config']['attribute_condition_selected_attrs']}
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in tqdm(dataloaders[phase]):
                labels = labels.float()                

                inputs = inputs.to(rank)
                labels = labels.float().to(rank)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):                    
                    outputs = model(inputs)



                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        step_count += 1

                        if (step_count % 10 == 0 or step_count == 1) and wandb is not None and rank == 0:
                            wandb.log({'train_loss': loss.item()})

                    else:
                        # Calculate accuracy per attribute
                        for attr_idx, attr in enumerate(condition_config['attribute_condition_config']['attribute_condition_selected_attrs']):
                            predicted_labels = torch.sigmoid(outputs) > 0.5

                            val_accuracies_per_attribute[attr] += torch.sum(predicted_labels[:, attr_idx] == labels[:, attr_idx])
                            # convert to int to avoid overflow
                            predicted_labels = predicted_labels.int()
                            labels = labels.int()

                            val_tps_per_attribute[attr] += torch.sum(predicted_labels[:, attr_idx] & labels[:, attr_idx])
                            val_fps_per_attribute[attr] += torch.sum(predicted_labels[:, attr_idx] & ~labels[:, attr_idx])
                            val_fns_per_attribute[attr] += torch.sum(~predicted_labels[:, attr_idx] & labels[:, attr_idx])
                            val_tns_per_attribute[attr] += torch.sum(~predicted_labels[:, attr_idx] & ~labels[:, attr_idx])


                running_loss += loss.item() * inputs.size(0)

                

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')            

            if phase == 'val' and rank == 0:
                # calculate accuracy per attribute
                for attr in condition_config['attribute_condition_config']['attribute_condition_selected_attrs']:
                    print(f'{attr} Accuracy: {val_accuracies_per_attribute[attr] / dataset_sizes[phase]:.4f}')

                if wandb is not None:
                    wandb.log({f'{phase}_loss': epoch_loss})
                    for attr in condition_config['attribute_condition_config']['attribute_condition_selected_attrs']:
                        wandb.log({f'{attr}_accuracy': val_accuracies_per_attribute[attr] / dataset_sizes[phase]})
                        #log precision, recall, f1
                        precision = val_tps_per_attribute[attr] / (val_tps_per_attribute[attr] + val_fps_per_attribute[attr])
                        recall = val_tps_per_attribute[attr] / (val_tps_per_attribute[attr] + val_fns_per_attribute[attr])
                        f1 = 2 * precision * recall / (precision + recall)
                        wandb.log({f'{attr}_precision': precision})
                        wandb.log({f'{attr}_recall': recall})
                        wandb.log({f'{attr}_f1': f1})

            if rank == 0:
                # Save the model
                torch.save(model.state_dict(), f'celeba_resnet18_latent_glasses_classifier_{epoch}.pth')

    if rank == 0:
        wandb.finish()

    return model


def main(rank, world_size):
    setup(rank, world_size)
    condition_config = {'task_name': 'celeba_attribute_classifier_resnet50_1_attributes',
                'condition_types': [ 'attribute' ],
                'attribute_condition_config': {
                    'attribute_condition_num': 1,
                    'attribute_condition_selected_attrs': ['Eyeglasses',]# 'Heavy_Makeup', 'Smiling'],
                    }
                }

    # Load pre-trained ResNet-18 model
    model = models.resnet18(pretrained=False)

    

    # Modify the fully connected layer to output 40 classes (for CelebA attributes)
    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, condition_config['attribute_condition_config']['attribute_condition_num'])

    '''
    # Freeze all layers except the last 4 layers
    for name, param in model.named_parameters():
        if 'layer4' not in name:
            param.requires_grad = False
    '''
            
    # Move model to GPU if available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(rank)

    model = DDP(model, device_ids=[rank])

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model for 25 epochs
    model = train_model(model, criterion, optimizer, condition_config, world_size, rank, num_epochs=3, wandb=None)

    
    
    cleanup()

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
    
    comm = MPI.COMM_WORLD

    hostname = socket.gethostbyname(socket.getfqdn())

    world_size = torch.cuda.device_count()  # Number of GPUs available

    print(world_size)

    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)

    print(hostname, port)

    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)

