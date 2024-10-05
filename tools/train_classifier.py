import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from dataset.celeb_dataset import CelebDataset
from torch.utils.data.distributed import DistributedSampler
import os
from tqdm import tqdm
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
    wandb.init(project="Face-diffusion", entity="megleczmate", sync_tensorboard=True, tags=["classifier"])
    # get current time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    wandb.run.name = config['task_name'] + current_time
    wandb.run.save()
    wandb.config.update(config)
    return wandb

# Training function
def train_model(model, criterion, optimizer, condition_config, world_size, rank, num_epochs=25, wandb=None):
    # Path to CelebA dataset
    data_dir = 'data/CelebAMask-HQ/'

    # Load CelebA dataset with attribute labels
    image_datasets = {
        'train': CelebDataset(split='train',
                                    im_path=data_dir,
                                    im_size=256,
                                    im_channels=3,
                                    condition_config=condition_config,
                                    classification=True),
        'val': CelebDataset(split='val',
                                    im_path=data_dir,
                                    im_size=256,
                                    im_channels=3,
                                    condition_config=condition_config,
                                    classification=True),
    }

    samplers = {
        'train': DistributedSampler(image_datasets['train'], num_replicas=world_size, rank=rank, shuffle=True),
        'val': DistributedSampler(image_datasets['val'], num_replicas=world_size, rank=rank, shuffle=False)
    }

    # Data loaders
    batch_size = 96
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, sampler=samplers['train'], num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, sampler=samplers['val'], num_workers=4)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print(f'Training dataset size: {dataset_sizes["train"]}')
    print(f'Validation dataset size: {dataset_sizes["val"]}')
    
    
    step_count = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        val_accuracies_per_attribute = {attr: 0.0 for attr in condition_config['attribute_condition_config']['attribute_condition_selected_attrs']}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in tqdm(dataloaders[phase]):
                labels = labels['attribute'].float()

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

                        if (step_count % 200 == 0 or step_count == 1) and wandb is not None and rank == 0:
                            wandb.log({'train_loss': loss.item()})

                    else:
                        # Calculate accuracy per attribute
                        for attr_idx, attr in enumerate(condition_config['attribute_condition_config']['attribute_condition_selected_attrs']):
                            val_accuracies_per_attribute[attr] += torch.sum((outputs[:, attr_idx] > 0) == labels[:, attr_idx])

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



    return model


def main(rank, world_size):
    setup(rank, world_size)
    condition_config = {'task_name': 'celeba_attribute_classifier_resnet50_19_attributes',
                'condition_types': [ 'attribute' ],
                'attribute_condition_config': {
                    'attribute_condition_num': 3,
                    'attribute_condition_selected_attrs': ['Male', 'Young', 'Bald', 'Bangs', 'Receding_Hairline', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Straight_Hair', 'Wavy_Hair', 'No_Beard', 'Goatee', 'Mustache', 'Sideburns', 'Narrow_Eyes', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose'],
                    }
                }

    # Load pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)

    

    # Modify the fully connected layer to output 40 classes (for CelebA attributes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, condition_config['attribute_condition_config']['attribute_condition_num'])

    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model = DDP(model, device_ids=[rank])

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if rank == 0:
        wandb = setup_wandb(condition_config)
    # Train the model for 25 epochs
    model = train_model(model, criterion, optimizer, condition_config, world_size, rank, num_epochs=10, wandb=wandb)

    if rank == 0:
        # Save the model
        torch.save(model.state_dict(), 'celeba_resnet18_attribute_classifier.pth')

    if rank == 0:
        wandb.finish()
    
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

    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)

