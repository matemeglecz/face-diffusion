import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from dataset.celeb_dataset import CelebDataset
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
from collections import OrderedDict
from PIL import Image

condition_config = {'task_name': 'celeba_attribute_classifier_resnet50_all_attributes',
                'condition_types': [ 'attribute' ],
                'attribute_condition_config': {
                    'attribute_condition_num': 19,
                    'attribute_condition_selected_attrs': ['Male', 'Young', 'Bald', 'Bangs', 'Receding_Hairline', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Straight_Hair', 'Wavy_Hair', 'No_Beard', 'Goatee', 'Mustache', 'Sideburns', 'Narrow_Eyes', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose']
#['Eyeglasses',]# 'Heavy_Makeup', 'Smiling'],
                    }
                }

# Load pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)


# Modify the fully connected layer to output 40 classes (for CelebA attributes)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, condition_config['attribute_condition_config']['attribute_condition_num'])

# load checkpoint
checkpoint = torch.load('celeba_resnet50_all_classifier_oracle.pth')

new_state_dict = OrderedDict()

for k, v in checkpoint.items():
    if k.startswith('module.'):
        name = k[7:]
    new_state_dict[name] = v  

model.load_state_dict(new_state_dict)

model = model.cuda()

model.eval()


data_dir = 'data/CelebAMask-HQ/'

dataset = CelebDataset(split='test',
                                    im_path=data_dir,
                                    im_size=256,
                                    im_channels=3,
                                    condition_config=condition_config,
                                    classification=True)


dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

samples_dir = '/mnt/g/visuals_from_server/512-64-700-test'

# Initialize the prediction and target lists(tensors)
predictions = torch.tensor([])
targets = torch.tensor([])
idx = 0
# Iterate over the data
for inputs, labels in tqdm(dataloader, total=len(dataloader)):
    # load .png image from the samples directory
    im = Image.open(os.path.join(samples_dir, f'{idx}.png'))

    # apply the same transformations as in the dataset
    im_tensor = transforms.Compose([
                    transforms.Resize((256, 256)),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])(im)
    
    im.close()

    inputs = im_tensor = (2 * im_tensor) - 1
   
    inputs = inputs.unsqueeze(0).cuda()

    labels = labels['attribute'].cuda()

    # Forward pass
    outputs = model(inputs)
    # apply sigmoid to the output
    outputs = torch.sigmoid(outputs)

    labels = labels.cpu().detach()
    outputs = outputs.cpu().detach()

    # Append the predictions and targets
    predictions = torch.cat((predictions, outputs), dim=0)
    targets = torch.cat((targets, labels), dim=0)

    idx += 1

from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

# Convert predictions and targets to CPU for metric calculations
predictions = predictions.cpu().detach().numpy()
targets = targets.cpu().detach().numpy()

# Apply a threshold of 0.5 to the predictions to obtain binary labels
predictions_binary = (predictions >= 0.5).astype(int)

# Calculate accuracy, precision, and recall for each attribute
accuracy_per_attribute = []
precision_per_attribute = []
recall_per_attribute = []

for i in range(targets.shape[1]):
    accuracy_per_attribute.append(accuracy_score(targets[:, i], predictions_binary[:, i]))
    precision_per_attribute.append(precision_score(targets[:, i], predictions_binary[:, i], zero_division=1))
    recall_per_attribute.append(recall_score(targets[:, i], predictions_binary[:, i], zero_division=1))

# Calculate average metrics across all attributes
average_accuracy = np.mean(accuracy_per_attribute)
average_precision = np.mean(precision_per_attribute)
average_recall = np.mean(recall_per_attribute)

# Print the metrics
for i, attr in enumerate(condition_config['attribute_condition_config']['attribute_condition_selected_attrs']):
    print(f"Attribute: {attr}")
    print(f"  Accuracy: {accuracy_per_attribute[i]:.4f}")
    print(f"  Precision: {precision_per_attribute[i]:.4f}")
    print(f"  Recall: {recall_per_attribute[i]:.4f}\n")

print(f"Average Accuracy: {average_accuracy:.4f}")
print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")

# save the results to a file
with open('ddpm_700_test_results.txt', 'w') as f:
    f.write(f"Average Accuracy: {average_accuracy:.4f}\n")
    f.write(f"Average Precision: {average_precision:.4f}\n")
    f.write(f"Average Recall: {average_recall:.4f}\n\n")
    for i, attr in enumerate(condition_config['attribute_condition_config']['attribute_condition_selected_attrs']):
        f.write(f"Attribute: {attr}\n")
        f.write(f"  Accuracy: {accuracy_per_attribute[i]:.4f}\n")
        f.write(f"  Precision: {precision_per_attribute[i]:.4f}\n")
        f.write(f"  Recall: {recall_per_attribute[i]:.4f}\n\n")

