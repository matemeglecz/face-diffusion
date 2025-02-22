from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import torch
from tqdm import tqdm

data_dir = '/mnt/g/visuals_from_server/512-64-700-test'


fid = FrechetInceptionDistance(feature=2048, normalize=True)
kid = KernelInceptionDistance(feature=2048, normalize=True, subset_size = 64)

real_images = torch.load(f'{data_dir}/celeba_test_real_images.pt')
synthetic_images = torch.load(f'{data_dir}/celeba_test_images.pt')

real_images_2 = torch.load(f'{data_dir}/celeba_test_real_images_2.pt')
synthetic_images_2 = torch.load(f'{data_dir}/celeba_test_images_2.pt')

# concatenate the two sets of images
real_images = torch.cat((real_images, real_images_2))
synthetic_images = torch.cat((synthetic_images, synthetic_images_2))

print(real_images.shape)
print(synthetic_images.shape)

real_images = (real_images + 1) / 2


batch_size = 64

# update with real images in batch sizes
for i in tqdm(range(0, len(real_images), batch_size)):
    real_batch = real_images[i:i+batch_size]
    fid.update(real_batch, real=True)
    kid.update(real_batch, real=True)

# update with synthetic images in batch sizes
for i in tqdm(range(0, len(synthetic_images), batch_size)):
    synthetic_batch = synthetic_images[i:i+batch_size]
    fid.update(synthetic_batch, real=False)
    kid.update(synthetic_batch, real=False)


fid_score = fid.compute()
kid_score = kid.compute()


print(f"FrechetInceptionDistance: {fid_score}")
print(f"KernelInceptionDistance: {kid_score}")