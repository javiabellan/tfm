"""
TODO

1. precompute=True
2. Use lr_find() to find highest learning rate where loss is still clearly improving
3. Train last layer from precomputed activations for 1-2 epochs
4. Train last layer with data augmentation (i.e. precompute=False) for 2-3 epochs with cycle_len=1
5. Unfreeze all layers
6. Set earlier layers to 3x-10x lower learning rate than next higher layer
7. Use lr_find() again
8. Train full network with cycle_mult=2 until over-fitting





AUGMENTATIONS

De posicion (dihedral)
	1. Horizontal Flip
	2. Vertical Flip    
	3. 90 Rotation
Position
	4. Scale
	5. Crop
	6. Translation
	7. Small rotation
Filter
	brightness
	contrast
	Gaussian Noise
To feed the net
	Normalizar
	Totensor
"""
transforms.RandomHorizontalFlip
transforms.RandomVerticalFlip

transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ts.transforms.Rotate(20), # data augmentation: rotation 
            ts.transforms.Rotate(-20), # data augmentation: rotation
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


#################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import PIL
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


########################################################### Hyperparameters

batch_size      = 256
val_percent     = 0.3
num_workers     = 4

model           = resnet18
pretrained      = True
epochs          = 90
learning_rate   = 0.1
momentum        = 0.9
weight_decay    = 1e-4
print-freq      = 10

gpu             = True
multiple_gpus   = False
check_point     = args.cp
trained_model   = args.pt
verbose         = args.v



########################################################### Data

"""
- 2000 images, format: JPG
  - 374 of **malignant** skin tumors: *Melanoma*
  - 1626 of **benign** skin tumors:
    - 254 of *Seborrheic Keratosis*
    - 1372 of *Nevus*
"""

csv_file = pathlib.Path("C:/Users/Javi/Desktop/tfm/Datasets/ISIC-2017/ISIC2017_GroundTruth.csv")
data_dir = pathlib.Path("D:/Datasets/TFM/ISIC-2017_Training_Data")


class skinDataset(Dataset):

	def __init__(self, csv_file, data_dir):
		self.labels      = pd.read_csv(csv_file)
		self.data_dir    = data_dir
		self.center_crop = transforms.CenterCrop(100)
		self.to_tensor   = transforms.ToTensor()

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		img_name = data_dir / (self.labels.iloc[idx, 0] + ".jpg")
		image    = PIL.Image.open(img_name)
		image    = self.center_crop(image)
		image    = self.to_tensor(image)

		label    = self.labels.iloc[idx, 1:3].values.tolist()
		if   label[0]==1.0: label = torch.tensor([1,0,0]) #"melanoma"
		elif label[1]==1.0: label = torch.tensor([0,1,0]) #"seborrheic_keratosis"
		else:               label = torch.tensor([0,0,1]) #"healthy"

		return {"image": image, "label": label}

dataset    = skinDataset(csv_file, data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
train_size = len(dataset)



def imshow(sample):
	img = sample["image"]
	img = img.numpy().transpose((1, 2, 0))
	img = np.clip(img, 0, 1)
	plt.imshow(img)

	label = sample["label"]
	if   lbl.numpy()[0]==1: label = "melanoma"
	elif lbl.numpy()[1]==1: label = "seborrheic_keratosis"
	else:                   label = "healthy"
	plt.title(label)
	
	plt.pause(0.001)  # pause a bit so that plots are updated

imshow(dataset[3])

# Get a batch of training data
inputs, classes = next(iter(dataloader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])