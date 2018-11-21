"""
TODO
====

- Crear documentacion con https://docs.readthedocs.io
"""
import torch
import torchvision
from torchvision import transforms
import albumentations as aug

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from fastprogress import master_bar, progress_bar
import pathlib
import PIL
from tqdm import tqdm
import time
import os
import copy
print("AI framework by Javi based in PyTorch:",torch.__version__)



################  _    _ _   _ _     
################ | |  | | | (_) |    
################ | |  | | |_ _| |___ 
################ | |  | | __| | / __|
################ | |__| | |_| | \__ \
################  \____/ \__|_|_|___/
################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Timer():
	def __init__(self):
		self.times = [time.time()]
		self.total_time = 0.0

	def __call__(self, include_in_total=True):
		self.times.append(time.time())
		dt = self.times[-1] - self.times[-2]
		if include_in_total:
			self.total_time += dt
		return dt


################    _____        _                                                  
################   |  __ \      | |                                                 
################   | |  | | __ _| |_ __ _     _ __  _ __ ___ _ __  _ __ ___   ___   
################   | |  | |/ _` | __/ _` |   | '_ \| '__/ _ \ '_ \| '__/ _ \ / __|  
################   | |__| | (_| | || (_| |   | |_) | | |  __/ |_) | | | (_) | (__.
################   |_____/ \__,_|\__\__,_|   | .__/|_|  \___| .__/|_|  \___/ \___(_)
################                             | |            | |                     
################                             |_|            |_|                     


def normalise(x, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def padding(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 



################                                             _        _   _             
################       /\                                   | |      | | (_)            
################      /  \  _   _  __ _ _ __ ___   ___ _ __ | |_ __ _| |_ _  ___  _ __  
################     / /\ \| | | |/ _` | '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \ 
################    / ____ \ |_| | (_| | | | | | |  __/ | | | || (_| | |_| | (_) | | | |
################   /_/    \_\__,_|\__, |_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|
################                   __/ |                                                
################                  |___/                                                 
################
################  Augmentations provided by albumentations
################  https://github.com/albu/albumentations/blob/master/albumentations/augmentations/transforms.py

"""

########## Por clasificar

'PadIfNeeded',        # Pad side of the image / max if side is less than desired number.
'Cutout',             # CoarseDropout of the square regions in the image.
'ToFloat',            # Divide pixel values by max_value to get a float32 output array where all values lie in the range [0, 1.0]. If max_value is None the transform will try to infer the maximum value by inspecting the data type of the input image.
'FromFloat',          # Take an input array where all values should lie in the range [0, 1.0], multiply them by max_value and then cast the resulted value to a type specified by dtype. If max_value is None the transform will try to infer the maximum value for the data type from the dtype argument.
'LongestMaxSize',     # Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.
'SmallestMaxSize',    # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.

########## Filter augmentations

'Normalize',          # Divide pixel values by 255 = 2**8 - 1, subtract mean per channel and divide by std per channel.
'RGBShift',           # Randomly shift values for each channel of the input RGB image.
'InvertImg',          # Invert the input image by subtracting pixel values from 255.
'HueSaturationValue', # Randomly change hue, saturation and value of the input image.
'ChannelShuffle',     # Randomly rearrange channels of the input RGB image.
'CLAHE',              # Apply Contrast Limited Adaptive Histogram Equalization to the input image.
'RandomContrast',
'RandomGamma',
'RandomBrightness',
'Blur',               # Blur the input image using a random-sized kernel.
'MedianBlur',         # Blur the input image using using a median filter with a random aperture linear size.
'MotionBlur',         # Apply motion blur to the input image using a random-sized kernel.
'GaussNoise',         # Apply gaussian noise to the input image.
'ToGray',             # Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater than 127, invert the resulting grayscale image.
'JpegCompression',    # Decrease Jpeg compression of an image.

########## Non destructive augmentations (Dehidral group D4)

'VerticalFlip',      # Flip the input vertically around the x-axis.
'HorizontalFlip',    # Flip the input horizontally around the y-axis.
'Flip',              # Flip the input either horizontally, vertically or both horizontally and vertically.
'RandomRotate90',    # Randomly rotate the input by 90 degrees zero or more times.
'Transpose',         # Transpose the input by swapping rows and columns.

########## Move augmentations (View position zoom rotation)

'ShiftScaleRotate',  # Randomly apply affine transforms: translate, scale and rotate the input.
'RandomSizedCrop'    # Crop a random part of the input and rescale it to some size.
'RandomCrop',        # Crop a random part of the input.
'Rotate',            # Rotate the input by an angle selected randomly from the uniform distribution.
'CenterCrop',        # Crop the central part of the input.
'Crop',              # Crop region from image.
'RandomScale',       # Randomly resize the input. Output image size is different from the input image size.
'Resize',            # Resize the input to the given height and width.

########## Non-rigid transformations augmentations

'GridDistortion',
'ElasticTransform',
'OpticalDistortion',



Blur(blur_limit=7, p=0.5)  

VerticalFlip(p=0.5)        
HorizontalFlip(p=0.5)      
Flip(p=0.5)                
Transpose(p=0.5)           


RandomCrop(height, width, p=1.0)      
RandomGamma(gamma_limit=(80, 120), p=0.5)
RandomRotate90(p=0.5)      
Normalize(mean=(0.485, 0.456, 0.406), 
	      std=(0.229, 0.224, 0.225),
	      max_pixel_value=255.0, p=1.0)



###############################

# Aerial and medicine image
aug.Flip(p=0.5)
aug.RandomRotate90(p=0.5)


# Picture image
aug.HorizontalFlip(p=0.5)


def augment_flips_color(p=.5):
	return aug.Compose([
		aug.CLAHE(),          # Apply Contrast Limited Adaptive Histogram Equalization to the input image.
		aug.RandomRotate90(),
		aug.Transpose(),
		aug.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
		aug.Blur(blur_limit=3),
		aug.OpticalDistortion(),
		aug.GridDistortion(),
		aug.HueSaturationValue()
	], p=p)

sampleAug = aug.OneOf([
	aug.CLAHE(clip_limit=2),
	aug.IAASharpen(),
	aug.RandomRotate90(),
	aug.IAAEmboss(),
	aug.Transpose(),
	aug.RandomContrast(),
	aug.RandomBrightness(),
], p=0.3)
"""

################    _____        _                 _   
################   |  __ \      | |               | |  
################   | |  | | __ _| |_ __ _ ___  ___| |_ 
################   | |  | |/ _` | __/ _` / __|/ _ \ __|
################   | |__| | (_| | || (_| \__ \  __/ |_ 
################   |_____/ \__,_|\__\__,_|___/\___|\__|
################


class ImageDataset(torch.utils.data.Dataset):

	def __init__(self, image_dir, images, labels, labels_map, transforms=False, limit=False):
		self.image_dir  = image_dir        
		self.images     = images
		self.labels     = labels
		self.labels_map = labels_map
		self.transforms = transforms
		self.limit      = limit  

	def __len__(self):
		return len(self.labels) if not self.limit else self.limit

	def __getitem__(self, idx):
		img_name = self.image_dir / self.images[idx]
		image = PIL.Image.open(img_name)
		if self.transforms: image = self.transforms(image)
		label = self.labels[idx]
		return image, label
    

	def get_dataloader(self, dataset, batch_size=1, balance=False, shuffle=False, num_workers=0, pin_memory=True, drop_last=False):
		
		if balance:
			sampler = self.get_balanced_sampler()
			shuffle = False
		else:
			sampler = None

		return torch.utils.data.DataLoader(dataset     = self,
		                                   batch_size  = batch_size,
		                                   shuffle     = shuffle,
		                                   sampler     = sampler,
		                                   num_workers = num_workers,
		                                   pin_memory  = pin_memory,
		                                   drop_last   = drop_last)

	def get_balanced_sampler(self):
		class_counts  = np.bincount(self.labels)
		class_weights = sum(class_counts)/class_counts
		class_weights2 = 1/class_counts

		sample_weights = []
		for idx in range(self.__len__()):
			label = self.labels[idx]
			sample_weights.append(class_weights[label])

		sample_weights = np.array(sample_weights, dtype='float')
		sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
		return sampler

	def plot_balance(self, ax=None, title=None):
		ax = ax or plt.gca()
		unique_labels, counts_labels = np.unique(self.labels, return_counts=True)
		semantic_labels = [self.labels_map[x]+": "+str(y) for x,y in zip(unique_labels, counts_labels)]
		if title: ax.set_title(title)
		ax.pie(counts_labels, labels=semantic_labels, autopct='%1.1f%%');
  
	def plot_images(self, columns=8, rows=4):
		fig = plt.figure(figsize=(16,6));
		for i in range(1, columns*rows+1):
			idx = np.random.randint(self.__len__());
			img, lbl = self.__getitem__(idx)

			fig.add_subplot(rows, columns, i)

			plt.title(self.labels_map[lbl])
			plt.axis('off')
			plt.imshow(img)
		plt.show()

	def get_subsets(self, percentage=0.7):
		length = self.__len__()
		train_length = int(percentage * length)
		valid_length = length - train_length
		train, valid = torch.utils.data.random_split(self, lengths=[train_length, valid_length])

		return train, valid

	def get_images_sizes(self):
		sizes = {}
		for idx in range(self.__len__()):
			img, lbl = self.__getitem__(idx)
			size     = img.size

			if size in sizes: sizes[size] += 1
			else:             sizes[size] = 1

		return sizes

	def get_min_sizes(self):
		sizes = self.get_images_sizes()
		min_w = 10000
		min_h = 10000

		for s in sizes:
			if s[0] < min_w: min_w = s[0]
			if s[1] < min_h: min_h = s[1]

		return {"min_w": min_w, "min_h": min_h}


	def get_mean_and_std(self):
		r_mean, g_mean, b_mean = 0., 0., 0.
		r_std,  g_std,  b_std  = 0., 0., 0.
		length = self.__len__()
		for idx in tqdm(range(length)):
			img, lbl = self.__getitem__(idx)
			img = transforms.ToTensor()(img)
			img = img.view(3, -1)
			r_mean += img[0].mean()
			g_mean += img[1].mean()
			b_mean += img[2].mean()
			r_std  += img[0].std()
			g_std  += img[1].std()
			b_std  += img[2].std()
		r_mean /= length
		g_mean /= length
		b_mean /= length
		r_std  /= length
		g_std  /= length
		b_std  /= length
		return {"mean": [r_mean, g_mean, b_mean],
		        "std":  [r_std,  g_std,  b_std]}






################    __  __           _      _ 
################   |  \/  |         | |    | |
################   | \  / | ___   __| | ___| |
################   | |\/| |/ _ \ / _` |/ _ \ |
################   | |  | | (_) | (_| |  __/ |
################   |_|  |_|\___/ \__,_|\___|_|
################                              
                            


################################    FREEZING

# Finetune the last layer
def freeze(model):
	for param in model.parameters():
		param.requires_grad = False

def partial_freeze(model, n):
	for i, (name, child) in enumerate(model.named_children()):
		if i < n:
			print(i+1,"frozen\t(",name,")")
			for param in child.parameters():
				param.requires_grad = False
		else:
			print(i+1,"unfrozen\t(",name,")")
			for param in child.parameters():
				param.requires_grad = True

# Finetune the whole model
def unfreeze(model):
	for param in model.parameters():
		param.requires_grad = True


def see_params_to_learn(model):
	print("\nParams to learn:")
	for name,param in model.named_parameters():
		if param.requires_grad == True:
			print("\t",name)


################################### SAVE/LOAD


def save_checkpoint(model, is_best, filename='checkpoint.pth.tar'):
	if is_best:
		torch.save(model.state_dict(), filename)  # save checkpoint
	else:
		print ("=> Validation Accuracy did not improve")

def load_checkpoint(model, filename='checkpoint.pth.tar'):
	sd = torch.load(filename, map_location=lambda storage, loc: storage)
	names = set(model.state_dict().keys())
	for n in list(sd.keys()): 
		if n not in names and n+'_raw' in names:
			if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
			del sd[n]
	model.load_state_dict(sd)



################    _                               
################   | |                              
################   | |     __ _ _   _  ___ _ __ ___ 
################   | |    / _` | | | |/ _ \ '__/ __|
################   | |___| (_| | |_| |  __/ |  \__ \
################   |______\__,_|\__, |\___|_|  |___/
################                 __/ |              
################                |___/               

class Flatten(torch.nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

class Concat(torch.nn.Module):
	def forward(self, *xs):
		return torch.cat(xs, 1)

class AdaptiveConcatPool2d(torch.nn.Module):
	def __init__(self, sz=None):
		super().__init__()
		sz = sz or (1,1)
		self.ap = torch.nn.AdaptiveAvgPool2d(sz)
		self.mp = torch.nn.AdaptiveMaxPool2d(sz)
	def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)







################    _      _____    ______ _           _           
################   | |    |  __ \  |  ____(_)         | |          
################   | |    | |__) | | |__   _ _ __   __| | ___ _ __ 
################   | |    |  _  /  |  __| | | '_ \ / _` |/ _ \ '__|
################   | |____| | \ \  | |    | | | | | (_| |  __/ |   
################   |______|_|  \_\ |_|    |_|_| |_|\__,_|\___|_|   
################                                                   
                                                 
"""
The learning rate finder looks for the optimal learning rate to start the training.
The technique is quite simple. For one epoch:

- Start with a very small learning rate (around 1e-8) and increase the learning rate linearly.
- Plot the loss at each step of LR.
- Stop the learning rate finder when loss stops going down and starts increasing.

A graph is created with the x axis having learning rates and the y axis having the losses.

https://medium.com/coinmonks/training-neural-networks-upto-10x-faster-3246d84caacd
https://github.com/nachiket273/One_Cycle_Policy/blob/master/CLR.ipynb
"""

def findLR(model, optimizer, criterion, trainloader, init_value=1e-5, final_value=100):

	save_checkpoint(model, True) # save current model
	model.train()         # setup model for training configuration

	num = len(trainloader) - 1 # total number of batches
	mult = (final_value / init_value) ** (1/num)

	losses = []
	lrs = []
	best_loss = 0.
	avg_loss = 0.
	beta = 0.98 # the value for smooth losses
	lr = init_value

	for batch_num, (inputs, targets) in enumerate(tqdm(trainloader, total=len(trainloader))):

		update_lr(optimizer, lr)

		batch_num += 1 # for non zero value
		inputs, targets = inputs.to(device), targets.to(device) # convert to cuda for GPU usage

		optimizer.zero_grad() # clear gradients
		outputs = model(inputs) # forward pass
		loss = criterion(outputs, targets) # compute loss

		#Compute the smoothed loss to create a clean graph
		avg_loss = beta * avg_loss + (1-beta) *loss.item()
		smoothed_loss = avg_loss / (1 - beta**batch_num)

		#Record the best loss
		if smoothed_loss < best_loss or batch_num==1:
			best_loss = smoothed_loss

		# append loss and learning rates for plotting
		lrs.append(lr) #lrs.append(math.log10(lr)) # Plot modification
		losses.append(smoothed_loss)

		# Stop if the loss is exploding
		if batch_num > 1 and smoothed_loss > 4 * best_loss:
			break

		# backprop for next step
		loss.backward()
		optimizer.step()

		# update learning rate
		lr = mult*lr

	load_checkpoint(model)   # restore original model

	plt.xlabel('Learning Rates')
	plt.ylabel('Losses')
	plt.semilogx(lrs[10:-5], losses[10:-5]) #plt.plot(lrs, losses) # Plot modification
	plt.show()






################    _______        _       
################   |__   __|      (_)      
################      | |_ __ __ _ _ _ __  
################      | | '__/ _` | | '_ \ 
################      | | | | (_| | | | | |
################      |_|_|  \__,_|_|_| |_|
################  

############################################################### TRAIN OLD



def get_optimizer(model, lr=0.0, momentum=0.9, weight_decay=0, nesterov=False):
	params = filter(lambda p: p.requires_grad, model.parameters())
	return torch.optim.SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
	#return torch.optim.Adam(params_to_update, lr=learning_rate);

metrics = { 'train': {'loss' : [], 'acc': [0]},
			'val':   {'loss' : [], 'acc': [0]}}

def train_old(model, optimizer, criterion, dataset, batch_size, num_epochs, num_workers=0, half=True):

	dataloader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
	dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}

	# Decay LR by a factor of 0.1 every 7 epochs
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	since = time.time()
	best_acc = 0.0
	#best_model_wts = copy.deepcopy(model.state_dict())

	mb  = master_bar(range(num_epochs))
	mb.names = ['train', 'val']
	mb.write("Epoch\tTrn_loss\tVal_loss\tTrn_acc\t\tVal_acc")
	# Iterate epochs
	#for epoch in range(num_epochs):
	for epoch in mb:

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step() # Scheduling the learning rate
				model.train()    # Set model to training mode
			else:
				model.eval()     # Set model to evaluate mode

			running_loss     = 0.0
			running_corrects = 0

			# Iterate over data
			#for inputs, labels in dataloader[phase]:
			for inputs, labels in progress_bar(dataloader[phase], parent=mb):
				inputs = inputs.to(device)
				labels = labels.to(device)
				if half: inputs = inputs.half()

				optimizer.zero_grad()			      # zero the parameter gradients
				outputs = model(inputs)                # forward
				preds = torch.argmax(outputs, dim=1)   # prediction
				loss = criterion(outputs, labels)      # loss
				if phase == 'train': loss.backward()   # backward 
				if phase == 'train': optimizer.step()  # optimize

				# statistics
				running_loss     += loss.item() * inputs.size(0)    # multiplicar si nn.CrossEntropyLoss(size_average=True) que es lo default
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc  = running_corrects.double() / dataset_sizes[phase]
			metrics[phase]["loss"].append(epoch_loss)
			metrics[phase]["acc"].append(epoch_acc)
			#draw_plot(metrics)

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				#best_model_wts = copy.deepcopy(model.state_dict())

		x = list(range(len(metrics["train"]["acc"])))
		graphs = [[x,metrics["train"]["acc"]], [x,metrics["val"]["acc"]]]
		mb.update_graph(graphs)
		mb.write("{}/{}\t{:06.6f}\t{:06.6f}\t{:06.6f}\t{:06.6f}".format(epoch+1, num_epochs,
			metrics["train"]["loss"][-1], metrics["val"]["loss"][-1],
			metrics["train"]["acc"][-1],  metrics["val"]["acc"][-1]))

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	#model.load_state_dict(best_model_wts)




############################################################### TRAIN NEW

class LinearInterpolation():
	def __init__(self, xs, ys):
		self.xs, self.ys = xs, ys
	def __call__(self, x):
		return np.interp(x, self.xs, self.ys)

def update_lr(optimizer, lr):
	optimizer.param_groups[0]['lr'] = lr

def update_mom(optimizer, mom):
	optimizer.param_groups[0]['momentum'] = mom

	
def forward(model, batch, criterion, stats):
	inputs   = batch[0].to(device).half()
	labels   = batch[1].to(device).long()
	outputs  = model(inputs)
	preds    = torch.argmax(outputs, dim=1)
	corrects = torch.sum(preds == labels.data) # Metric 1
	loss     = criterion(outputs, labels)      # Metric 2

	stats["loss"].append(loss.item()) # loss.item() * inputs.size(0)
	stats["correct"].append(corrects.item())

	return loss

def backward(model, optimizer, lr, loss):
	assert model.training
	update_lr(optimizer, lr)
	loss.backward()
	optimizer.step()
	model.zero_grad()

def train_epoch(mb, model, batches, optimizer, criterion, lrs, stats):
	model.train(True)
	for lr, batch in zip(lrs, progress_bar(batches, parent=mb)):
		loss = forward(model, batch, criterion, stats)
		backward(model, optimizer, lr, loss)
	return stats

def test_epoch(mb, model, batches, criterion, stats):
	model.train(False)
	for batch in progress_bar(batches, parent=mb):
		loss = forward(model, batch, criterion, stats)
	return stats


metric = {
	"epoch": [],
	"learning rate": [],
	"total time": [],
	"train loss": [],
	"train acc": [],
	"val loss": [],
	"val acc": []
}

def train(model, epochs, learning_rates, optimizer, criterion, dataset, batch_size=512, num_workers=0, drop_last=False, timer=None):
	
	t = timer or Timer()

	train_batches = get_dataloader["train"].
	if balance: train_batches = torch.utils.data.DataLoader(dataset["train"], batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=drop_last, sampler=dataset["train"].get_balanced_sampler())
	else:       train_batches = torch.utils.data.DataLoader(dataset["train"], batch_size, shuffle=True,  pin_memory=True, num_workers=num_workers, drop_last=drop_last)
	test_batches              = torch.utils.data.DataLoader(dataset["val"],   batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
	train_size, val_size = len(dataset["train"]), len(dataset["val"])
	if drop_last: train_size -= (train_size % batch_size)

	num_epochs    = epochs[-1]
	lr_schedule   = LinearInterpolation(epochs, learning_rates)
	#mo_schedule   = LinearInterpolation(epochs, momentum)

	mb = master_bar(range(num_epochs))
	mb.write("Epoch\tTime\tLearRate\tT_loss\tT_accu\t\tV_loss\tV_accu")
	mb.write("-"*70)
	for epoch in mb:

		#train_batches.dataset.set_random_choices() 
		lrs = (lr_schedule(x)/batch_size for x in np.arange(epoch, epoch+1, 1/len(train_batches)))
		train_stats, train_time = train_epoch(mb, model, train_batches, optimizer, criterion, lrs, {'loss': [], 'correct': []}), t()
		test_stats, test_time   = test_epoch(mb, model, test_batches, criterion, {'loss': [], 'correct': []}), t()
		
		metric["epoch"].append(epoch+1)
		metric["learning rate"].append(lr_schedule(epoch+1))
		metric["total time"].append(t.total_time)
		metric["train loss"].append(sum(train_stats['loss'])/train_size)
		metric["train acc"].append(sum(train_stats['correct'])/train_size)
		metric["val loss"].append(sum(test_stats['loss'])/val_size)
		metric["val acc"].append(sum(test_stats['correct'])/val_size)

		mb.write("{}/{}\t{:.0f}:{:.0f}\t{:.4f}\t\t{:.4f}\t{:.4f}\t\t{:.4f}\t{:.4f}".format(
			metric["epoch"][-1],
			num_epochs,
			metric["total time"][-1]//60,
			metric["total time"][-1]%60,
			metric["learning rate"][-1],
			metric["train loss"][-1],
			metric["train acc"][-1],
			metric["val loss"][-1],
			metric["val acc"][-1]
		))
		graphs = [[metric["epoch"], metric["train acc"]],
		          [metric["epoch"], metric["val acc"]]]
		mb.update_graph(graphs)

	return metric












################   __      ___                 _ _          _   _             
################   \ \    / (_)               | (_)        | | (_)            
################    \ \  / / _ ___ _   _  __ _| |_ ______ _| |_ _  ___  _ __  
################     \ \/ / | / __| | | |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
################      \  /  | \__ \ |_| | (_| | | |/ / (_| | |_| | (_) | | | |
################       \/   |_|___/\__,_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
################                                                              
                                                            

# plt.xkcd();  # commic plots plt.rcdefaults() to disable


"""
Plot linear one-cycle lr in the format
epochs         = [0, 15, 30, 35]
learning_rates = [0, 0.1, 0.005, 0]
"""
def plot_lr(epochs, lrs):
    plt.title("Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.plot(epochs, lrs)



def plot_bottleneck(model, criterion, optimizer, dataloader, batch_size=64):
	t = Timer()

	batch = next(iter(dataloader))
	time_data = t()

	inputs   = batch[0].to(device).half()
	labels   = batch[1].to(device).long()
	time_gpu = t()
	outputs, time_forward   = model(inputs), t()
	preds, time_preds       = torch.argmax(outputs, dim=1), t()
	corrects, time_corrects = torch.sum(preds == labels.data), t()
	loss, time_loss         = criterion(outputs, labels) , t()

	update_lr(optimizer, 0.001)
	loss.backward()
	optimizer.step()
	model.zero_grad()
	time_backward = t()


	labels = ['read data', 'to gpu', 'forward', 'preds', 'corrects', 'loss', 'backward']
	times  = [time_data, time_gpu, time_forward, time_preds, time_corrects, time_loss, time_backward]
	y_pos = np.arange(len(times))

	plt.rcParams["figure.figsize"] = (16,4)
	plt.barh(y_pos, times)
	plt.yticks(np.arange(len(labels)), labels)


def plot_dataset_bottleneck(dataset, idx):
	t = Timer()
	img_name, find_img_time = dataset.imgs_dir / (dataset.df.iloc[idx, 0])  , t()
	image   , open_img_time = PIL.Image.open(img_name)                      , t()
	image   , transforms_time = dataset.data_transforms[dataset.subset](image), t()
	label   , find_lbl_time = dataset.df.iloc[idx, 1] - 1                   , t()

	labels = ["find_img_time", "open_img_time", "transforms", "find_lbl_time"]
	times  = [find_img_time,    open_img_time, transforms_time,   find_lbl_time]
	y_pos = np.arange(len(times))
	plt.rcParams["figure.figsize"] = (16,4)
	plt.barh(y_pos, times)
	plt.yticks(np.arange(len(labels)), labels)
	plt.xlim((0,0.1))
    

"""
Live plot inspired from https://github.com/stared/livelossplot
"""
def plot_train(metrics):

	# Plot the metrics
	#clear_output(wait=True)
	#plt.figure(figsize=figsize)

	# Loss
	plt.subplot(1, 2, 1)
	x_range = list(range(1, len(metrics["train"]["loss"]) + 1))
	plt.plot(x_range, metrics["train"]["loss"], label="training")
	#plt.plot(x_range, metrics["val"]["loss"], label="validation")
	plt.title("Loss")
	plt.xlabel('epoch')
	plt.legend(loc='center right')

	# Loss
	plt.subplot(1, 2, 2)
	x_range = range(1, len(metrics["train"]["acc"]) + 1)
	plt.plot(x_range, metrics["train"]["acc"], label="training")
	#plt.plot(x_range, metrics["val"]["acc"], label="validation")
	plt.title("Accuracy")
	plt.xlabel('epoch')
	plt.legend(loc='center right')

	plt.tight_layout()
	plt.show();



def plot_confusion(model, dataset):

	classes = dataset.labels

	batch   = next(iter(dataloader))
	images = batch['image'].to(device)
	labels = batch['label'].to(device)
	outputs = model(images)           # forward

	y_true = labels.cpu().numpy()
	y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
	_, y_pred2 = torch.max(outputs, 1).cpu().numpy()

	cm = confusion_matrix(y_true, y_pred)
	print(y_true)
	print(y_pred)
	print(y_pred2)
	print(cm)

	#plt.matshow(cm)
	plt.imshow(cm, interpolation='nearest')
	plt.title("Confusion matrix")
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes)
	plt.yticks(tick_marks, classes)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	for i in range(len(classes)):
		for j in range(len(classes)):
			plt.text(j, i, cm[i, j], fontsize=18, horizontalalignment="center", color="white")
	plt.tight_layout()