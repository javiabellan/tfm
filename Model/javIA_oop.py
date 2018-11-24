import torch
import torchvision
from torchvision import transforms

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from fastprogress import master_bar, progress_bar
import pathlib
import PIL
from tqdm import tqdm
import time
import os
import copy
print("AI framework by Javi based in PyTorch:",torch.__version__)



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
		if type(idx)==torch.Tensor: idx = idx.item() # TODO: bug of WeightedRandomSampler?
		img_name = self.image_dir / self.images[idx]
		image = PIL.Image.open(img_name)
		if self.transforms: image = self.transforms(image)
		label = self.labels[idx]
		return image, label

	def get_balanced_sampler(self):
		class_counts  = np.bincount(self.labels)
		class_weights = sum(class_counts)/class_counts
		class_weights2 = 1/class_counts

		sample_weights = []
		for idx in range(self.__len__()):
			label = self.labels[idx]
			sample_weights.append(class_weights[label])

		sample_weights = np.array(sample_weights, dtype='float')
		# WeightedRandomSampler: The size of the epoch is the same, but now are balanced (undersampling and oversampling)
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

	def split(self, val_size=0.3):
		x_train, x_valid, y_train, y_valid = train_test_split(self.images, self.labels, test_size=val_size, stratify=self.labels)

		train_ds = ImageDataset(image_dir=self.image_dir, images=x_train, labels=y_train, labels_map=self.labels_map, transforms=self.transforms, limit=self.limit)
		valid_ds = ImageDataset(image_dir=self.image_dir, images=x_valid, labels=y_valid, labels_map=self.labels_map, transforms=self.transforms, limit=self.limit)

		return train_ds, valid_ds

	def split2(self, percentage=0.7):
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











################    _                               
################   | |                              
################   | |     __ _ _   _  ___ _ __ ___ 
################   | |    / _` | | | |/ _ \ '__/ __|
################   | |___| (_| | |_| |  __/ |  \__ \
################   |______\__,_|\__, |\___|_|  |___/
################                 __/ |              
################                |___/               

class Flat1D(torch.nn.Module):
	def forward(self, x):
		return x.view(-1)

class Flat2D(torch.nn.Module):
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











class DeepLearner():

	def __init__(self, model_name, train_ds, valid_ds, test_ds=None, batch_size=64, pretrained=True, half_prec=True, balance=False):

		self.device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.train_ds   = train_ds
		self.valid_ds   = valid_ds
		self.test_ds    = test_ds
		self.batch_size = batch_size
		self.pretrained = pretrained
		self.half_prec  = half_prec
		self.get_model(model_name)
		self.lr         = 0.01
		self.mom        = 0.9
		self.wd         = 5e-4 #1e-4
		self.nesterov   = False
		self.optimizer  = self.get_optimizer()
		self.log        = {
			"epoch": [],
			"learning rate": [],
			"total time": [],
			"train loss": [],
			"train acc": [],
			"val loss": [],
			"val acc": []
		}		
		
		# TODO
		num_workers= 0
		pin_memory = True
		drop_last  = False
		if balance:
			shuffle = False
			sampler = dataset["train"].get_balanced_sampler()
		else:
			shuffle = True
			sampler = None
		self.train_batches = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler)
		self.valid_batches = torch.utils.data.DataLoader(self.valid_ds, batch_size=self.batch_size)
		if test_ds:
			self.test_batches = torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size)
		"""
		scale       = 360
		input_size  = 540 #224
		mean        = [0.485, 0.456, 0.406] #[0.5, 0.5, 0.5]
		std         = [0.229, 0.224, 0.225] #[0.5, 0.5, 0.5]
		self.balance    = False
		self.shuffle    = False
		self.num_workers= 0
		self.pin_memory = True
		self.drop_last  = False
		self.train_loader = self.get_dataloader()
		self.valid_loader = self.get_dataloader()
		transforms = {
			"train": transforms.Compose([
				#transforms.Resize(scale),
				transforms.RandomResizedCrop(input_shape),
				transforms.RandomHorizontalFlip(),
				transforms.RandomVerticalFlip(),
				transforms.RandomRotation(degrees=90),
				transforms.ToTensor(),
				transforms.Normalize(mean, std)]),
			"valid": transforms.Compose([
				#transforms.Resize(scale),
				transforms.CenterCrop(input_shape),
				transforms.ToTensor(),
				transforms.Normalize(mean, std)]),
			"test": transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean, std)])}
		"""


	def get_optimizer(self):
		params = filter(lambda p: p.requires_grad, self.model.parameters())
		return  torch.optim.SGD(params=params, lr=self.lr, momentum=self.mom, weight_decay=self.wd, nesterov=self.nesterov)
		#return torch.optim.Adam(params_to_update, lr=learning_rate);

	def get_model(self, model_name="resnet18"):

		# Get model from torchvision
		try:
			model_fn   = eval("torchvision.models."+model_name) # Create model fn
			self.model = model_fn(pretrained=self.pretrained)   # Create model
		except AttributeError:
			print(model_name+" model don't exists in torchvision.")

		# if self.pretrained: self.freeze()

		# Edit last 2 layers
		num_classes = len(np.bincount(self.train_ds.labels))
		assert num_classes >= 2
		self.model.avgpool = AdaptiveConcatPool2d()  # Adaptative->variable image sizes.  Concat-> Avg + Max
		self.model.fc      = torch.nn.Linear(self.model.fc.in_features*2, num_classes) # new layer unfreezed by default

		# Define the loss function
		if num_classes > 2:
			self.criterion = torch.nn.CrossEntropyLoss()   # Softmax + Cross entropy
		else: # Binary classification
			self.criterion = torch.nn.BCEWithLogitsLoss()  # Sigmoid + Binary cross entropy
			self.model.fc  = torch.nn.Sequential(torch.nn.Linear(self.model.fc.in_features, 1), Flat1D())

		# Define metric
		if num_classes > 2:
			self.metric = self.accuracy
		else: # Binary classification
			self.metric = self.binary_metric

		# Send the model to GPU
		self.model = self.model.to(self.device)
		if self.half_prec: self.model = self.model.half()




	def accuracy(self, output, target):
		preds    = torch.argmax(output, dim=1)
		return torch.sum(preds == target.data).item()

	def binary_metric(self, output, target):
		a = target.data.cpu().numpy()
		b = output.detach().cpu().numpy()
		return roc_auc_score(a, b)



	# Finetune the last layer
	def freeze(self):
		for param in self.model.parameters():
			param.requires_grad = False


	def print_model(self):
		print("Model:")
		for i, (name, child) in enumerate(self.model.named_children()):
			print("\t("+str(i+1)+") "+name)


	def get_dataloader(self, dataset, batch_size, balanced=False, shuffle=False, num_workers=0, pin_memory=True, drop_last=False):
		
		sampler = None
		if balanced:
			sampler = dataset.get_balanced_sampler()
			shuffle = False			

		return torch.utils.data.DataLoader(dataset     = self,
		                                   batch_size  = batch_size,
		                                   shuffle     = shuffle,
		                                   sampler     = sampler,
		                                   num_workers = num_workers,
		                                   pin_memory  = pin_memory,
		                                   drop_last   = drop_last)


	################################### SAVE/LOAD

	def save_model(self, filename='model.pt'):
		torch.save(self.model.state_dict(), filename)

	def load_model(self, filename='model.pt'):
		self.model.load_state_dict(torch.load(filename))

	def load_model_2(self, filename='model.pt'):
		sd = torch.load(filename, map_location=lambda storage, loc: storage)
		names = set(self.model.state_dict().keys())
		for n in list(sd.keys()): 
			if n not in names and n+'_raw' in names:
				if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
				del sd[n]
		self.model.load_state_dict(sd)




	################    _______        _       
	################   |__   __|      (_)      
	################      | |_ __ __ _ _ _ __  
	################      | | '__/ _` | | '_ \ 
	################      | | | | (_| | | | | |
	################      |_|_|  \__,_|_|_| |_|
	################  

	def update_lr(self, lr):
		self.optimizer.param_groups[0]['lr'] = lr

	def update_mom(self, mom):
		self.optimizer.param_groups[0]['momentum'] = mom
	

	def get_batch(self, batch):
		inputs  = batch[0].to(self.device)
		labels  = batch[1].to(self.device)
		if self.half_prec: inputs, labels = inputs.half(), labels.half()
		return inputs, labels

	def backward(self, lr, loss):
		assert self.model.training
		self.update_lr(lr)
		#self.update_mom(mom)
		loss.backward()
		self.optimizer.step()
		self.model.zero_grad()

	def train_epoch(self, stats, epoch, mb, lrs):
		it = epoch*len(self.train_batches)
		self.model.train(True)
		for lr, batch in zip(lrs, progress_bar(self.train_batches, parent=mb)):
			input, target = self.get_batch(batch)
			output        = self.model(input)
			metric        = self.metric(output, target)
			loss          = self.criterion(output, target)

			it += 1
			stats["train_it"].append(it)
			stats["train_loss"].append(loss.item()) # loss.item() * inputs.size(0)
			stats["train_metric"].append(metric)

			graphs = [	[stats["train_it"], stats["train_loss"]],
						[stats["train_it"], stats["train_metric"]],
						[stats["valid_it"], stats["valid_loss"]],
						[stats["valid_it"], stats["valid_metric"]]]
			mb.update_graph(graphs)

			self.backward(lr, loss)
		return stats

	def valid_epoch(self, stats, epoch, mb):
		it = epoch*len(self.valid_batches)
		self.model.train(False)
		with torch.no_grad():
			for batch in progress_bar(self.valid_batches, parent=mb):
				#loss = self.forward(batch, stats)
				input, target = self.get_batch(batch)
				output        = self.model(input)
				metric        = self.metric(output, target)
				loss          = self.criterion(output, target)

				it += 1
				stats["valid_it"].append(it)
				stats["valid_loss"].append(loss.item()) # loss.item() * inputs.size(0)
				stats["valid_metric"].append(metric)

				graphs = [	[stats["train_it"], stats["train_loss"]],
							[stats["train_it"], stats["train_metric"]],
							[stats["valid_it"], stats["valid_loss"]],
							[stats["valid_it"], stats["valid_metric"]]]
				mb.update_graph(graphs)

		return stats

	def test_epoch(self):
		preds = []
		self.model.train(False)
		with torch.no_grad():
			for batch in tqdm(self.test_batches):
				input, target = self.get_batch(batch)
				output        = self.model(input).cpu().numpy()
				preds         = np.concatenate((preds, output))

		torch.cuda.empty_cache() # free cache mem after test
		return preds


	def train(self, num_epochs, max_lr=0.1):
	
		t = Timer()

		valid_loss_min = np.Inf
		patience       = 10
		p              = 0 # current number of epochs, where validation loss didn't increase
		#train_size, val_size = len(self.train_ds), len(self.valid_ds)
		#if drop_last: train_size -= (train_size % self.batch_size)

		self.epochs         = [0, num_epochs/4, num_epochs] #[0, 15, 30, 35]
		self.learning_rates = [0, max_lr, 0]                #[0, 0.1, 0.005, 0]
		lr_schedule    = LinearInterpolation(self.epochs, self.learning_rates)
		#mo_schedule   = LinearInterpolation(epochs, momentum)

		stats = {"train_it": [], 'train_loss': [], 'train_metric': [],
				"valid_it": [], 'valid_loss': [], 'valid_metric': []}

		mb = master_bar(range(num_epochs))
		mb.names = ["train loss", "train acc", "val loss", "val acc"]
		mb.write("Epoch\tTime\tLearRate\tT_loss\tT_accu\t\tV_loss\tV_accu")
		mb.write("-"*70)
		for epoch in mb:

			mb.write("epoch")

			#self.train_batches.dataset.set_random_choices() 
			lrs = (lr_schedule(x)/self.batch_size for x in np.arange(epoch, epoch+1, 1/len(self.train_batches)))
			stats, train_time = self.train_epoch(stats, epoch, mb, lrs), t()
			stats, valid_time = self.valid_epoch(stats, epoch, mb,    ), t()
			
			self.log["epoch"].append(epoch+1)
			self.log["learning rate"].append(lr_schedule(epoch+1))
			self.log["total time"].append(t.total_time)
			self.log["train loss"].append(np.mean(stats['train_loss'])) # or np.mean
			self.log["train acc"].append(np.mean(stats['train_metric']))
			self.log["val loss"].append(np.mean(stats['valid_loss']))
			self.log["val acc"].append(np.mean(stats['valid_metric']))


			if self.log["val loss"][-1] <= valid_loss_min:   # Val loss improve
				mb.write('Saving model!')
				self.save_model()
				valid_loss_min = self.log["val loss"][-1]
				p = 0
			else:                                            # Val loss didn't improve
				p += 1
				if p > patience:
					mb.write('Stopping training')
					break

			mb.write("{}/{}\t{:.0f}:{:.0f}\t{:.4f}\t\t{:.4f}\t{:.4f}\t\t{:.4f}\t{:.4f}".format(
				self.log["epoch"][-1],
				num_epochs,
				self.log["total time"][-1]//60,
				self.log["total time"][-1]%60,
				self.log["learning rate"][-1],
				self.log["train loss"][-1],
				self.log["train acc"][-1],
				self.log["val loss"][-1],
				self.log["val acc"][-1]
			))

			#graphs = [[self.log["epoch"], self.log["train acc"]],
			#          [self.log["epoch"], self.log["val acc"]]]
			#mb.update_graph(graphs)

			torch.cuda.empty_cache() # free cache mem after train 
			

	def test(self, filename):
		preds = self.valid_epoch()


	def plot_lr(self):
		plt.xlabel("Epochs")
		plt.ylabel("Learning Rate")
		plt.plot(self.epochs, self.learning_rates)


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

	def findLR(self, init_value=1e-5, final_value=100):

		model_weights = self.model.state_dict()  # save current model
		self.model.train()                       # setup model for training configuration

		num = len(self.train_batches) - 1 # total number of batches
		mult = (final_value / init_value) ** (1/num)

		losses = []
		lrs = []
		best_loss = 0.
		avg_loss = 0.
		beta = 0.98 # the value for smooth losses
		lr = init_value

		for batch_num, (inputs, targets) in enumerate(tqdm(self.train_batches, total=len(self.train_batches))):

			self.update_lr(lr)

			batch_num += 1 # for non zero value
			inputs, targets = inputs.to(self.device), targets.to(self.device) # convert to cuda for GPU usage

			self.optimizer.zero_grad() # clear gradients
			outputs = self.model(inputs) # forward pass
			loss = self.criterion(outputs, targets) # compute loss

			# Compute the smoothed loss to create a clean graph
			avg_loss = beta * avg_loss + (1-beta) *loss.item()
			smoothed_loss = avg_loss / (1 - beta**batch_num)

			# Record the best loss
			if smoothed_loss < best_loss or batch_num==1:
				best_loss = smoothed_loss

			# Append loss and learning rates for plotting
			lrs.append(lr) #lrs.append(math.log10(lr)) # Plot modification
			losses.append(smoothed_loss)

			# Stop if the loss is exploding
			if batch_num > 1 and smoothed_loss > 4 * best_loss:
				break

			# Backprop for next step
			loss.backward()
			self.optimizer.step()

			# update learning rate
			lr = mult*lr

		self.model.load_state_dict(model_weights) # restore original model

		plt.xlabel('Learning Rates')
		plt.ylabel('Losses')
		plt.semilogx(lrs[10:-5], losses[10:-5]) #plt.plot(lrs, losses) # Plot modification
		plt.show()












class LinearInterpolation():
	def __init__(self, xs, ys):
		self.xs, self.ys = xs, ys
	def __call__(self, x):
		return np.interp(x, self.xs, self.ys)


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





def gpu_info():
	print("cuda available:  ", torch.cuda.is_available())
	print("Id of GPU:       ", torch.cuda.current_device())
	print("Name of GPU:     ", torch.cuda.get_device_name(0))
	print("Total mem of GPU:", torch.cuda.get_device_properties(0).total_memory/1e9, "GB")
	print("Tensor used mem: ", torch.cuda.memory_allocated())
	print("Tensor max mem:  ", torch.cuda.max_memory_allocated())
	print("Cache used mem:  ", torch.cuda.memory_cached())
	print("Cache max mem:   ", torch.cuda.max_memory_cached())

def mem():
	tensor_mem = torch.cuda.memory_allocated()
	cache_mem  = torch.cuda.memory_cached()
	free_mem   = torch.cuda.get_device_properties(0).total_memory - tensor_mem - cache_mem
	mem_values = [tensor_mem, cache_mem, free_mem]
	mem_lables = ["tensor", "cache", "free"]
	plt.title("GPU mem")
	plt.pie(mem_values, labels=mem_lables, autopct='%1.1f%%');

def free():
    torch.cuda.empty_cache()
    mem()
































################################################################## DEPRECATED



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

"""