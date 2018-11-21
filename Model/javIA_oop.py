import torch
import torchvision
from torchvision import transforms

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














class DeepLearner():

	def __init__(self, dataset, model_name, pretrained=True, half_prec=True):

		self.device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.dataset    = dataset
		self.pretrained = pretrained
		self.half_prec  = half_prec
		self.get_model(model_name)
		self.criterion  = torch.nn.CrossEntropyLoss()
		self.lr         = 0.01
		self.mom        = 0.9
		self.wd         = 1e-4
		self.nesterov   = False
		self.optimizer  = self.get_optimizer()
		self.metrics    = {
			"epoch": [],
			"learning rate": [],
			"total time": [],
			"train loss": [],
			"train acc": [],
			"val loss": [],
			"val acc": []
		}		
		
		# TODO
		self.dataset["train"].transforms = transforms.Compose([transforms.CenterCrop(128), transforms.ToTensor()])
		self.dataset["valid"].transforms = transforms.Compose([transforms.CenterCrop(128), transforms.ToTensor()])
		self.batch_size = 8
		balanced_sampler = dataset["train"].get_balanced_sampler()
		self.train_batches = torch.utils.data.DataLoader(dataset["train"], batch_size=self.batch_size, sampler=balanced_sampler)
		self.valid_batches = torch.utils.data.DataLoader(dataset["train"], batch_size=self.batch_size, sampler=balanced_sampler)
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

		if self.pretrained: self.freeze()

		# Edit last 2 layers
		num_classes = len(np.bincount(self.dataset["train"].labels))
		self.model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))                      # for variable image sizes
		self.model.fc      = torch.nn.Linear(self.model.fc.in_features, num_classes) # new layer unfreezed by default

		# Send the model to GPU
		self.model = self.model.to(self.device)
		if self.half_prec: self.model = self.model.half()


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

		
	def forward(self, batch, stats):
		inputs   = batch[0].to(self.device).half()
		labels   = batch[1].to(self.device).long()
		outputs  = self.model(inputs)
		preds    = torch.argmax(outputs, dim=1)
		corrects = torch.sum(preds == labels.data) # Metric 1
		loss     = self.criterion(outputs, labels) # Metric 2

		stats["loss"].append(loss.item()) # loss.item() * inputs.size(0)
		stats["correct"].append(corrects.item())

		return loss

	def backward(self, lr, loss):
		assert self.model.training
		self.update_lr(lr)
		#self.update_mom(mom)
		loss.backward()
		self.optimizer.step()
		self.model.zero_grad()

	def train_epoch(self, mb, lrs, stats):
		self.model.train(True)
		for lr, batch in zip(lrs, progress_bar(self.train_batches, parent=mb)):
			loss = self.forward(batch, stats)
			self.backward(lr, loss)
		return stats

	def valid_epoch(self, mb, stats):
		self.model.train(False)
		for batch in progress_bar(self.valid_batches, parent=mb):
			loss = self.forward(batch, stats)
		return stats

	def train(self, epochs, learning_rates):
	
		t = Timer()

		train_size, val_size = len(self.dataset["train"]), len(self.dataset["valid"])
		#if drop_last: train_size -= (train_size % self.batch_size)

		num_epochs    = epochs[-1]
		lr_schedule   = LinearInterpolation(epochs, learning_rates)
		#mo_schedule   = LinearInterpolation(epochs, momentum)

		mb = master_bar(range(num_epochs))
		mb.write("Epoch\tTime\tLearRate\tT_loss\tT_accu\t\tV_loss\tV_accu")
		mb.write("-"*70)
		for epoch in mb:

			mb.write("epoch")

			#self.train_batches.dataset.set_random_choices() 
			lrs = (lr_schedule(x)/self.batch_size for x in np.arange(epoch, epoch+1, 1/len(self.train_batches)))
			train_stats, train_time = self.train_epoch(mb, lrs, {'loss': [], 'correct': []}), t()
			valid_stats, valid_time = self.valid_epoch(mb,      {'loss': [], 'correct': []}), t()
			
			metric["epoch"].append(epoch+1)
			metric["learning rate"].append(lr_schedule(epoch+1))
			metric["total time"].append(t.total_time)
			metric["train loss"].append(sum(train_stats['loss'])/train_size)
			metric["train acc"].append(sum(train_stats['correct'])/train_size)
			metric["val loss"].append(sum(valid_stats['loss'])/val_size)
			metric["val acc"].append(sum(valid_stats['correct'])/val_size)

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