import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
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

# plt.xkcd();  # commic plots plt.rcdefaults() to disable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



############################################################### VISUALIZATION

######################## SHOW BALANCED PIE

def plot_balance(dataset):
	balance = dataset.df.groupby(['Label']).count()

	count = balance["Image"].values
	#label_indexes = balance.index.values
	labels = [a+": "+str(b) for a, b in zip(dataset.labels, count)]

	plt.pie(count, labels=labels, autopct='%1.1f%%');


######################## SHOW IMAGES

def plot_images(dataset, columns=8, rows=4):

	fig = plt.figure(figsize=(16,6));
	for i in range(1, columns*rows+1):
		idx = np.random.randint(len(dataset));
		img, lbl = dataset[idx]

		fig.add_subplot(rows, columns, i)

		plt.title(dataset.labels_map[lbl])
		plt.axis('off')
		plt.imshow(img)
	plt.show()






################################################################### MODEL


################################### FREEZING

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



################################### LAYERS

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

class AdaptiveConcatPool2d(nn.Module):
	def __init__(self, sz=None):
		super().__init__()
		sz = sz or (1,1)
		self.ap = nn.AdaptiveAvgPool2d(sz)
		self.mp = nn.AdaptiveMaxPool2d(sz)
	def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)




################################################################### LR FINDER
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

def update_lr(optimizer, lr):
	for g in optimizer.param_groups:
		g['lr'] = lr

def update_mom(optimizer, mom):
	for g in optimizer.param_groups:
		g['momentum'] = mom

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



def plot_confusion():

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