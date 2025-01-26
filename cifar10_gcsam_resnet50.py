from pathlib import Path
from pyhessian import hessian
import torchvision.models as models
import numpy as np
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import datetime
now = datetime.datetime.now
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
import torchvision
from torchvision import transforms
import copy
import os
from functools import partial
from itertools import product
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from torchvision import datasets, transforms as T
from sklearn.model_selection import train_test_split, KFold
import natsort
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import image
from os import listdir
import pandas
from pandas import DataFrame
import tensorflow as tf
import cv2
import glob
from itertools import chain
import os
import random
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from vit_pytorch import ViT
#from mlp_mixer_pytorch import MLPMixer
#from vit_pytorch.efficient import ViT


"""**Dataset Loading**"""
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
    


import torch
#from torch.optim.optimizer import Optimizer, required

def centralized_gradient(x,use_gc=True,gc_conv_only=False):
    if use_gc:
      if gc_conv_only:
        if len(list(x.size()))>3:
            x.add_(-x.mean(dim = tuple(range(1,len(list(x.size())))), keepdim = True))
      else:
        if len(list(x.size()))>1:
            x.add_(-x.mean(dim = tuple(range(1,len(list(x.size())))), keepdim = True))
    return x


"""**SAM Optimizer**"""

import torch
#from torch.optim.optimizer import Optimizer, required

class GCSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(GCSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                #GC operation
                p.grad =centralized_gradient(p.grad ,use_gc=True,gc_conv_only=False)
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                #p.grad =centralized_gradient(p.grad ,use_gc=True,gc_conv_only=False)
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

"""**Model Loading**"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

import torchvision.models as models
model = models.resnet.resnet50(num_classes=10)
print(model)

criterion = nn.CrossEntropyLoss()


model = model.to(device,dtype=torch.float)

base_optimizer = torch.optim.Adam

optimizer = GCSAM(model.parameters(), base_optimizer, rho=0.05, lr=1e-4)

"""**Model Training**"""
num_epochs = 120
# keeping-track-of-losses
train_losses = []
test_losses = []

# Patience for early stopping (based on non-increasing classification test accuracy)
n_epochs_stop = 10

# Initialize the minimum test loss
min_test_loss = np.inf
epochs_no_improve = 0

for epoch in range(1, num_epochs + 1):
    # keep-track-of-training-loss
    train_loss = 0.0

    # training-the-model
    model.train()
    for data, target in train_loader:
        # move-tensors-to-GPU
        data = data.to(device, dtype=torch.float)
        target = target.to(device)

        # first forward-backward pass
        predictions = model(data)
        loss = criterion(predictions, target)  # use this loss for any training statistics
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward pass
        criterion(model(data), target).backward()  # full forward pass
        optimizer.second_step(zero_grad=True)

        # update the training loss
        train_loss += loss.item() * data.size(0)

    # test-the-model
    model.eval()  # disables dropout
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average losses
    train_loss = train_loss / len(train_loader.sampler)
    test_loss = test_loss / len(test_loader.sampler)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f} \tTest Accuracy: {} %'.format(
        epoch, train_loss, test_loss, 100 * correct / total))

    # Check early stopping condition
    if test_loss < min_test_loss:
        epochs_no_improve = 0
        min_test_loss = test_loss
    else:
        epochs_no_improve += 1

    if epochs_no_improve == n_epochs_stop:
        print('Early stopping!')
        early_stop = True
        break


# test-the-model
model.eval()  # it-disables-dropout
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))


# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'

path = "/lfs/hass7410.ui/cifar10/"
os.chdir(path)
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Test loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.savefig('results of GCSAM CIFAR10 Resnet50.png')
