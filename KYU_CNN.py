#-*- coding: utf-8 -*-

import os
import sys
import time
import math
import argparse
import queue
import shutil
import random
import math
import time
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein as Lev 
import cv2
import glob
import shutil
from joblib import Parallel, delayed
import nsml
from nsml import GPU_NUM, DATASET_PATH, DATASET_NAME, HAS_DATASET
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.init as init

if HAS_DATASET == False:
    DATASET_PATH = 'fundus3'

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.init as init

train_transform = transforms.Compose([
    transforms.Resize(114),
    transforms.CenterCrop(100), 
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])])
   
test_transform = transforms.Compose([
    transforms.Resize(114),
    transforms.CenterCrop(100), 
    transforms.ToTensor()])

batch_size = 16

train_data = dsets.ImageFolder(os.path.join(DATASET_PATH,'train','train'), train_transform)
test_data = dsets.ImageFolder(os.path.join(DATASET_PATH,'train','test'), test_transform)

train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = Data.DataLoader(test_data, batch_size=5, shuffle=True, num_workers=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,16,5),
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            nn.Conv2d(16,32,5), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2,2), 
            nn.Conv2d(32,64,5), 
            nn.ReLU(),
            nn.MaxPool2d(2,2), 
            nn.Conv2d(64,128,6),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2), 
            nn.Conv2d(128,256,5),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(256*4*4,100),
            nn.ReLU(),
            nn.Linear(100,4)
        )
       
    def forward(self,x):
        out = self.layer(x)
        out = out.view(-1, 256*4*4)
        out = self.fc_layer(out)

        return out

model = CNN().cuda()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001) 
num_epochs = 150

for epoch in range(num_epochs):
   
    for i, (batch_images, batch_labels) in enumerate(train_loader):
        
        X = Variable(batch_images).cuda()
        Y = Variable(batch_labels).cuda()

        pre = model(X)
        cost = loss(pre, Y)
       
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
       

        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], lter [%d/%d] Loss: %.4f'
                 %(epoch+1, num_epochs, i+1, len(train_data)//batch_size, cost.data[0]))
    
model.eval()

correct = 0
total = 0

for images, labels in test_loader:
    
    images = Variable(images).cuda()
    outputs = model(images)
    
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
print('acc: %f %%' % (100 * correct / total))
