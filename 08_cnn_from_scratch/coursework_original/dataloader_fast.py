# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 08:48:04 2021

@author: aamna
"""


import pandas as pd
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
import os
from glob import glob
import time
from matplotlib import pyplot

data_path = 'Data/images'
############### Data Loading ###############################
def getimages(data_path):      
    classes = os.listdir(data_path)
    img_list, images = [], []
    t = time.time()
    for i in classes:
        img_path = os.listdir(os.path.join(data_path, i))
        img_path = glob(os.path.join(data_path, i, '*.jpg'))
        img_list = img_list + img_path
        data_index = list(range(int(len(img_list))))
    for index in data_index :
        img_path = img_list[index]
        im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        images.append(im)
    print('Dataset loaded...')
    print('\n\nTotal time taken in loading: ', (time.time()-t)//60, '\n\n')
    return np.array(img_list), np.array(images)

img_list, images = getimages("Data\images")

# plot first few images
def plot_images(x):
    for i in range(9):
    	# define subplot
    	pyplot.subplot(330 + 1 + i)
    	# plot raw pixel data
    	pyplot.imshow(x[i], cmap = pyplot.get_cmap('gray'))
    # show the figure
    pyplot.show()
  
plot_images(images)
# DATASET CLASS
class DatasetLoader(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_imgs, list_lb, transform):
        'Initialization'
        self.list_imgs = list_imgs
        self.list_lb = list_lb
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_imgs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img = self.list_imgs[index]
        #applying transform
        img = self.transform(img)
        img = img.type(torch.FloatTensor)
        #get label
        label = self.list_lb[index]

        return img, int(label)
    
def dataloader(data_x, data_y, path, batch_size, shuffle):
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=0, std = np.sqrt(0.5))])
    data = DatasetLoader(data_x, data_y, transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader

train_loader = dataloader(train_x, train_y, 32, True)
test_loader =  dataloader(test_x, test_y, 32, True)