#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 20:55:53 2022

@author: aamna
"""

import os, shutil
import time
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
#from data_split import split_data

#read images into csv and find identities
def read_get_identities(path):
    df = pd.read_csv(path, sep=" ", names=['image_name','Class'])
    labels = df.sort_values('Class')
    actual_names = list(labels.Class.unique())
    #as got 55 identities earlier so 120-54
    subset = df.sample(n = 120, replace=False)
    subset.shape
    sampled_labels = subset.sort_values('Class')
    class_names = list(sampled_labels.Class.unique())

    for c in class_names:
        for i in list(labels[labels['Class']==c]['image_name']):
            #create path to the image 
            get_image = os.path.join('CelebA/Img/img_align_celeba/',str(i))
            dest = 'root/'+str(c)
            print("Copying {0} image to {1} ".format(i, c))
            time.sleep(2)
            if not os.path.exists(dest):
                os.makedirs(dest)
            shutil.copy(get_image,dest)


def make_data_loaders(root, transform, batch_size, train=True):
    #create datasets
    dataset = datasets.ImageFolder(root, transform=transform['train'] if train else 'test')
    #make dataloaders
    
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True, drop_last=True)
    return data_loader

if __name__ == "__main__":
    path = "CelebA/Anno/identity_CelebA.txt"
    img_path = "CelebA/Img/img_align_celeba/"
    batch_size = 1
    
    transform = {
        'train': transforms.Compose([
                transforms.Resize([100,100]), 
                transforms.RandomHorizontalFlip(), 
                transforms.CenterCrop((100, 100)),
                transforms.RandomCrop((80, 80)),
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.0,0.0,0.0), std=(0.1,0.1,0.1))
                
        ]),
        'test': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    }
    
    root = 'data/train'
    train_loader = make_data_loaders(root, transform, batch_size=4, train=True)
    
    #check dataloaders
    for i, data in enumerate(train_loader):
        inputs, lab = data
        print(inputs.shape, len(lab))
        break
