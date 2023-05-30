#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 05:33:15 2022

@author: aamna
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from data_split import df_split
import numpy as np


class CustomDatasetLoader(Dataset):
    def __init__(self, df, transform, root):
        #transformations
        self.transforms = transform
        #get images array
        self.images = np.asarray(df.iloc[:, 0])
        #get labels array 
        self.labels = np.asarray(df.iloc[:, 1:])
        # # Calculate len
        # self.data_len = len(self.label_arr)
        #images root folder
        self.root = root
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # get single image name
        image_name = self.images[index]
        # open image
        img = Image.open(os.path.join(self.root, image_name))
        #Use transforms
        if self.transforms:
            tensor_img = self.transforms(img)
        #get image labels from df
        label = self.labels[index]
        return (tensor_img, label)


def make_dataloaders(df,root,transform,shuffle,batch_size):
    data = CustomDatasetLoader(df, transform, root)
    dataloader = torch.utils.data.DataLoader(data,shuffle=shuffle,
                                             batch_size=batch_size)
    return dataloader


    
if "__name__" == "__main__":
    csv_file = 'task3_train.csv'
    train_df, val_df, test_df = df_split(csv_file)
    tfs = transforms.Compose([transforms.Resize((256, 256)),
                               transforms.ToTensor()])   
    train_loader=make_dataloaders(train_df, 'root/train', tfs, 
                                  shuffle=True, batch_size=12)
    val_loader=make_dataloaders(val_df, 'root/val', tfs, 
                                  shuffle=True, batch_size=12)
    test_loader=make_dataloaders(test_df, 'root/test', tfs, 
                                  shuffle=True, batch_size=12)
    
    for x in next(iter(val_loader)):
        print(x)