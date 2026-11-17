#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 21:17:18 2022

@author: aamna
"""
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
from PIL import Image
import random
from matplotlib import image as img
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def load_data(path):
    classes = os.listdir(path)
    data_x, data_y = [], []
    for i in classes:
        for filename in glob.glob(path + '/' + i +'/*.jpg'):
            print(filename)
            im=img.imread(filename)
            data_x.append(im)
            data_y.append(int(i))
    print('Dataset loaded...')
    return np.array(data_x), np.array(data_y)

def image_triplets(data_x, data_y):
    image_pairs = []
    labels_pairs = []
    #get total number of identities/classes in dataset
    classes = np.unique(data_y)
    #get all indexes where classes and label are same
    #idx = [(np.where(data_y == c)[0]) for c in classes]
    for id_ in range(0, len(classes)):
        #print(id_)
        #target image and its label
        random_idx = random.randint(0, data_x.shape[0] - 1)
        target_img = data_x[id_]
        label = data_y[id_]
        #get positive image pair to create label 1
        idx_p = np.where(label == data_y)[0]
        idxp = np.random.choice(idx_p)
        pos_image = data_x[idxp]
        #update image pairs and label 1
        image_pairs.append([target_img, pos_image])
        labels_pairs.append([1])
        #get negative image for label 0
        idx_n = np.where(label != data_y)[0]
        idx_n = np.random.choice(idx_n)
        neg_image = data_x[idx_n]
        #update image pairs and label 0
        image_pairs.append([target_img, neg_image])
        labels_pairs.append([0])
    return np.array(image_pairs), np.array(labels_pairs)

def imshow(image, text):
    np_img = image.numpy()
    plt.axis("off")
    if text:
        plt.text(65, 5, text, style='italic',fontweight='bold',
            bbox={'facecolor':'blue', 'alpha':0.5, 'pad':15})
        
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()    

class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self,datasetfolder,transform=None):
        self.datasetfolder = datasetfolder   
        self.transform = transform
        
    def __len__(self):
         return len(self.datasetfolder.imgs)
        
    def __getitem__(self,index):
        img_0_tuple = random.choice(self.datasetfolder.imgs)

        #We need to approximately 50% of images to be in the same class
        same_class_idx = random.randint(0,1) 
        if same_class_idx:
            while True:
                #Look untill the same class image is found
                img_1_tuple = random.choice(self.datasetfolder.imgs) 
                if img_0_tuple[1] == img_1_tuple[1]:
                    break
        else:

            while True:
                #Look untill a different class image is found
                img_1_tuple = random.choice(self.datasetfolder.imgs) 
                if img_0_tuple[1] != img_1_tuple[1]:
                    break
        #read images
        img_0 = Image.open(img_0_tuple[0])
        img_1 = Image.open(img_1_tuple[0])
        #convert to grayscale
        img_0 = img_0.convert("L")
        img_1 = img_1.convert("L")

        if self.transform is not None:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)
        
        return img_0, img_1, torch.from_numpy(np.array([int(img_1_tuple[1] == img_0_tuple[1])],dtype=np.float32))

def create_data_loaders(root):
    # Load the training dataset
    dataset_folder = datasets.ImageFolder(root="data/train")
    transform = transforms.Compose([transforms.Resize((100,100)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(0,0.5)
                                        ])
    dataset = DatasetLoader(dataset_folder, transform)
    data_loader = DataLoader(dataset,
                            shuffle=True,
                            num_workers=2,
                            batch_size=8)
    return data_loader

if __name__ == "__main__":
    path = 'root'
    data_x, data_y = load_data(path)
    image_pairs, labels_pairs = image_triplets(data_x, data_y)
    train_loader = create_data_loaders('data/train')
    # Extract one batch
    data_batch = next(iter(train_loader))
    concat_ = torch.cat((data_batch[0], data_batch[1]),0)
    imshow(torchvision.utils.make_grid(concat_), text = 'data_batch')
    print(data_batch[2].numpy().reshape(-1))
