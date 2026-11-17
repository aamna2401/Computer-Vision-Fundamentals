# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:40:34 2021

@author: aamna
"""
import os
import numpy as np
from glob import glob
import time
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def split_data(data_path, root_path_train,root_path_val,root_path_test):
    
    print("traversing path", data_path, "for images")
    
    data_path_list = np.array(glob(data_path))
    print("total ",len(data_path_list), "items traversed")
    
    y = [i for i in range(len(data_path_list))]
    print("Splitting into train and trest sets")
    train, test, _, _ = train_test_split(data_path_list, y, test_size=0.3, random_state=42)
    
    y = [i for i in range(len(train))]
    train, val, _, _ = train_test_split(train, y, test_size=0.2, random_state=42)
    
    print(f"train items traversed {train.shape} \ntest items traversed {test.shape}")
    
    print("processing train data")
    if os.path.exists(root_path_train) and os.path.isdir(root_path_train):
        print("Found existing directory, deleting...")
        shutil.rmtree(root_path_train)
        print("\ndone\n")
    print("Copying train items")
    time.sleep(2)
    for f in tqdm(train):
        folder = f.split("/")[1]
        img = f.split("/")[2]
        destination = root_path_train+"/"+ folder + "/"+ img
        if not os.path.exists(destination):
            os.makedirs(destination)
        shutil.copy(f, destination)
    
    print("processing valid data")
    if os.path.exists(root_path_val) and os.path.isdir(root_path_val):
        print("Found existing directory, deleting...")
        shutil.rmtree(root_path_val)
        print("\ndone\n")
    print("Copying val items")
    time.sleep(2)
    for f in tqdm(val):
        folder = f.split("/")[1]
        img = f.split("/")[2]
        destination = root_path_val+"/"+ folder
        if not os.path.exists(destination):
            os.makedirs(destination)
        shutil.copy(f, destination)
    
    print("processing test data")
    if os.path.exists(root_path_test) and os.path.isdir(root_path_test):
        print("Found existing directory, deleting...")
        shutil.rmtree(root_path_test)
        print("\ndone\n")
    print("Copying test items")
    time.sleep(2)
    for f in tqdm(test):
        folder = f.split("/")[1]
        img = f.split("/")[2]
        destination = root_path_test+"/"+folder
        if not os.path.exists(destination):
            os.makedirs(destination)
        shutil.copy(f, destination)
    time.sleep(1)
    print("\n======== Done ================")

if __name__ == "__main__":
    data_path = 'root/**/*.jpg'
    root_path_train = "data/train"
    root_path_val = "data/val"
    root_path_test = "data/test"
    
    split_data(data_path, root_path_train, root_path_val, root_path_test)
