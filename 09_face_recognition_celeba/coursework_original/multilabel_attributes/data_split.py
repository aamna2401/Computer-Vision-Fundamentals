#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 01:58:12 2022

@author: aamna
"""

import pandas as pd
from glob import glob
import os
import time
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def df_split(csv_file):
    label_df = pd.read_csv(csv_file, nrows=500)
    #shuffle = np.random.permutation(len(files))
    #split data into train valid
    y = [i for i in range(len(label_df))]
    print("Splitting into train and trest sets")
    train, test, _, _ = train_test_split(label_df, y, test_size=0.2, random_state=42)
    y = [i for i in range(len(train))]
    train, val, _, _ = train_test_split(train, y, test_size=0.3, random_state=42)
    return train, val, test

def data_split(csv_file,root,root_train,root_val,root_test):
    train, val, test = df_split(csv_file)
    files = glob(root)
    ###split into train val test Folders #############
    print("processing train data")
    if os.path.exists(root_train) and os.path.isdir(root_train):
        print("Found existing directory, deleting...")
        shutil.rmtree(root_train)
        print("\ndone\n")
    print("Copying train items")
    time.sleep(2)
    imgs = train['Img_id']
    for f in tqdm(files):
        folder = f.split("/")[-1]
        for j in imgs:
            if folder == j:
                destination = root_train
                if not os.path.exists(destination):
                    os.makedirs(destination)
                shutil.copy(f, destination)
    
    print("processing val data")
    if os.path.exists(root_val) and os.path.isdir(root_val):
        print("Found existing directory, deleting...")
        shutil.rmtree(root_val)
        print("\ndone\n")
    print("Copying val items")
    time.sleep(2)
    imgs = val['Img_id']
    for f in tqdm(files):
        folder = f.split("/")[-1]
        for j in imgs:
            if folder == j:
                destination = root_val
                if not os.path.exists(destination):
                    os.makedirs(destination)
                shutil.copy(f, destination)
    
    print("processing test data")
    if os.path.exists(root_test) and os.path.isdir(root_test):
        print("Found existing directory, deleting...")
        shutil.rmtree(root_test)
        print("\ndone\n")
    print("Copying test items")
    time.sleep(2)
    imgs = test['Img_id']
    for f in tqdm(files):
        folder = f.split("/")[-1]
        for j in imgs:
            if folder == j:
                destination = root_test
                if not os.path.exists(destination):
                    os.makedirs(destination)
                shutil.copy(f, destination)

if "__name___" == "__main__":
    csv_file = "task_3/task3_train.csv"
    root = "CelebA/Img/img_align_celeba/*.jpg"
    root_train = "task_3/root/train"
    root_val = "task_3/root/val"
    root_test = "task_3/root/test"
    data_split(csv_file,root,root_train,root_val,root_test)      