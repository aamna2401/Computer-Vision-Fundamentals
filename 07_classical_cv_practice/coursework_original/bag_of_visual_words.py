#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:44:55 2021

@author: aamna
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pylab as pl
from sklearn.metrics import confusion_matrix,accuracy_score


train_path="dataset"
class_names=os.listdir(train_path)

image_paths=[]
image_classes=[]

#Function to List all the filenames in the directory
def img_list(path):
    return (os.path.join(path,f) for f in os.listdir(path))

for training_name in class_names:
    dir_= os.path.join(train_path,training_name)
    class_path = img_list(dir_)
    image_paths += class_path
   
image_classes_0=[0]*(len(image_paths)//2)
image_classes_1=[1]*(len(image_paths)//2)
image_classes=image_classes_0+image_classes_1

#Append all the image path and its corresponding labels in a list
D=[]
for i in range(len(image_paths)):
    D.append((image_paths[i],image_classes[i]))
    

#Shuffle Dataset and split into Training and Testing
dataset = D
random.shuffle(dataset)
train = dataset[:180]
test = dataset[180:]

image_paths, y_train = zip(*train)
image_paths_test, y_test = zip(*test)

#Feature Extraction using ORB
des_list=[]
orb=cv2.ORB_create()
im=cv2.imread(image_paths[1])
plt.imshow(im)

#Function for plotting keypoints
def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
            x, y = kp.pt
            plt.imshow(cv2.circle(vis, (int(x), int(y)), 2, color))
            
kp = orb.detect(im,None)
kp, des = orb.compute(im, kp)
img=draw_keypoints(im,kp)

#Appending descriptors of the training images in list
for image_pat in image_paths:
    im=cv2.imread(image_pat)
    kp=orb.detect(im,None)
    keypoints,descriptor= orb.compute(im, kp)
    des_list.append((image_pat,descriptor))

descriptors=des_list[0][1]
for image_path,descriptor in des_list[1:]:
    descriptors=np.vstack((descriptors,descriptor))
    
descriptors.shape
descriptors_float=descriptors.astype(float)

#Performing K Means clustering on Descriptors
from scipy.cluster.vq import kmeans,vq
k=200
voc,variance=kmeans(descriptors_float,k,1)

#Creating histogram of training image
im_features=np.zeros((len(image_paths),k),"float32")
for i in range(len(image_paths)):
    words,distance=vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w]+=1
        
#Applying standardisation on training feature
from sklearn.preprocessing import StandardScaler
stdslr=StandardScaler().fit(im_features)
im_features=stdslr.transform(im_features)

#Creating Classification Model with SVM
from sklearn.svm import LinearSVC
clf=LinearSVC(max_iter=80000)
clf.fit(im_features,np.array(y_train))

#Testing the Classification Model
des_list_test=[]
for image_pat in image_paths_test:
    image=cv2.imread(image_pat)
    kp=orb.detect(image,None)
    keypoints_test,descriptor_test= orb.compute(image, kp)
    des_list_test.append((image_pat,descriptor_test))
    
len(image_paths_test)

from scipy.cluster.vq import vq
test_features=np.zeros((len(image_paths_test),k),"float32")
for i in range(len(image_paths_test)):
    words,distance=vq(des_list_test[i][1],voc)
    for w in words:
        test_features[i][w]+=1
        
test_features=stdslr.transform(test_features)

true_classes=[]
for i in y_test:
    if i==1:
        true_classes.append("Cat")
    else:
        true_classes.append("Dog")
        
predict_classes=[]
for i in clf.predict(test_features):
    if i==1:
        predict_classes.append("Cat")
    else:
        predict_classes.append("Dog")
        
clf.predict(test_features)

accuracy=accuracy_score(true_classes,predict_classes)
print(accuracy)
    
