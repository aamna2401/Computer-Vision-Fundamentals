# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 23:28:31 2021

@author: aamna
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from matplotlib import cm
import math
import cv2


########### Image Preprocessing ###########
def img_read(path):
    img = plt.imread(path)
    return img
    
def rgb_to_grayscale(img):
    #imgGray = np.round(np.mean(img, -1))
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    imgGray = np.round(0.2989 * R + 0.5870 * G + 0.1140 * B)
    return imgGray

def calculate_filter_size(sigma, T):
    sHalf = round(np.sqrt(-(np.log(T))*2*sigma**2))
    N = 2*sHalf + 1
    [X, Y] = np.mgrid[-sHalf:sHalf+1, -sHalf:sHalf+1]
    normalize = 1 / (2.0 * np.pi * sigma**2)
    gau = np.exp(-((X**2 + Y**2) / (2.0*sigma**2))) * normalize
    #gaussian derivative
    #derivative w.r.t x
    Gx = np.round((-X / sigma**2)*(np.exp(-((X**2 + Y**2) / 
                                            (2.0*sigma**2))))*255)
    Gy = np.round((-Y / sigma**2)*(np.exp(-((X**2 + Y**2) / 
                                            (2.0*sigma**2))))*255)
    return Gx, Gy

#applying gaussian mask to images
def convolution(img,Gx,Gy):
    fx = ndimage.filters.convolve(img, Gx)
    fy = ndimage.filters.convolve(img, Gy)
    return fx, fy

def calculate_gradient (fx,fy):
    gradient = np.sqrt(fx**2, fy**2)
    gradient = gradient / gradient.max() * 255
    angle =  np.arctan2(fy, fx)
    theta = angle/math.pi * 180 + 180
    return gradient, theta

def NMS(gradient, theta):
    img_row, img_col = gradient.shape
    nms_result = np.zeros(gradient.shape)
    for row in range(1, img_row-1):
        for col in range(1, img_col-1):
            direction = theta[row, col]
            
            if (0 <= direction < 22.5) or (157.5 <= direction < 202.5 ) or (337.5 <= direction < 360):
                 (gradient[row, col-1]) or (gradient[row, col+1]) == 0
            elif (22.5 <= direction < 67.5) or (202.5 <= direction < 247.5 ):
                (gradient[row-1, col-1]) or (gradient[row+1, col+1]) == 1
            elif (67.5 <= direction < 112.5) or (247.5 <= direction < 292.5 ):
                (gradient[row-1, col]) or (gradient[row+1, col]) == 2
            else:
                (gradient[row+1, col-1]) or (gradient[row-1, col+1]) == 3
                
            #if gradient[row, col] >= forth_pixel and gradient[row, col] >= next_pixel:
                nms_result[row, col] = gradient[row, col]
                
       
        
    return  nms_result        

  
if __name__ == "__main__":
    #image path
    path = 'test_images/circle.jpg'
    #read image
    img = img_read(path)
    plt.imshow(img)
    #convert to grayscale
    imgGray = rgb_to_grayscale(img)
    plt.imshow(imgGray , cmap='gray')
    #Filter size
    Gx, Gy = calculate_filter_size(sigma=0.5, T=0.3)
    #convolving images with gaussian filter
    fx, fy = convolution(imgGray,Gx,Gy)
    plt.imshow(fx , cmap='gray')
    plt.imshow(fy, cmap='gray')
    #gradient calculation
    gradient, theta = calculate_gradient(fx,fy)
    plt.imshow(gradient, cmap='gray')
    plt.imshow(theta, cmap=cm.coolwarm)
    #Non-maximum supression
    nms_result = NMS(gradient, theta)
    plt.imshow(nms_result, cmap='gray')
    
    