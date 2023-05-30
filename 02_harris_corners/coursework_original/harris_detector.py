# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt
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
    #N = 2*sHalf + 1
    [X, Y] = np.mgrid[-sHalf:sHalf+1, -sHalf:sHalf+1]
    #normalize = 1 / (2.0 * np.pi * sigma**2)
    #gau = np.exp(-((X**2 + Y**2) / (2.0*sigma**2))) * normalize
    #gaussian derivative
    #derivative w.r.t x
    Gx = np.round((-X / sigma**2)*(np.exp(-((X**2 + Y**2) / 
                                            (2.0*sigma**2))))*255)
    Gy = np.round((-Y / sigma**2)*(np.exp(-((X**2 + Y**2) / 
                                            (2.0*sigma**2))))*255)
    return Gx, Gy

def convolution(imgGray, G):
    m, n = G.shape
    if (m == n):
        y, x = imgGray.shape
        y = y - m + 1
        x = x - m + 1
        convolved_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                convolved_image[i][j] = np.sum(imgGray[i:i+m, j:j+m]*G) 
    return convolved_image
    
#applying gaussian mask to images
def derivatives(img,Gx,Gy):
    fx = convolution(img, Gx)
    fy = convolution(img, Gy)
    #plt.imshow(fx , cmap='gray')
    dx2 = np.square(fx)
    dy2 = np.square(fy)
    dxy = fx*fy
    return fx, fy, dx2, dy2, dxy

def response(img, dx2, dy2, dxy, k):
    #compute response before NMS
    img_height = img.shape[0]
    img_width = img.shape[1]
    r_matrix = np.zeros((img_height,img_width))
    # determinant
    detA = dx2 * dxy - dxy ** 2
    # trace
    traceA = dx2 + dy2
    response = detA - k * traceA ** 2
    #r_matrix[img_height,img_width] = r
    plt.title("Response before NMS")
    plt.imshow(response, cmap='gray')
    return response

def harris_calculation(dx2, dy2, dxy, img, window_size, k, threshold):
    img_height = img.shape[0]
    img_width = img.shape[1]
    result_matrix = np.zeros((img_height,img_width))
    offset = int( window_size / 2 )
    for y in range(offset, img_height-offset):
        for x in range(offset, img_width-offset):
            Ix = np.sum(dx2[y-offset:y+1+offset, x-offset:x+1+offset])
            Iy = np.sum(dy2[y-offset:y+1+offset, x-offset:x+1+offset])
            Ixy = np.sum(dxy[y-offset:y+1+offset, x-offset:x+1+offset])
            
            #matrix M after applying taylor series        
            matrix_M = np.array([[Ix,Ixy],[Ixy,Iy]]) 
            
            # r = detA - k(traceD)2
            #determinant
            det=np.linalg.det(matrix_M)
            trace=np.matrix.trace(matrix_M)
            r=det - k*(trace**2)
            result_matrix[y-offset, x-offset]=r
            
    cv2.normalize(result_matrix, result_matrix, 0, 1, cv2.NORM_MINMAX)
    plt.title("R matrix")
    plt.imshow(result_matrix, cmap='gray')
    #thresholding
    for x in range(offset, img_height-offset):
        for y in range(offset, img_width-offset):
            value=result_matrix[x, y]
            if value>threshold:
                # cornerList.append([x, y, value])
                #cv2.putText(img,"+", (x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
                cv2.circle(img,(x,y),3,(0,255,0))
    # plot and save image
    plt.figure("Harris detector")
    plt.title("Harris detector")
    plt.imshow(img, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.savefig('Img2 Harris_detector_threshold_%s.png'%(threshold), 
                bbox_inches='tight')
    plt.show()


  
if __name__ == "__main__":
    #image path
    path = 'images/Img3.png'
    #read image
    img = img_read(path)
    plt.imshow(img)
    #convert to grayscale
    imgGray = rgb_to_grayscale(img)
    plt.imshow(imgGray , cmap='gray')
    #Filter size
    Gx, Gy = calculate_filter_size(sigma=0.5, T=0.3)
    #convolving images with gaussian filter
    fx, fy, dx2, dy2, dxy = derivatives(imgGray,Gx,Gy)
    # plt.title("fx")
    # plt.imshow(fx , cmap='gray')
    # plt.title("fy")
    # plt.imshow(fy , cmap='gray')
    # plt.title("dx2")
    # plt.imshow(dx2 , cmap='gray')
    # plt.title("dy2")
    # plt.imshow(dy2, cmap='gray')
    # plt.title("dxy")
    # plt.imshow(dxy, cmap='gray')
    r_value = response(img, dx2, dy2, dxy, k=0.04)
    harris_calculation(dx2, dy2, dxy, img=img, window_size=3, k=0.04, 
                       threshold=0.8)
   
    
    

