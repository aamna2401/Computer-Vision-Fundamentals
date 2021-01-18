# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:56:56 2021

@author: aamna
"""

import skimage.io
import skimage.filters
import sys

image = skimage.io.imread(fname=path)
skimage.io.imshow(image)

blurred = skimage.filters.gaussian(image, sigma=(sigma, sigma), truncate=3.5, 
                                   multichannel=True)

skimage.io.imshow(blurred)

#nms_result = np.multiply(nms_result, 255.0 / nms_result.max()) 
#convert gray scale to rgb
dim = np.zeros(nms_result.shape, dtype=np.uint8)
dim_result = np.stack((nms_result,dim, dim), axis=2).astype(int)
# Get 256 entries from "viridis" or any other Matplotlib colormap
colmap = cm.get_cmap('gist_gray', 256)
# Make a Numpy array of the 256 RGB values
# Each line corresponds to an RGB colour for a greyscale level
np.savetxt('cmap.csv', (colmap.colors[...,0:3]*255).astype(np.uint8), fmt='%d', delimiter=',')
# Make output image, same height and width as grey image, but 3-channel RGB
result = np.zeros((*dim.shape, 3), dtype=np.uint8)   
# Load RGB LUT from CSV file
lut = np.loadtxt('cmap.csv', dtype=np.uint8, delimiter=',')
# Take entries from RGB LUT according to greyscale values in image
image = np.take(lut, gradient.astype(int) , axis=0, out=result)