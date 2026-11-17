#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:56:35 2021

@author: aamna
"""

import cv2
import numpy as np

# Load the images
img1 = cv2.imread('images/book.jpg')
img2 = cv2.imread('images/book.jpg')

# Calculate the histograms, and normalize them
#cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
hist_img1 = cv2.calcHist([vocab_size], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
hist_img2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#hist,bins = np.histogram(img.ravel(),256,[0,256])

#chi square distance
def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])
	# return the chi-squared distance
	return d


#d_metrics = chi2_distance(histA, histB)

import matplotlib.pyplot as plt

#create list of data
x = [2, 4, 4, 5, 6, 6, 7, 7, 7, 8, 8, 8, 12, 13]

#specify bin start and end points
bin_ranges = [0, 5, 10, 15]

#create histogram with 4 bins
_ = plt.hist(x, 4, edgecolor='black')
