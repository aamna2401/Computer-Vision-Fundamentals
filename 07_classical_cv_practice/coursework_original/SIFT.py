#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 23:34:02 2021

@author: aamna
"""

import cv2 as cv
import matplotlib.pyplot as plt

#load image
image = cv.imread("images/book.jpg")

#convert to grayscale image
gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#initialize SIFT object
sift = cv.xfeatures2d.SIFT_create()

#detect keypoints
keypoints, _= sift.detectAndCompute(image, None)

#draw keypoints
sift_image = cv.drawKeypoints(gray_scale, keypoints, None)

plt.imshow(sift_image)

# cv.imshow("Features Image", sift_image)
# cv.waitKey(1)
