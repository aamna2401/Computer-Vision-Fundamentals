#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:40:19 2022

@author: aamna
"""

# Import the library modules
import os
import cv2 
import sys
import numpy as np
import matplotlib.pyplot as plt


#get cascade xml file
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_alt.xml')

#face detection in single image
original_image = cv2.imread("images/index.jpeg")
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(haar_model) 
detected_faces = face_cascade.detectMultiScale(grayscale_image)
for (column, row, width, height) in detected_faces:
    cv2.rectangle(original_image,(column, row),(column + width, row + height),(0, 255, 0),4)    
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()


# Specify the path where Haar Cascade XML file is located
#cascPath = 'images/gray.png'
faceCascade = cv2.CascadeClassifier(haar_model)

cap = cv2.VideoCapture("shibuya.mp4")
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()

while(cap.isOpened()):
    ret, frame = cap.read()
    # Converts each frame to grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # You must enter the values for the parameters denoted with an x
    faces = faceCascade.detectMultiScale(gray,scaleFactor=0.1)

    # Drawing rectangle around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Video', frame)

    # Quit the program with the 'q' key is pressed

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when program terminates
cap.release()
cv2.destroyAllWindows()