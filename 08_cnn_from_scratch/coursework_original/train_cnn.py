# -*- coding: utf-8 -*-
"""
Created on Tue May  4 23:45:35 2021

@author: aamna
"""

import numpy as np
import time 
import os
from glob import glob
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import cv2
from matplotlib import pyplot
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
import pickle

########################################## Dataset Loading ###################################################### 
def loadDataset(path, train = True):
    print('Loading Dataset...')
    if train == True:
        data_path = os.path.join(path, 'train')
    else:
        data_path = os.path.join(path, 'test')
    classes = os.listdir(data_path)
    t = time.time()
    data_imgs, data_x, data_y =  [], [], []
    for i in classes:
            filename = os.listdir(os.path.join(data_path, i))
            filename = glob(os.path.join(data_path, i , '*.png'))
            data_imgs = data_imgs + filename
            labels = [i] * len(filename)
            data_y = data_y + labels
            data_index = list(range(int(len(data_imgs))))
    for index in data_index :
            img_path = data_imgs[index]
            im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            data_x.append(im)
    print('Dataset loaded...')
    print('\n\nTotal time taken in loading: ', (time.time()-t)//60, '\n\n')
    return np.array(data_x), np.array(data_y)

# plot first few images
def plot_images(x):
    for i in range(5):
    	# define subplot
    	pyplot.subplot(330 + 1 + i)
    	# plot raw pixel data
    	pyplot.imshow(x[i], cmap = pyplot.get_cmap('gray'))
    # show the figure
    pyplot.show()
    
#function to plot results of operations   
def plot_results(output):
    for i in range(0,5):
        plt.imshow(output[:,:,35],cmap='gray')
        plt.show()
        
########################################## Data Preprocessing ####################################################
#label Encoding
def label_encoding(labels):
    ohe = preprocessing.OneHotEncoder()
    # Reshape data
    labels = labels.reshape(-1, 1)
    # Fit and transform  data
    ohe.fit(labels)
    transformed_labels = ohe.transform(labels).toarray()
    return transformed_labels

#################################### Convolution Operation ##################################################### 
#get image patches for filter operations and save them for convolutional operation
def get_patch(img_data, filter_dims): #generator function
  height = img_data.shape[0]
  width = img_data.shape[1]
  for i in range(height - filter_dims + 1):
    for j in range(width - filter_dims +1):
        patch = img_data[i:(i+filter_dims),j:(j+filter_dims)]
        yield patch, i, j
        
def convolution_operation(img_data, no_of_filters, filter_dims, pad, stride):  
    #initilize filters
    filters = np.random.randn(no_of_filters, filter_dims, filter_dims)/(filter_dims * filter_dims)
    #Forward Propagation
    height, width = img_data.shape
    feature_map = np.zeros((height - filter_dims + 1, width - filter_dims+1, no_of_filters))
    for patch, i,j in get_patch(img_data, filter_dims):
        #image_patch = (image_patch.reshape(train_x.shape[0], -1))
        feature_map[i,j]= np.sum(patch * filters, axis= (1,2))
    #check ouput dimensions
    n_h = (height - filter_dims + 2 * pad / stride ) + 1
    n_w = (width - filter_dims + 2 * pad / stride ) + 1
    return feature_map, n_h, n_w

#################################### Max Pooling ############################################################### 
#get patches for pooling operations
def pool_patch(feature_map, filter_size, stride = 2):
    height = feature_map.shape[0] // filter_size
    width = feature_map.shape[1] // filter_size
    for i in range(height):
      for j in range(width):
        image_patch = feature_map[(i*filter_size):(i*filter_size + filter_size), (j*filter_size):(j*filter_size + filter_size)]
        yield image_patch, i, j
#forward pooling
def pool_forward(feature_map, mode , filter_size, stride, padding):
    #apply padding
    feature_map = np.pad(feature_map, padding, mode='constant')
    height, width, num_filters = feature_map.shape
    feature_volume = np.zeros((height // filter_size, width // filter_size, num_filters))
    for image_patch, i, j in pool_patch(feature_map, 2):
      feature_volume[i,j] = np.amax(image_patch, axis = (0,1))
    #check ouput dimensions
    n_h = (height - filter_size / stride) + 1
    n_w = (width - filter_size / stride) + 1
    return feature_volume, n_h, n_w

#################################### Fully Connected Layer ######################################################
def relu(s):
    s = np.maximum(0, s)
    return s # apply relu function on s and return it's value

def relu_derivative(s):
    s = 1 * (s > 0)
    return s  # apply derivative of relu on s and return it's value
  
def softmax(s):
        n = s - np.max(s, axis=1, keepdims=True)
        e_n = np.exp(n)
        return e_n / np.sum(e_n, axis=1, keepdims=True)
    
def softmax_derivative(s):
    s = 1 * (s > 0)
    return s  # apply derivative of relu on s and return it's value

def flatten_inputs(x):
    x_flatten = x.reshape(x.shape[0], -1)
    return x_flatten
   
def fully_connected_forward(feature_volume, hidden_node, output_node):
    #input flattening
    feature_vector = feature_volume.flatten()
    input_node = feature_vector.shape[0]
    #weights and biases initialization
    w1 = np.random.rand(input_node, hidden_node)
    b1 = np.zeros(hidden_node)
    w2 = np.random.rand(hidden_node, output_node)
    b2 = np.zeros(output_node)
    #feed forward propagation
    l1 = np.dot(input_node, w1) + b1
    z1 = relu(l1)
    l2 = np.dot(z1, w2) + b2
    z2 = softmax(l2)
    return z2,w2

################################# compute accuracy ########################################
def predict(feature_volume, hidden_node, output_node):
    Y_pred,_ = fully_connected_forward(feature_volume, hidden_node, output_node)
    Y_pred = np.argmax(Y_pred, axis=0)
    return Y_pred

def accuracy(X, Y, feature_volume):
    Y_pred = np.argmax(Y, axis=0)
    Y_pred = predict(feature_volume, hidden_node = 30, output_node = 10)
    return np.equal(Y, Y_pred).sum() / X.shape[0]

################### Focal loss implementation ###################################################
def crossentropy(Y, Y_pred):
        true_label = Y.shape[0]
        Y_pred = Y_pred + 1e-20
        loss = (1/true_label)* np.sum(-1*Y*np.log(Y_pred))
        return loss
    
def focal_loss(alpha , gamma, y_pred, labels):
    #formula of 'The class-balanced (CB) Focal Loss'  is:
    N = len(labels)
    focal_loss = -((1 - alpha)/(1 - alpha * np.exp(10))) * np.sum(1 - y_pred)* np.exp(gamma) * np.log(y_pred)
    loss = focal_loss/ N
    return loss

def focal_loss_deriv(Y, y_pred, gamma):
    m = len(Y)
    #loss =  Y * (gamma * (1 - y_pred) * np.exp(gamma - 1) * np.log(y_pred) - ((1 - y_pred) * np.exp(gamma))/y_pred)
    loss = gamma * Y - y_pred * np.exp(gamma)
    d_loss = np.sum(loss, axis=0, keepdims=True) / m 
    return d_loss

#################################### Backward Propagation ######################################################
def manual_fc_backprop(feature_volume, dl_do, feature_vector):
    output,w2 = fully_connected_forward(feature_volume, 30, 10)
    for i, grad in enumerate(dl_do):
      if grad[i] ==0:
        continue
    #softmax derivative
    transformation_eq = np.exp(output)
    S_total = np.sum(transformation_eq)
    #gradients with respect to out (z)[softmax_layer]
    dy_dz2 = -transformation_eq[i]*transformation_eq / (S_total **2)
    dy_dz2[i] = transformation_eq[i]*(S_total -  transformation_eq[i]) / (S_total **2)
    #multiplied softmax derivative with error(dl/dy * dy/dz2)
    dl_dz2 = grad * dy_dz2
    #rate of change in hidden layer(dl/dz2 * dz2/dw2)
    dl_dw2 = np.dot(dl_dz2, w2.T) #check ye step thk h ya nhi 
    #multiply dl_dw2 with dw2/dz1(relu deriv)
    dl_dz1 = dl_dw2 * relu_derivative(dl_dw2)
    #multiply dl_dz1 and dz1/dw1
    dl_dw1 = np.multiply(dl_dz1.T, feature_vector).T
    return dl_dw1, dl_dw2


############### main function ###############################
#Load Train and test data
train_x, train_y = loadDataset("MNIST_Imbalanced_Dataset", train = True)
test_x, test_y = loadDataset("MNIST_Imbalanced_Dataset", train = False)
#plot iamges to check dataset loading
plot_images(train_x)

#onehotencoded labels
train_y = label_encoding(train_y)
test_y = label_encoding(test_y)

#flatten feature volume
feature_vector = flatten_inputs(feature_volume)
#extract train images and labels
def forward_propagation(train_x, train_y):
    for i, (images, labels) in enumerate(zip(train_x, train_y)):
        #convolutional layer output
        feature_map, h_out_dim, w_out_dim = convolution_operation(images, 36, 17, 0, 1)
        #visualize convolutional layer output
        #plot_results(feature_map)
        #maxpool layer output
        feature_volume, h_out_dim, w_out_dim, = pool_forward(feature_map, mode = 'max', filter_size = 2, stride = 2, padding = 0)
        #plot_results(feature_volume)
        #flatten feature volume
        feature_vector = flatten_inputs(feature_volume)
        #Fully connected layer output
        output, w2 = fully_connected_forward(feature_volume, 30, 10)
        #predict value
        y_pred = predict(feature_volume, 30, 10)
        #calculate  accuracy
        train_accuracy = accuracy(images, labels, feature_volume)
        #derivative of focal loss
        dl_do = focal_loss_deriv(labels, output, 2)       
        return y_pred, train_accuracy , dl_do
                
    
    
