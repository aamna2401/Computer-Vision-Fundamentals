#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:34:20 2022

@author: aamna
"""

import shutil, os
import pandas as pd
import time
from glob import glob

#read images into csv and find identities
path = "CelebA/Anno/identity_CelebA.txt"
df = pd.read_csv(path, sep=" ", names=['image_name','Class'])
labels = df.sort_values('Class')
actual_names = list(labels.Class.unique())
#as got 55 identities earlier so 120-54
subset = df.sample(n = 120, replace=False)
subset.shape
sampled_labels = subset.sort_values('Class')
class_names = list(sampled_labels.Class.unique())

for c in class_names:
    for i in list(labels[labels['Class']==c]['image_name']):
        #create path to the image 
        get_image = os.path.join('CelebA/Img/img_align_celeba/',str(i))
        dest = 'root/'+str(c)
        print("Copying {0} image to {1} ".format(i, c))
        time.sleep(2)
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.copy(get_image,dest)

 #check total number       
files = glob('root/**/*.jpg')

#deleted root
# for filename in os.listdir('root'):
#     print(filename)
#     os.remove(filename)
#     print('done')

# import numpy as np
# import pandas as pd
# df = pd.DataFrame(np.random.random((200, 3)))
# df["date"] = pd.date_range("2000-1-1", periods=200, freq="D")
# mask = (df["date"] > "2000-6-1") & (df["date"] <= "2000-6-10")
# print(df.loc[mask])

# #make subset on the basis of some condition
# #get random nums
# random_nums = random.sample(range(55, 10000), 120)
# #sampling on the basis of condition 
# condition = df['Class'] > 55
# true_index = condition[condition == True].index
# as got 55 identities earlier so 120-54 = 65
# subset = df[condition].sample(n = 65, replace=False)
# labels = subset.sort_values('Class')
# class_names = list(labels.Class.unique())
# for i in range(0,55):
#     class_names.append(i)
# class_names.sort()
# #remove extra dirs other than needed
# for i in class_names:
#     for j in actual_names:
#         print(j)
#         if i not in actual_names:
#             os.rmdir('root/'+str(j))
#             print("removed {0} dir".format(i))
#         else:
#             pass
# #logic # 2
# for filename in os.listdir('root'):
#     print(filename)
#     if filename not in class_names:
#         os.remove(filename)
#     print('done')

