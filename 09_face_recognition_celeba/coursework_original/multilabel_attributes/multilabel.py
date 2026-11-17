#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:47:43 2022

@author: aamna
"""
import pandas as pd
import glob

nRowsRead = 1000 # specify 'None' if want to read whole file
# list_attr_celeba.csv has 202599 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('CelebA/Anno/list_attr_celeba.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'list_attr_celeba.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')





path = 'CelebA/Anno/list_attr_celeba.csv'
file = open(path, 'r')
files = file.readlines()
file.close()
    
import csv  

header = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 
          'Bald','Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair', 
          'Blurry','Brown_Hair','Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses',
          'Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open',
          'Mustache','Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin','Pointy_Nose',
          'Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling','Straight_Hair',
          'Wavy_Hair','Wearing_Earrings','Wearing_Hat','Wearing_Lipstick',
          'Wearing_Necklace','Wearing_Necktie','Young']



lst = []


for i in range(0,len(header)):
    for str in header:
      lst.append([str, i])  
#
features = []

with open('attr.csv', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    # write the header
    writer.writerow(header)
    for data in files:
        # write the data
        writer.writerow(data)
    f.close()
        
labels_df = pd.read_csv(path)
labels_df.drop(labels_df['202599'],axis=1, 
        inplace=True)

label_dict = {}
for i in range(1, len(labels_df)):
    for x in labels_df['202599'][1:]:
        print(x)

for i in range(0, len(labels_df)):
    label_dict[labels_df['202599'][i].split()[0]] = [x for x in labels_df['202599'][i].split()[1:]]

label_df = pd.DataFrame(label_dict).T
label_df.columns['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 
          'Bald','Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair', 
          'Blurry','Brown_Hair','Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses',
          'Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open',
          'Mustache','Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin','Pointy_Nose',
          'Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling','Straight_Hair',
          'Wavy_Hair','Wearing_Earrings','Wearing_Hat','Wearing_Lipstick',
          'Wearing_Necklace','Wearing_Necktie','Young']

label_df.replace(['-1'], ['0'], inplace = True)
label_df.columns
label_df.head()

train_files = glob.glob('data/train/**/*.jpg')

img_names = []
for f in train_files:
    img = f.split('/')[-1]

#make subset
label_df.columns
