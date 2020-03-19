#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

image_dir = '/home/Kxy/Lane_Segmentation_Project/data/Image_Data'
label_dir = '/home/Kxy/Lane_Segmentation_Project/data/Gray_Label'
'''
make train & validation lists
'''
label_list = []
image_list = []

for s1 in os.listdir(image_dir):
    # print(s1)
    image_sub_dir1 = os.path.join(image_dir, s1)#'Image_Data/Road02'
    label_sub_dir1 = os.path.join(label_dir, 'Label_' + str.lower(s1), 'Label')#'Gray_Label/Label_road02'
    #print('sub1',image_sub_dir1, label_sub_dir1)

    for s2 in os.listdir(image_sub_dir1):
        # print(s2)
        image_sub_dir2 = os.path.join(image_sub_dir1, s2)       #'Image_Data/Road02/ColorImage_road02'
        label_sub_dir2 = os.path.join(label_sub_dir1, s2)       #''Gray_Label/Label_road02/Label'
        #print('sub2',image_sub_dir2, label_sub_dir2)

        for s3 in os.listdir(image_sub_dir2):
            #print(s3)
            image_sub_dir3 = os.path.join(image_sub_dir2, s3)   #'Image_Data/Road02/ColorImage_road02/ColorImage'
            label_sub_dir3 = os.path.join(label_sub_dir2, s3)   #''Gray_Label/Label_road02/Label'
            #print('sub3',image_sub_dir3, label_sub_dir3)

             
            for s4 in os.listdir(image_sub_dir3):
                s44 = s4.replace('.jpg', '_bin.png')            
                image_sub_dir4 = os.path.join(image_sub_dir3, s4)
                label_sub_dir4 = os.path.join(label_sub_dir3, s44)
                if not os.path.exists(image_sub_dir4):
                    print('image not exists', image_sub_dir4)
                    continue
                if not os.path.exists(label_sub_dir4):
                    print('label not exists', label_sub_dir4)
                    continue
                #image和label的list中添加数据集
                image_list.append(image_sub_dir4)
                label_list.append(label_sub_dir4)

assert len(image_list) == len(label_list)#判断两个列表长度是否一致
print(len(image_list),len(label_list))
            
save = pd.DataFrame({'image': image_list, 'label': label_list})
save_shuffle = shuffle(save)

#划分train、validation、test 7：1:2
length = len(save_shuffle)
train = save_shuffle[0:int(length * 0.7)]
test = save_shuffle[int(length * 0.7):int(length*0.9)]
validation = save_shuffle[int(length*0.9):]

train.to_csv('../data_list/train.csv', index=False)
test.to_csv('../data_list/test.csv',index=False)
validation.to_csv('../data_list/val.csv',index = False)