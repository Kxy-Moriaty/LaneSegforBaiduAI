#!/usr/bin/python
# -*- coding: utf-8 -*-
from tqdm import tqdm
import torch
import os
import shutil
from utils.metric import compute_iou
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug
from utils.image_process import ScaleAug, CutOut, ToTensor
from utils.loss import MySoftmaxCrossEntropyLoss
from model.deeplabv3plus import DeeplabV3Plus
from model.unet import ResNetUNet
from config import Config

device_list = [0,1]
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
train_dataset = LaneDataset("train.csv", transform=transforms.Compose([ImageAug(), DeformAug(),
                                                                              ScaleAug(), CutOut(32, 0.5), ToTensor()]))
train_data_batch = DataLoader(train_dataset, batch_size=2*len(device_list), shuffle=True, drop_last=True, **kwargs)
                                                                              

val_dataset = LaneDataset("val.csv", transform=transforms.Compose([ToTensor()]))

val_data_batch = DataLoader(val_dataset, batch_size=2*len(device_list), shuffle=False, drop_last=False, **kwargs)                       


dataprocess = tqdm(train_data_batch)
for batch_item in dataprocess:
    image,mask = batch_item['image'],batch_item['mask']


