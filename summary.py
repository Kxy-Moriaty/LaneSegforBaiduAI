#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ToTensor


def adjust_lr(optimizer, epoch):
    if epoch == 0:
        lr = 1e-3
    elif epoch == 2:
        lr = 1e-2
    elif epoch == 100:
        lr = 1e-3
    elif epoch == 150:
        lr = 1e-4
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = LaneDataset("train.csv", transform=transforms.Compose([ToTensor()]))

    train_data_batch = DataLoader(train_dataset, batch_size=8, **kwargs)
    #miou就是每个类别的iou，然后算平均
    #统计每个类的分布情况
    number_class = {i: 0 for i in range(8)}
    for item in train_data_batch:
        temp = item['mask'].numpy()
        for i in range(8):
            number_class[i] += np.sum(temp==i)
    for i in range(8):
        print("{} has number of {}".format(i, number_class[i]))


if __name__ == "__main__":
    main()
