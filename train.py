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


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

device_list = [0,1,2,3]
train_net = 'deeplabv3p'
nets = {'deeplabv3p': DeeplabV3Plus, 'unet': ResNetUNet}

Resume = False

def train_epoch(net, epoch, dataLoader, optimizer, trainF, config):
    net.train()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        optimizer.zero_grad()
        out = net(image)
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(out, mask)
        total_mask_loss += mask_loss.item()
        mask_loss.backward()
        optimizer.step()
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))
    trainF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    trainF.flush()


def test(net, epoch, dataLoader, testF, config):
    net.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    result = {"TP": {i:0 for i in range(8)}, "TA":{i:0 for i in range(8)}}
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        out = net(image)
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(out, mask)
        total_mask_loss += mask_loss.detach().item()
        pred = torch.argmax(F.softmax(out, dim=1), dim=1)
        result = compute_iou(pred, mask, result)
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss))
    testF.write("Epoch:{} \n".format(epoch))
    for i in range(8):
        result_string = "{}: {:.4f} \n".format(i, result["TP"][i]/result["TA"][i])
        print(result_string)
        testF.write(result_string)
    testF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    testF.flush()


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
    lane_config = Config()
    if not os.path.exists(lane_config.SAVE_PATH):
        #shutil.rmtree(lane_config.SAVE_PATH)
        os.makedirs(lane_config.SAVE_PATH, exist_ok=True)
    trainF = open(os.path.join(lane_config.SAVE_PATH, "train.csv"), 'w')
    testF = open(os.path.join(lane_config.SAVE_PATH, "test.csv"), 'w')
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = LaneDataset("train.csv", transform=transforms.Compose([ImageAug(), DeformAug(),
                                                                              ScaleAug(), CutOut(32, 0.5), ToTensor()]))

    train_data_batch = DataLoader(train_dataset, batch_size=2*len(device_list), shuffle=True, drop_last=True, **kwargs)
    val_dataset = LaneDataset("val.csv", transform=transforms.Compose([ToTensor()]))

    val_data_batch = DataLoader(val_dataset, batch_size=2*len(device_list), shuffle=False, drop_last=False, **kwargs)
    
    
    net = nets[train_net](lane_config)
    optimizer = torch.optim.Adam(net.parameters(), lr=lane_config.BASE_LR, weight_decay=lane_config.WEIGHT_DECAY)

    epoch_to_continue = 0#
    if Resume is True:     
        checkpoint_path = os.path.join(os.getcwd(), lane_config.SAVE_PATH, "epoch{}Net.pth.tar".format(epoch_to_continue))
        checkpoint = torch.load(checkpoint_path)
        #model_param = torch.load(checkpoint_path)['state_dict']
        #model_param = {k.replace('module.', ''):v for k, v in model_param.items()}
        net.load_state_dict(checkpoint['state_dict']) #加载net参数
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_to_continue = checkpoint['epoch']


    if torch.cuda.is_available():
        net = net.cuda(device=device_list[0])

        #在这里加了一个数据并行，相当于甲类一个moduel
        net = torch.nn.DataParallel(net, device_ids=device_list)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lane_config.BASE_LR,
    #                             momentum=0.9, weight_decay=lane_config.WEIGHT_DECAY)


    for epoch in range(epoch_to_continue,epoch_to_continue+lane_config.EPOCHS):
        # adjust_lr(optimizer, epoch)
        train_epoch(net, epoch, train_data_batch, optimizer, trainF, lane_config)
        if epoch % 5 == 0:
            #存储的参数是net的模型参数，没有网络结构
            #torch.save({'state_dict': net.module.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "laneNet{}.pth.tar".format(epoch)))
            #torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "laneNet{}.pth.tar".format(epoch)))
            torch.save({
                        'epoch':epoch,
                        'state_dict': net.module.state_dict(),#加了module
                        'optimizer_state_dict':optimizer.state_dict(),
                       },os.path.join(os.getcwd(), lane_config.SAVE_PATH, "epoch{}Net.pth.tar".format(epoch)))
        
        test(net, epoch, val_data_batch, testF, lane_config)
        
        
    
    trainF.close()
    testF.close()
    #torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "finalNet.pth.tar"))

   

if __name__ == "__main__":
    main()
