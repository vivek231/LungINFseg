# -*- coding: utf-8 -*-
import os
import math
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
from model import LungINFseg
from loss import iou_binary
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from collections import OrderedDict

MainPath='/LungINFseg/train/'
image_pth='a/'      
gt_path='b/'         

ImageList=[]
AnnList=[]
for file in os.listdir(MainPath+'/'+image_pth):
    if file.endswith('png'):
        ImageList.append(file)
        main_part=file.split('.png')[0]
        main_part=main_part+'.png'
        AnnList.append(main_part)

classes = 1
model  = LungINFseg(classes=1)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print ("\nTrainable parameters", pytorch_total_params)
# print (model)
model.cuda()
batch_size = 8
weight_decay = 1e-4
epochs_num = 100
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
criterion = nn.BCEWithLogitsLoss().cuda()

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def soft_dice_loss(inputs, targets):
        num = targets.size(0)
        m1  = inputs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1 - score.sum()/num
        return score

for epoch in range(epochs_num):
    batches_num = 0
    total_error=0
    for  i in range(0,len(ImageList)-batch_size,batch_size):
        batches_num = batches_num+1
        ImageBatch=ImageList[i:i+batch_size]
        AnnBatch=AnnList[i:i+batch_size]
        ImageArr =[]
        AnnArr=[]
        for a  in (ImageBatch):
           img=cv2.imread(MainPath+'/'+image_pth+'/'+a)
           img=cv2.resize(img,(256,256))
           ann = cv2.imread(MainPath+'/'+gt_path+'/'+a)
           ann=ann[:,:,0]
           ann = cv2.resize(ann,(256,256))
           ret,ann = cv2.threshold(ann,0,255,cv2.THRESH_BINARY)
           ann=ann/255
           img=img/255.
           ImageArr.append(img)
           AnnArr.append(ann)
        ImageArr=np.array(ImageArr)
        ImageArr=np.transpose(ImageArr,(0,3,1,2))
        #!-------------------------------------------------!
        ImageArr[:,0,:,:]=ImageArr[:,0,:,:]-0.485
        ImageArr[:,1, :, :] = ImageArr[:, 1, :, :] - 0.456
        ImageArr[:,2, :, :] = ImageArr[:, 2, :, :] - 0.406
        #!-------------------------------------------------!
        ImageArr[:, 0, :, :] = ImageArr[:, 0, :, :] /0.229
        ImageArr[:, 1, :, :] = ImageArr[:, 1, :, :] /0.224
        ImageArr[:, 2, :, :] = ImageArr[:, 2, :, :] /0.225
        #!-------------------------------------------------!
        ImageArr = Variable(torch.FloatTensor(ImageArr)).cuda()
        AnnArr = np.array(AnnArr,dtype=np.int64)
        AnnArr = Variable(torch.from_numpy(AnnArr)).cuda()
        e1,e2,e3,e4,out = model(ImageArr)
        AnnArr = AnnArr.unsqueeze(1)
        AnnArr = AnnArr.type(torch.float)
        
        out2 = F.sigmoid(out)
        
        out2_e1 = F.sigmoid(e1)
        AnnArr_1 = F.interpolate(AnnArr, size=(64, 64), mode='bicubic', align_corners=False)
        
        out2_e2 = F.sigmoid(e2)
        AnnArr_2 = F.interpolate(AnnArr, size=(32, 32), mode='bicubic', align_corners=False)
       
        out2_e3 = F.sigmoid(e3)
        AnnArr_3 = F.interpolate(AnnArr, size=(16, 16), mode='bicubic', align_corners=False)
        
        out2_e4 = F.sigmoid(e4)
        AnnArr_4 = F.interpolate(AnnArr, size=(8, 8), mode='bicubic', align_corners=False)

        #!------------------------Loss_function_computation-------------------------!
        loss1 = 200*criterion(out,AnnArr)
        # ------------------------------------------
        loss2 = 150*dice_loss(out2_e1,AnnArr_1.long())
        loss3 = 100*dice_loss(out2_e2,AnnArr_2.long())
        loss4 = 100*dice_loss(out2_e3,AnnArr_3.long())
        loss5 = 100*dice_loss(out2_e4,AnnArr_4.long())
        loss6 = 100*dice_loss(out2,AnnArr.long())

        loss = loss1+loss2+loss3+loss4+loss5+loss6
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_error = total_error + loss.item()
        
        if (i + batch_size) % 40 == 0:
            out2 = out2.cpu()
            dd = out2.data
            AnnArr = AnnArr.cpu().data
            print ('Epoch:', epoch, ',avrLoss ', loss.item(), ', step:', i, ' of ', len(ImageList))
    torch.save(model.state_dict(), 'Net.model')


