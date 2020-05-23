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
from model import COVID19Seg


MainPath='/home/vivek/dataset/covid/train/'
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

classes = 2
model  = COVID19Seg(classes=2)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print ("\nTrainable parameters", pytorch_total_params)
print (model)
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
batch_size = 4
weight_decay = 1e-4
epochs_num = 100

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
           ann = cv2.resize(ann, (256, 256))
           ann=ann/255
           img=img/255.
           ImageArr.append(img)
           AnnArr.append(ann)
        ImageArr=np.array(ImageArr)
        ImageArr=np.transpose(ImageArr,(0,3,1,2))
        ImageArr[:,0,:,:]=ImageArr[:,0,:,:]-0.485
        ImageArr[:,1, :, :] = ImageArr[:, 1, :, :] - 0.456
        ImageArr[:,2, :, :] = ImageArr[:, 2, :, :] - 0.406
        #*****************************************
        ImageArr[:, 0, :, :] = ImageArr[:, 0, :, :] /0.229
        ImageArr[:, 1, :, :] = ImageArr[:, 1, :, :] / 0.224
        ImageArr[:, 2, :, :] = ImageArr[:, 2, :, :] / 0.225
        ImageArr = Variable(torch.FloatTensor(ImageArr).cuda())
        AnnArr = np.array(AnnArr,dtype=np.int64)
        AnnArr = Variable(torch.from_numpy(AnnArr).cuda())
        out = model(ImageArr)
        _,out2=(torch.max(out,1))
        loss = criterion(out,AnnArr) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_error = total_error + loss.item()

        if (i + batch_size) % 20 == 0:
            out2 = out2.cpu()
            dd = out2.data
            AnnArr = AnnArr.cpu().data
            AnnArr = AnnArr.view(dd.size(0), 1, dd.size(1), dd.size(2))
            dd = dd.view(dd.size(0), 1, dd.size(1), dd.size(2))
            print ('Epoch:', epoch, ',avrLoss ', loss.item(), ', step:', i, ' of ', len(ImageList))
    torch.save(model.state_dict(), 'Net.model')

