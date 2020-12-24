import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import models
import cv2
import numpy as np
import time
import torch.optim as optim
import os
from model import LungINFseg
from torch.autograd import Variable


classes = 1
model  = LungINFseg(classes=1)
model.cuda()
model.load_state_dict(torch.load('Net.model'))


MainPath='/LungINFseg/test/'
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


TestPath='/LungINFseg/test/'
ImageList=[]

for file in os.listdir(TestPath+'/'+image_pth):
    if file.endswith('png'):
        ImageList.append(file)
        main_part=file.split('.png')[0]
        main_part=main_part+'.png'


for a in (ImageList):
    model.eval()
    print(a)
    im = cv2.imread(TestPath + '/' + image_pth + '/' + a)
    h,w,c = im.shape
    # shape = im.shape
    im = cv2.resize(im, (256,256))   
    im2 = im.copy()
    im3 = im.copy()
    im = im / 255.0
    im = np.expand_dims(im, 0)
    im = np.transpose(im, (0, 3, 1, 2))
    im = Variable(torch.FloatTensor(im)).cuda()

    im[:, 0, :, :] = im[:, 0, :, :] - 0.485
    im[:, 1, :, :] = im[:, 1, :, :] - 0.456
    im[:, 2, :, :] = im[:, 2, :, :] - 0.406
    # *****************************************
    im[:, 0, :, :] = im[:, 0, :, :] / 0.229
    im[:, 1, :, :] = im[:, 1, :, :] / 0.224
    im[:, 2, :, :] = im[:, 2, :, :] / 0.225
    e1,e2,e3,e4,out = model(im)
    #_, out = torch.max(out, 1) # in case of cross entropy loss
    out = out.type(torch.float) # in case of binary cross entropy loss
    out = torch.sigmoid(out) # in case of binary cross entropy loss
    out = out.cpu().data.numpy()
    out = out>0.5
    out = np.squeeze(out)
    loc2 = np.where(out == 0)
    out[loc2] = 0
    out=np.array(out,dtype=np.uint8)
    #im2[loc2]=0
    out = cv2.resize(out,(w,h))
    cv2.imwrite('out/'+a, 255*out)

