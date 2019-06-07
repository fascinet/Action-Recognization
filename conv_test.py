from __future__ import print_function, division
import pickle
import torch
import torch.nn as nn
import glob
from PIL import Image
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torchvision import *
#from code5 import dataloader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import skimage
import re
import time
import os
import copy

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.spat_feature = spat_model #feature_size =   Nx512x7x7
        self.temp_feature = temp_model #feature_size = Nx512x7x7
        self.layer1       = nn.Sequential(nn.Conv1d(202,1024,1),nn.ReLU())
        self.fc           = nn.Sequential(nn.LSTMCell(1024,1024))
    def forward(self,spat_data,temp_data):
        x1       = self.spat_feature(spat_data)
        x2       = self.temp_feature(temp_data)
        
        y        = torch.cat((x1,x2), dim= 1)
        #print(x1.shape,x2.shape,y.shape)
        for i in range(x1.size(1)):
            y[:,(2*i)]   = x1[:,i]
            y[:,(2*i+1)] = x2[:,i]
        print(y.shape)
        b=y.size(0)   
        y        = y.view(1,202,b)
        cnn_out  = self.layer1(y)
        print(cnn_out.shape)
        cnn_out  = cnn_out.view(b,1024)
        lis=[1]*b
        cnn_out=torch.split(cnn_out,lis,0)
        p = torch.empty(1, 1024, dtype=torch.float)
        for i in range(b):
            out= self.fc(cnn_out[i])
            #out=torch.tensor()
            if i==0:
                p=out[0]
            else:
                p=torch.cat((p,out[0]),dim=0)
        #print(p)
        p=torch.sum(p,1)
        p=torch.div(p,202)
        print(p.shape)
        return p

class Test_Convertor():
    def __init__(self):
        self.train_transform = transforms.Compose([transforms.RandomCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.val_transform = transforms.Compose([transforms.Resize([224,224]),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        #self.val_transform = val_transform
    def stackopf(self, video):
        print('stackopf')
        self.video = video
        s='Opticalflow/Recipe/Recipe'
        flow = torch.FloatTensor(3,100,224,224)
        for idx in range(1,100):
            v_image =self.video+'/'+s+str(idx)+'.jpg'
            print(v_image)
            imgV=(Image.open(v_image))
            V=self.val_transform(imgV)
            flow[:,idx,:,:] = V
            imgV.close()  
        return flow


    #loading image from path and frame number
    def load_ucf_image(self,video_name, index, mode):
        print('In load_ucf')
        print(video_name,index)
        path = video_name+'/'
        s='rgb/Recipe/Recipe'+str(index)
        path=path+s+'.jpg'
        print(path)
        img = Image.open(path)
        if(mode == 'train'):
            transformed_img = self.train_transform(img)
        else:
            transformed_img = self.val_transform(img)
        img.close()
        return transformed_img
#spatial RGB and Temporal Optical FLow
    def Converting(self):
        for i in range(10,101,10):
            video_name="Images"
            temp = self.load_ucf_image(video_name, i,mode='val')
            #temp = temp.view([1,3,224,224])
            if(i == 10):
                spatial_data = temp
            else:
#                     print("data shape ", data.shape)
                spatial_data = torch.cat((spatial_data,temp))

        temp_data = self.stackopf(video=video_name)
        sample = (spatial_data, temp_data)
        return sample

def train_model(model,sample):
    #since = time.time()
    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_acc = 0.0
    print("loading the data")
    spat_data, temp_data = sample
    #spat_data=spat_data.view([3,128,224,224])
    spat_data=torch.unsqueeze(spat_data,dim=0)
    temp_data=torch.split(temp_data,[1,1,1],dim=0)[0]
    spat_data = Variable(spat_data)
    temp_data = Variable(temp_data)
    #labels    =  Variable(labels)
    print(spat_data.shape)
    print(temp_data.shape)
    outputs = model(spat_data, temp_data)
    # backward and optimize for training m
    return outputs
p=Test_Convertor()
sample=p.Converting()
print(sample[1].shape)
print(sample[0].shape)
model=torch.load('extra100',map_location='cpu')
print(train_model(model,sample))
