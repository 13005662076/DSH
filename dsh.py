from torchvision import transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
import os
import numpy as np
from datetime import datetime
from DataDeal import *
from model import *
from calMap import *
from loss import *

#将图像所属的标签转化为01矩阵
def EncodingOnehot(target,nclasses):
    target=target.long()
    target_onehot=torch.FloatTensor(target.size(0),nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1,target.view(-1,1),1)
    return target_onehot

#计算每张图像与全部图像是否相似
def calS(batch_label_onehot,train_label_onehot):
    S=(batch_label_onehot.mm(train_label_onehot.t())).type(torch.FloatTensor)
    return S

#创建图像特征提取模型
def createModel(model_name="vgg16",bit=8):
    vgg16 = models.vgg16(pretrained=True)
    return model(vgg16, model_name, bit)
    

#计算log()
def Log(x):
    if torch.cuda.is_available():
        #lt=torch.log(1+torch.exp(-torch.abs(x)))+torch.max(x,Variable(torch.FloatTensor([0.]).cuda()))
        lt=torch.log(1+torch.exp(x))
    else:
        #lt=torch.log(1+torch.exp(-torch.abs(x)))+torch.max(x,Variable(torch.FloatTensor([0.])))
        lt=torch.log(1+torch.exp(x))
    
    return lt



def Dpsh(bit,gpu_ind=0):
    '''
    tf=transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    '''
    #train_set,train_labels=get_valid("E:/testdl/train")

    #获得训练数据
    train_set=DatasetP("E:/testdl/train")
    train_labels=train_set.get_labels().astype(float)
    num_train=train_set.__len__()
    train_loader=DataLoader(train_set,batch_size=16,shuffle=False,num_workers=0)

    #获得验证数据
    test_set=DatasetP("E:/testdl/test")
    test_labels=test_set.get_labels().astype(float)
    num_test=test_set.__len__()
    test_loader=DataLoader(test_set,batch_size=8,shuffle=False,num_workers=0)
    
    model=createModel()
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    '''
    #定义输出[-1,1]矩阵
    B=torch.zeros(num_train,bit)
    #定义输出矩阵
    U=torch.zeros(num_train,bit)
    '''
    nclasses=8
    train_labels_onehot=EncodingOnehot(torch.from_numpy(train_labels),nclasses)
    

    test_labels_onehot=EncodingOnehot(torch.from_numpy(test_labels),nclasses)

    #获得全部图像的相互相似矩阵
    St=calS(train_labels_onehot,train_labels_onehot)
    
    criterion = dpshloss(num_train,bit)
    for epoch in range(150):
        epoch_loss=0.0
        for im,label,batch_index in train_loader:
            
            im=im.type('torch.FloatTensor')
            if torch.cuda.is_available():
                train_label_onehot=EncodingOnehot(label,nclasses)
                im,label=Variable(im.cuda()),Variable(label.cuda())
                
            else:
                train_label_onehot=EncodingOnehot(label,nclasses)
                im,label=Variable(im),Variable(label)
                
                
            #model.zero_grad()
            output=model(im)
            
            loss1,loss2=criterion(output,batch_index,train_label_onehot)
            loss=loss1+loss2 #50是一个超参数，需要根据模型的精确度来确定
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()

        
        #valid_test  使用验证集来测试模型
        B=criterion.B
        Bv=np.zeros([num_test,bit],dtype=np.float32)
        for im,label,batch_index in test_loader:
            im=im.type("torch.FloatTensor")
            if torch.cuda.is_available():
                im=Variable(im.cuda())
            else:
                im=Variable(im)
            output=model(im)
            for i,ind in enumerate(batch_index):
                Bv[ind,:]=torch.sign(output.data[i])
        Bt=torch.sign(B).numpy()
        map=CalcMap(Bv,Bt,test_labels_onehot.numpy(),train_labels_onehot.numpy())
        
        
        
        print("Epoch %d. loss:%f map:%f"%(epoch+1,epoch_loss/num_train,map))
            
Dpsh(8)


#http://t.zoukankan.com/youmuchen-p-13547393.html
