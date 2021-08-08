import torch
import os,cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from random import shuffle

#数据集处理模块
class DatasetP(Dataset):
    def __init__(self, filename, transform=None):
        self.images=[]
        self.labels=[]
        self.transform=transform
        l=-1
        self.dat=[]
        for root,dirs,files in os.walk(filename):
            for filename in (x for x in files if x.endswith(".jpg")):
                filepath=os.path.join(root,filename)
                


                image=cv2.imread(filepath,1)
                
                name=filepath.split("\\")[-1]
                self.labels.append(l)
               
                data=image
                #data=LBP(data)
                #data=huiduhua(data)
                data = cv2.resize(data, (224, 224))
                
                self.images.append(data)
            l+=1
        for i,j in  zip(self.images,self.labels):
            self.dat.append([i,j])
    
        self.sshuffle()
    def __getitem__(self, index):
        
        img = self.images[index]
        img = img.transpose((2,0,1))
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([self.labels[index]])
        return img/255, label, index
    def __len__(self):
        return len(self.images)
    #对数据集进行打乱
    def sshuffle(self):
        shuffle(self.dat)
        self.dat=np.array(self.dat)
        self.images.clear()
        self.labels.clear()
        for i,j in zip(self.dat[:,0],self.dat[:,1]):
            self.images.append(i)
            self.labels.append(j)
    def get_labels(self):
        return np.array(self.labels)
    def get_images(self):
        return np.array(self.images)
            
        
        
    

def get_valid(filename):
    images=[]
    labels=[]
    l=-1
    
    for root,dirs,files in os.walk(filename):
        
        for filename in (x for x in files if x.endswith(".jpg")):
            filepath=os.path.join(root,filename)
            


            image=cv2.imread(filepath,0)
            
            name=filepath.split("\\")[-1]
            labels.append(l)
           
            data=image
            #data=LBP(data)
            #data=huiduhua(data)
            data = cv2.resize(data, (48, 48))
            
            images.append(data)
                
        l+=1
    #print(labels)
    data=[]
    #images=tranpca(images)
    for i,j in  zip(images,labels):
        data.append([i,j])
    print(len(data))
    data=np.array(data)
    images=list(data[0:len(data),0])
    labels=data[:,1]
    # 进行数据转换,并归一化
    # (60000,28,28) -> (60000, 28, 28, 1)
    images=np.array(images).reshape(-1, 48, 48, 1)/255.0
    return [images,labels]

def get_image(input_path):          
    images=[]
    labels=[]
    l=-1
    
    for root,dirs,files in os.walk(input_path):
        
        for filename in (x for x in files if (x.endswith(".png") or x.endswith(".jpg"))):
            filepath=os.path.join(root,filename)
            


            image=cv2.imread(filepath,0)
            name=filepath.split("\\")[-1]
            labels.append(l)
           
            data=image
            data = cv2.resize(data, (48, 48))
            #data=data_tf(data)
            images.append(data)
                
           
        l+=1
    #print(labels)
    data=[]
    #images=tranpca(images)
    for i,j in  zip(images,labels):
        data.append([i,j])
    return data
