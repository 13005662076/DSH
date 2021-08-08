import torch.nn as nn
from torchvision import models
import torchvision.datasets as datasets
from torch.autograd import Variable

#创建图像特征提取模块
#使用vgg16预训练模型
class model(nn.Module):
    def __init__(self,m,model_name,bit):
        super(model,self).__init__()
        self.features=m.features
        fc1=nn.Linear(25088,4096)
        fc2=nn.Linear(4096,4096)
        fc3=nn.Linear(4096,bit)
        self.classifier = nn.Sequential(
                fc1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                fc2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                fc3
            )
    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x
        
        
