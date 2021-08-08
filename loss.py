import torch
import torch.nn as nn
from torch.autograd import Variable

class dshloss(nn.Module):
    def __init__(self,num_train,bit=8,numclasses=8):
        super(dpshloss,self).__init__()
        #定义输出[-1,1]矩阵
        self.B=torch.zeros(num_train,numclasses)
        #定义输出矩阵
        self.train_labels_onehot=torch.zeros(num_train,bit)

    def forward(self,output,batch_index,train_label_onehot):
        
        for i,ind in enumerate(batch_index):
            self.train_labels_onehot[ind,:]=train_label_onehot[i]
            self.B[ind,:]=torch.sign(output.data[i])

        b=torch.sign(output)
        dist=(b.unsqueeze(1)-self.B.unsqueeze(0)).pow(2).sum(dim=2)
        
        label_onehot=(train_label_onehot.mm(self.train_labels_onehot.t())==0).float()
        
        loss=0.5*(1-label_onehot)*dist+0.5*(label_onehot)*(16-dist).clamp(min=0)
        
        loss1=loss.mean()
        
        loss2=0.1*(self.B-1).abs().mean()
        return loss1,loss2
