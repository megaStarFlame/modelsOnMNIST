import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

class Inception(nn.Module):
    def __init__(self,in_channels,c1:int,c2:Tuple[int,int],c3:Tuple[int,int],c4:int,**kwargs):
        super(Inception,self).__init__(**kwargs)

        self.p1=nn.Conv2d(in_channels,c1,kernel_size=1)

        self.p2_1=nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2=nn.Conv2d(c2[0],c2[1],kernel_size=3,padding='same')

        self.p3_1=nn.Conv2d(in_channels,c3[0],kernel_size=1)
        self.p3_2=nn.Conv2d(c3[0],c3[1],kernel_size=5,padding='same')

        self.p4_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2=nn.Conv2d(in_channels,c4,kernel_size=1)

    def forward(self,x):
        p1=F.relu(self.p1(x))
        p2=F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3=F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4=F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1,p2,p3,p4),dim=1)

class GoogLeNet(nn.Module):
    def __init__(self,in_channels=3,num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0),
            nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.stage3 = nn.Sequential(
            Inception(192,64,(96,128),(16,32),32),
            Inception(256,128,(128,192),(32,96),64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.stage4 = nn.Sequential(
            Inception(480,192,(96,208),(16,48),64),
            Inception(512,160,(112,224),(24,64),64),
            Inception(512,128,(128,256),(24,64),64),
            Inception(512,112,(144,288),(32,64),64),
            Inception(528,256,(160,320),(32,128),128),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.stage5 = nn.Sequential(
            Inception(832,256,(160,320),(32,128),128),
            Inception(832,384,(192,384),(48,128),128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024,num_classes)
        )

    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        # 注意交叉熵会自动求softmax，所以这里不用求softmax
        return x
    

class Inception_modify(nn.Module):
    def __init__(self,in_channels,c1:int,c2:Tuple[int,int],c3:Tuple[int,int],c4:int,**kwargs):
        super(Inception_modify,self).__init__(**kwargs)

        self.p1=nn.Sequential(
            nn.Conv2d(in_channels,c1,kernel_size=1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )

        self.p2_1=nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2=nn.Conv2d(c2[0],c2[1],kernel_size=3,padding='same')

        self.p2 = nn.Sequential(
            self.p2_1,
            
            nn.BatchNorm2d(c2[0]),
            nn.ReLU(),

            self.p2_2,

            nn.BatchNorm2d(c2[1]),
            nn.ReLU()
        )


        self.p3_1=nn.Conv2d(in_channels,c3[0],kernel_size=1)
        self.p3_2=nn.Conv2d(c3[0],c3[1],kernel_size=5,padding='same')

        self.p3=nn.Sequential(
            self.p3_1,

            nn.BatchNorm2d(c3[0]),
            nn.ReLU(),

            self.p3_2,

            nn.BatchNorm2d(c3[1]),
            nn.ReLU()
        )

        self.p4_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2=nn.Conv2d(in_channels,c4,kernel_size=1)

        self.p4 = nn.Sequential(
            self.p4_1,
            self.p4_2,

            nn.BatchNorm2d(c4),

            nn.ReLU()
        )


    def forward(self,x):
        p1=self.p1(x)
        p2=self.p2(x)
        p3=self.p3(x)
        p4=self.p4(x)
        return torch.cat((p1,p2,p3,p4),dim=1)

class GoogLeNet_modify(nn.Module):
    def __init__(self,in_channels=3,num_classes=1000):
        super(GoogLeNet_modify, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3),
            
            nn.BatchNorm2d(64),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0),
            nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1),
            
            nn.BatchNorm2d(192),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.stage3 = nn.Sequential(
            Inception_modify(192,64,(96,128),(16,32),32),
            Inception_modify(256,128,(128,192),(32,96),64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.stage4 = nn.Sequential(
            Inception_modify(480,192,(96,208),(16,48),64),
            Inception_modify(512,160,(112,224),(24,64),64),
            Inception_modify(512,128,(128,256),(24,64),64),
            Inception_modify(512,112,(144,288),(32,64),64),
            Inception_modify(528,256,(160,320),(32,128),128),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.stage5 = nn.Sequential(
            Inception_modify(832,256,(160,320),(32,128),128),
            Inception_modify(832,384,(192,384),(48,128),128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024,num_classes)
        )

    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        # 注意交叉熵会自动求softmax，所以这里不用求softmax
        return x