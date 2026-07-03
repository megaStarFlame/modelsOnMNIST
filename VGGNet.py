import torch
from torch import nn
from torch.nn import functional as F

def vgg_block(num_convs,in_channels,out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self,in_channels=3,num_classes=10):
        super(VGG,self).__init__()

        self.feature = nn.Sequential(
            vgg_block(1,in_channels,64),
            vgg_block(1,64,128),
            vgg_block(2,128,256),
            vgg_block(2,256,512),
            vgg_block(2,512,512),

            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
        )

    def forward(self,x):
        x = self.feature(x)
        x = self.classifier(x)
        return x
    

def vgg_block_modify(num_convs,in_channels,out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

class VGG_modify(nn.Module):
    def __init__(self,in_channels=3,num_classes=10):
        super(VGG_modify,self).__init__()

        self.feature = nn.Sequential(
            vgg_block_modify(1,in_channels,64),
            vgg_block_modify(1,64,128),
            vgg_block_modify(2,128,128), # 原：128,256
            vgg_block_modify(2,128,64), # 原：256,512
            vgg_block_modify(2,64,32),  # 原：512,512

            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(32*7*7,512),  # 原：512*7*7，4096
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,128),    # 原：4096，4096
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,num_classes)  # 原：4096，1000
        )

    def forward(self,x):
        x = self.feature(x)
        x = self.classifier(x)
        return x