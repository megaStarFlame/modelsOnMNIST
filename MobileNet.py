import torch
from torch import nn
from torch.nn import functional as F

def mobile_block(in_channels,out_channels,stride=1):
    return nn.Sequential(
        # print_shape(),
        nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=stride,padding=1,groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        # print_shape(),
        nn.Conv2d(in_channels,out_channels,kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class MobileNet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(MobileNet,self).__init__()
        self.stage1 = nn.Sequential(
            # print_shape(),
            nn.Conv2d(in_channels,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.stage2 = nn.Sequential(
            mobile_block(32,64,1),
            mobile_block(64,128,2),

            mobile_block(128,128,1),
            mobile_block(128,256,2),

            mobile_block(256,256,1),
            mobile_block(256,512,2)
        )

        self.stage3 = nn.Sequential(
            mobile_block(512,512,1),
            mobile_block(512,512,1),
            mobile_block(512,512,1),
            mobile_block(512,512,1),
            mobile_block(512,512,1)
        )

        self.stage4 = nn.Sequential(
            mobile_block(512,1024,2),
            mobile_block(1024,1024,1),

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024,out_channels)
        )

    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x