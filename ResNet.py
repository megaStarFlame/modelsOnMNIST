import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.conv3 = None if not use_1x1conv else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage2 = nn.Sequential(
            Residual(64,64),
            Residual(64,64)
        )

        self.stage3 = nn.Sequential(
            Residual(64,128,True,2),
            Residual(128,128)
        )

        self.stage4 = nn.Sequential(
            Residual(128,256,True,2),
            Residual(256,256)
        )

        self.stage5 = nn.Sequential(
            Residual(256,512,True,2),
            Residual(512,512)
        )

        self.stage6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, out_channels)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        return x