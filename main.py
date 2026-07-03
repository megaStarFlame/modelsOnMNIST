import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import AlexNet,VGGNet,GoogLeNet,ResNet,MobileNet

import random

import common_methods as cm

class print_shape(nn.Module):
    def __init__(self):
        super(print_shape, self).__init__()
    
    def forward(self, x):
        print("input shape: ",x.shape)
        return x

'''
alexnet: SGD
vgg: SGD,
googlenet: Adam,
resnet: Adam,
mobilenet: Adam
'''

loop_train = [
    ('AlexNet',128,AlexNet.AlexNet_modify,torch.optim.SGD),
    ('VGG',64,VGGNet.VGG_modify,torch.optim.SGD),
    ('GoogLeNet',128,GoogLeNet.GoogLeNet_modify,torch.optim.Adam),
    ('ResNet',128,ResNet.ResNet,torch.optim.Adam),
    ('MobileNet',128,MobileNet.MobileNet,torch.optim.Adam)
]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

num_epochs = 10

def main():
    for name,batch_size,model_class,optimizer in loop_train:
        trainIter,testIter = cm.load_mnist(batch_size=batch_size,resize=(224,224))

        model = model_class(1,10).to(device)

        loss = nn.CrossEntropyLoss().to(device)
        updater = optimizer(model.parameters(),lr=0.01)

        print(f'{name}开始训练')
        cm.train_loop_models_save(model,trainIter,testIter,loss,updater,num_epochs,device,name)
        print(f'{name}训练完成')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    setup_seed(33550336)
    main()