import torch
import torchvision
from torch.utils import data
from torchvision import transforms

import numpy as np

from d2l import torch as d2l
import matplotlib.pyplot as plt

from typing import Tuple
import os

data_path = "train data path"

local_path = "save path"

label_dirt = {
    0:"T-shirt",
    1:"Trouser",
    2:"Pullover",
    3:"Dress",
    4:"Coat",
    5:"Sandal",
    6:"Shirt",
    7:"Sneaker",
    8:"Bag",
    9:"Ankle boot"
}
def get_labels(labels):
    return [label_dirt[int(i)] for i in labels]

def show_images(imgs,row,col,title=None,scale=1.5):
    figsize = (col*scale,row*scale)
    _,axes = plt.subplots(row,col,figsize=figsize)
    axes = axes.flatten()
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.cpu().detach().numpy())
        else:
            ax.imshow(img)
        ax.set_title(title[i])

def load_fashion_mnist(batch_size,resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=trans,download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=trans, download=True)
    return (
        data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=2),
        data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,num_workers=2)
    )

def accuracy(y_hat,y):
    ''' 计算正确的数量,仅限分类模型 '''
    if y_hat.shape[0]>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())

def evaluate_model(model,dataIter,device='cpu'):
    if isinstance(model,torch.nn.Module):
        model.eval()
    acc_sum,n=0.0,0
    with torch.no_grad():
        for X,y in dataIter:
            X,y = X.to(device),y.to(device)
            acc_sum+=accuracy(model(X),y)
            n+=y.shape[0]
    return acc_sum/n


def train(model,trainIter,loss,updater,device='cpu'):
    if isinstance(model,torch.nn.Module):
        model.train()
    acc_num,n=0.0,0
    loss_sum=0.0

    for X,y in trainIter:
        X,y = X.to(device),y.to(device)
        y_pred = model(X)
        l = loss(y_pred,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])

        acc_num+=accuracy(y_pred,y)
        n+=y.shape[0]
        loss_sum+=l.item()

        # print("A batch has been finished")
    return acc_num/n,loss_sum/n

def train_loop(model,trainIter,testIter,loss,updater,num_epochs,device='cpu'):
    for epoch in range(num_epochs):
        train_metrics,loss_sum = train(model,trainIter,loss,updater,device)

        test_metrics = evaluate_model(model,testIter,device)

        # if epoch%10==0:
        print("epoch %d, train_loss %.3f, train_acc %.3f, test_acc %.3f"%(epoch+1,loss_sum,train_metrics,test_metrics))

def train_loop_models_save(model,trainIter,testIter,loss,updater,num_epochs,device='cpu',name="model",path=local_path):
        for epoch in range(num_epochs):
            train_metrics,loss_sum = train(model,trainIter,loss,updater,device)

            test_metrics = evaluate_model(model,testIter,device)

            if not os.path.exists(os.path.join(path,name+".csv")):
                with open(os.path.join(path,name+".csv"),"a+") as f:
                    f.write("epoch,train_loss,train_acc,test_acc\n")
            
            with open(os.path.join(path,name+".csv"),"a+") as f:
                f.write("%d,%.5f,%.4f,%.4f\n"%(epoch+1,loss_sum,train_metrics,test_metrics))

            print("%s: epoch %d, train_loss %.3f, train_acc %.3f, test_acc %.3f"%(name,epoch+1,loss_sum,train_metrics,test_metrics))
            

def predict(model,test_iter):
    for X,y in test_iter:
        trues = get_labels(y)
        preds = get_labels(model(X).argmax(axis=1))
        titles = [true+'\n'+pred for true,pred in zip(trues,preds)]
        show_images(X.reshape((X.shape[0],28,28)),1,X.shape[0],titles)
        break


class DataSet(data.Dataset):
    def __init__(self,feature,label,transform=None):
        self.data=feature.type(torch.float32)
        self.labels=label
        self.transform=transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        feature=self.data[index]
        target=self.labels[index]
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, target

def get_data_iter(features,label,batch_size,shuffle=True):
    dataset = DataSet(features,label)
    
    return data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=2)


def load_mnist(batch_size,resize=None)->Tuple[data.DataLoader,data.DataLoader]:
    path = data_path
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(root=path, train=True, transform=trans,download=True)
    mnist_test = torchvision.datasets.MNIST(root=path, train=False, transform=trans, download=True)
    return (
        data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=2),
        data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,num_workers=2)
    )