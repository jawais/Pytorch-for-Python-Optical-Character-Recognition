import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


import tarfile
tar = tarfile.open('/EnglishFnt.tgz')
tar.extractall('./EnglishFnt')
tar.close()

data= torchvision.datasets.ImageFolder(
    root = '/content/EnglishFnt/English/Fnt',
    transform = transforms.Compose([transforms.Resize((48,48)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5))])

def load_split(dataset,batch_size,test_split=0.3):
    shuffle_dataset = True
    random_seed=42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split*dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    val_indices, test_indices = indices[split:],indices[:split]
    
    # Creating data samplers and loaders:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    
    train_loader= torch.utils.data.DataLoader(dataset, batch_size,
                                           sample=train_sampler)
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size,
                                                sampler=test_sampler)
                                                
    val_loader = torch.utils.data.DataLoader(dataset, batch_size,
                                                sampler=val_sampler)
    
    return train_loader,test_loader,val_loader

batch_size = 36
train_loader, test_loader, val_loader, load_split(data,batch_size,test_split=0.3)

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*9*9, 62)
        
        self.max_pool = nn.MaxPool2d(2, 2,ceil_mode=True)
        self.dropout = nn.Dropout(0.2)
        
        self.conv_bn1 = nn.BatchNorm2d(48,3)
        self.conv_bn2 = nn.BatchNorm2d(16)
        self.conv_bn3 = nn.BatchNorm2d(32)
        self.conv_bn4 = nn.BatchNorm2d(64)
        
    def forward(self, x):
    
        x  = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.conv_bn3(x)
        
        x = F.relu(self.conv2(x))
        #x = self.max_pool(x)
        x = self.conv_bn4(x)
        
        x = x.view(-1, 64*9*9)
        
        x = self.dropout(x)
        x = self.fc1(x)
        return x

class MyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, lables):
        y = one_hot(lables,len(pred[0]))
        y = y.cuda()
        ctx.save_for_backward(y, pred)
        loss = -y * torch.log(pred)
        loss = loss.sum()/len(lables)
        return loss
        
    @staticmethod 
    def backward(ctx, grad_output):
        y, pred = ctx.saved_tensors
        gradinput= (- y/ pred) - y
        grad_input = grad_input/len(pred)
        return grad_input, grad_output

 class MyCEL(torch.nn.Module):
    
    def __init__(self):
        super(MyCEL, self).__init__()
        
    def forward(self, pred, lables):
        y = to_once_hot(lables, len(pred[0]))
        y = y.cuda()
        loss = - y * torch.log(pred)
        loss = loss.sum()/len(lables)
        return loss

network = Network()
use_cuda = True
if  use_cuda and torch.cuda.is_available():
    network.cuda()
    print('cuda')
    
optimizer = optim.SGD(network.parameters(), Lr=0.01, momentum=0.9)

epoch = 0
max_epoch = 5
end = False
myloss = MyCEL()
while epoch <max_epoch and not end:
    epoch +=1
    total_loss = 0
    total_correct = 0
    total_val = 0
    total_train = 0
    for data in (train_loader):
        
        images, lables = data
        if use_cuda and torch.coda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        pred = network(images)
        pred = F.softmax(pred)
        loss= myloss(pred,labels)
        total_loss += loss.item()
        total_train += len(pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_correct += pred.argmax(dim-1).eq(labels).sum()
        
    print("epoch : ",epoch, "Training Accuracy : " , total_correct*1.0/total_train, "Train loss : " , total_loss*1.0/len(train_loss))
    
    if total_correct*1.0/total_train>= 0.98:
        end = True
    total_loss = 0
    val_total_correct = 0
    for batch in (val_loader):
        images, labels = batch
        if use_cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        pred = network(images)
        loss = F.cross_entropy(pred,labels)
        total_loss += loss.item()
        total_val+= len(pred)
        val_total_correct +=pred.argmax(dim = 1).eq(labels).sum()
        print("epoch : ", epoch, "Val Accuracy : ", val_total_correct*1.0/total_val, "Val Loss : " , total_loss*1.0/len(val_total_correct))
    torch.cuda.empty_cache()

test_total_correct = 0
total_test = 0
x=0

for batch in (test_loader) :
    images, labels= batch
    if use_cuda and torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
    pred = network(images)
    total_test+=len(pred)
    x += 1
    test_total_correct+= pred.argmax(dim = 1).eq(labels).sum()
print("Test Accuracy : ", test_total_correct*1.0/total_test,)


PATH = "entire_model.pt"

torch.save(network, PATH)