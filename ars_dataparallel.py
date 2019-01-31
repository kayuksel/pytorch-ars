import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from copy import deepcopy

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"

batch_size = 2048
epochs = 100
lr_rate = 1e-4

import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class BigNetwork(nn.Module):
    def __init__(self):
        super(BigNetwork, self).__init__()
        self.net1 = Network()
        self.net2 = Network()
        self.net3 = Network()
        self.net4 = Network()
        self.net5 = Network()
        self.net6 = Network()
        self.net7 = Network()
        self.net8 = Network()

    def forward(self, x):
        return self.net1(x), self.net2(x), self.net3(x), self.net4(x), self.net5(x), self.net6(x), self.net7(x), self.net8(x)

kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

prev = np.ones(len(train_loader)) * 10.0

def calculate_loss(model, features, t):
    a1, a2, a3, a4, a5, a6, a7, a8 = model(features)
    return F.nll_loss(a1, t), F.nll_loss(a2, t), F.nll_loss(a3, t), F.nll_loss(a4, t), F.nll_loss(a5, t), F.nll_loss(a6, t), F.nll_loss(a7, t), F.nll_loss(a8, t)

import pdb

def train(model):
    global prev
    global train_loader

    noise_model = nn.DataParallel(BigNetwork()).cuda().half()

    with torch.no_grad():

        tsum_loss = 0.0

        model.eval()

        for i, (batch_features, batch_targets) in enumerate(train_loader):

            features = batch_features.cuda(non_blocking=True).half()
            targets = batch_targets.cuda(non_blocking=True)

            for model_param, noise_param in zip(model.parameters(), noise_model.parameters()):
                noise = torch.randn(model_param.size()).cuda().half()

                #noise_norm = noise.norm()
                #if noise_norm != 0.0: noise /= noise_norm
                velo = lr_rate * noise

                noise_param.data.copy_(velo)

            for model_param, noise_param in zip(model.parameters(), noise_model.parameters()):
                model_param.add_(noise_param.data)

            ad1, ad2, ad3, ad4, ad5, ad6, ad7, ad8 = calculate_loss(model, features, targets)


            for model_param, noise_param in zip(model.parameters(), noise_model.parameters()):
                model_param.sub_(noise_param.data * 2.0)

            sb1, sb2, sb3, sb4, sb5, sb6, sb7, sb8 = calculate_loss(model, features, targets)

            x1 = torch.min(ad1, sb1)
            x2 = torch.min(ad2, sb2)
            x3 = torch.min(ad3, sb3)
            x4 = torch.min(ad4, sb4)
            x5 = torch.min(ad5, sb5)
            x6 = torch.min(ad6, sb6)
            x7 = torch.min(ad7, sb7)
            x8 = torch.min(ad8, sb8)

            if x1 < x2 and x1 < x3 and x1 < x4 and x1 < x5 and x1 < x6 and x1 < x7 and x1 < x8:
                for model_param, noise_param in zip(model.parameters(), noise_model.module.net1.parameters()):
                    model_param.add_(noise_param.data * (1.0 + x1))

            if x2 < x1 and x2 < x3 and x2 < x4 and x2 < x5 and x2 < x6 and x2 < x7 and x2 < x8:
                for model_param, noise_param in zip(model.parameters(), noise_model.module.net2.parameters()):
                    model_param.add_(noise_param.data * (1.0 + x2))

            if x3 < x1 and x3 < x2 and x3 < x4 and x3 < x5 and x3 < x6 and x3 < x7 and x3 < x8:
                for model_param, noise_param in zip(model.parameters(), noise_model.module.net3.parameters()):
                    model_param.add_(noise_param.data * (1.0 + x3))

            if x4 < x1 and x4 < x2 and x4 < x3 and x4 < x5 and x4 < x6 and x4 < x7 and x4 < x8:
                for model_param, noise_param in zip(model.parameters(), noise_model.module.net4.parameters()):
                    model_param.add_(noise_param.data * (1.0 + x4))

            if x5 < x1 and x5 < x2 and x5 < x3 and x5 < x4 and x5 < x6 and x5 < x7 and x5 < x8:
                for model_param, noise_param in zip(model.parameters(), noise_model.module.net5.parameters()):
                    model_param.add_(noise_param.data * (1.0 + x5))

            if x6 < x1 and x6 < x2 and x6 < x3 and x6 < x4 and x6 < x5 and x6 < x7 and x6 < x8:
                for model_param, noise_param in zip(model.parameters(), noise_model.module.net6.parameters()):
                    model_param.add_(noise_param.data * (1.0 + x6))

            if x7 < x1 and x7 < x2 and x7 < x3 and x7 < x4 and x7 < x5 and x7 < x6 and x7 < x8:
                for model_param, noise_param in zip(model.parameters(), noise_model.module.net7.parameters()):
                    model_param.add_(noise_param.data * (1.0 + x7))

            if x8 < x1 and x8 < x2 and x8 < x3 and x8 < x4 and x8 < x5 and x8 < x6 and x8 < x7:
                for model_param, noise_param in zip(model.parameters(), noise_model.module.net8.parameters()):
                    model_param.add_(noise_param.data * (1.0 + x8))


            loss = calculate_loss(model, features, targets)[0]

 
            #checkpoint = deepcopy(model)
            prev[i] = loss
            print(np.mean(prev))

            tsum_loss = tsum_loss + loss.item()

        torch.cuda.empty_cache()
        train(model)



if __name__ == '__main__':

    model = nn.DataParallel(BigNetwork()).cuda().half()
    train(model)
