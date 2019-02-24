import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from copy import deepcopy

import os, pdb
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"

batch_size = 2048
epochs = 100
lr_rate = 1e-2

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
    def __init__(self, k = 16):
        super(BigNetwork, self).__init__()

        self.networks = []
        for i in range(k):
            self.networks.append(nn.DataParallel(Network()).cuda())

    def forward(self, x):
        return [self.networks[i](x) for i in range(len(self.networks))]

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
    outputs = model(features)
    return torch.Tensor([F.nll_loss(output, t) for output in outputs])

checkpoint = None

def train(model):
    global prev
    global train_loader

    noise_model = nn.DataParallel(BigNetwork()).cuda()

    with torch.no_grad():

        tsum_loss = 0.0

        model.eval()

        for i, (batch_features, batch_targets) in enumerate(train_loader):

            features = batch_features.cuda(non_blocking=True)
            targets = batch_targets.cuda(non_blocking=True)

            for i in range(16):
                for noise_param in noise_model.module.networks[i].parameters():
                    noise = torch.randn(noise_param.size())
                    velo = lr_rate * noise.cuda()
                    noise_param.data.copy_(velo)

            #checkpoint = deepcopy(model)

            for i in range(16):
                for model_param, noise_param in zip(model.module.networks[i].parameters(), noise_model.module.networks[i].parameters()):
                    model_param.add_(noise_param.data)

            add_losses = calculate_loss(model, features, targets)

            for i in range(16):
                for model_param, noise_param in zip(model.module.networks[i].parameters(), noise_model.module.networks[i].parameters()):
                    model_param.sub_(noise_param.data * 2.0)

            sub_losses = calculate_loss(model, features, targets)

            std = 1#torch.std(torch.cat([add_losses, sub_losses], dim=0))

            min_losses = torch.min(add_losses, sub_losses)

            dif_losses = sub_losses - add_losses

            #model.load_state_dict(checkpoint.state_dict())

            for i in range(16):
                for model_param, noise_param in zip(model.module.networks[i].parameters(), noise_model.module.networks[i].parameters()):
                    model_param.add_(noise_param.data)

            top = torch.topk(min_losses, 4, largest=False)[1]

            for i in range(16):
                for k in top:
                    for model_param, noise_param in zip(model.module.networks[i].parameters(), noise_model.module.networks[k].parameters()):
                        model_param.add_(noise_param.data * dif_losses[k] / std)

            loss = calculate_loss(model, features, targets)[0]

            prev[i] = loss
            print(np.mean(prev))

            tsum_loss = tsum_loss + loss.item()

        torch.cuda.empty_cache()
        train(model)



if __name__ == '__main__':

    model = nn.DataParallel(BigNetwork()).cuda()
    train(model)
