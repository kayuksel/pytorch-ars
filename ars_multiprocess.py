import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from copy import deepcopy
from threading import Lock

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"

lock = Lock()

batch_size = 64
epochs = 100
lr_rate = 1e-3

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

def calculate_loss(model, features, targets):
    output = model(features)
    return F.nll_loss(output, targets)

def train(checkpoint, index):
    global prev
    global train_loader

    dd = (index % 3) + 1

    model = Network().cuda(dd).half()
    model.load_state_dict(checkpoint.state_dict())

    noise_model = Network().cuda(dd).half()

    with torch.no_grad():

        selected_batches = np.random.multinomial(int(len(train_loader)*0.5), prev/np.sum(prev))

        tsum_loss = 0.0

        model.eval()

        for i, (batch_features, batch_targets) in enumerate(train_loader):

            if (selected_batches[i] == 0): continue

            features = batch_features.cuda(dd, non_blocking=True).half()
            targets = batch_targets.cuda(dd, non_blocking=True)

            for model_param, noise_param in zip(model.parameters(), noise_model.parameters()):
                
                noise = torch.randn(model_param.size()).cuda(dd).half()

                #noise_norm = noise.norm()
                #if noise_norm != 0.0: noise /= noise_norm
                velo = lr_rate * noise

                noise_param.data.copy_(velo)

            for model_param, noise_param in zip(model.parameters(), noise_model.parameters()):
                model_param.add_(noise_param.data)

            add_loss = calculate_loss(model, features, targets)


            for model_param, noise_param in zip(model.parameters(), noise_model.parameters()):
                model_param.sub_(noise_param.data * 2.0)
            
            sub_loss = calculate_loss(model, features, targets)

            for model_param, noise_param in zip(model.parameters(), noise_model.parameters()):
                model_param.add_(noise_param.data * (1.0 + sub_loss - add_loss))

            loss = calculate_loss(model, features, targets)

            lock.acquire()
            if prev[i] > loss:
                checkpoint = deepcopy(model)
                prev[i] = loss
                print(np.mean(prev))
            lock.release()

            tsum_loss = tsum_loss + loss.item()

        torch.cuda.empty_cache()
        train(checkpoint, index)



if __name__ == '__main__':

    checkpoint = Network().half().share_memory()
    torch.backends.cudnn.benchmark = True
    processes = []
    for i in range(8):
        p = mp.Process(target=train, args=(checkpoint, i))
        p.start()
        processes.append(p)
    for p in processes: p.join()

