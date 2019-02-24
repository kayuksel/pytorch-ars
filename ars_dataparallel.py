from itertools import count

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

import os, pdb
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"

lr_rate = 1e-1
no_dir = 32
top_dir = 2


POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5

def get_batch(batch_size=2048):
    random = torch.randn(batch_size).unsqueeze(1)
    x = torch.cat([random ** i for i in range(1, POLY_DEGREE+1)], 1)
    y = x.mm(W_target) + b_target.item()
    return x.cuda(), y.cuda()

class BigNetwork(nn.Module):
    def __init__(self, k = no_dir):
        super(BigNetwork, self).__init__()

        self.networks = []
        for i in range(k):
            self.networks.append(nn.DataParallel(nn.Linear(W_target.size(0), 1)).cuda())

    def forward(self, x):
        return [self.networks[i](x) for i in range(len(self.networks))]


def calculate_loss(model, features, t):
    outputs = model(features)
    return torch.Tensor([F.smooth_l1_loss(output, t) for output in outputs])

checkpoint = None

def train(model):

    noise_model = nn.DataParallel(BigNetwork()).cuda()

    with torch.no_grad():

        tsum_loss = 0.0

        model.eval()

        for batch_idx in count(1):

            features, targets = get_batch()

            for i in range(no_dir):
                for noise_param in noise_model.module.networks[i].parameters():
                    noise = torch.randn(noise_param.size())
                    velo = lr_rate * noise.cuda()
                    noise_param.data.copy_(velo)

            #checkpoint = deepcopy(model)

            for i in range(no_dir):
                for model_param, noise_param in zip(model.module.networks[i].parameters(), noise_model.module.networks[i].parameters()):
                    model_param.add_(noise_param.data)

            add_losses = calculate_loss(model, features, targets)

            for i in range(no_dir):
                for model_param, noise_param in zip(model.module.networks[i].parameters(), noise_model.module.networks[i].parameters()):
                    model_param.sub_(noise_param.data * 2.0)

            sub_losses = calculate_loss(model, features, targets)

            std = torch.std(torch.cat([add_losses, sub_losses], dim=0))

            min_losses = torch.min(add_losses, sub_losses)

            dif_losses = sub_losses - add_losses

            #model.load_state_dict(checkpoint.state_dict())

            for i in range(no_dir):
                for model_param, noise_param in zip(model.module.networks[i].parameters(), noise_model.module.networks[i].parameters()):
                    model_param.add_(noise_param.data)

            top = torch.topk(min_losses, top_dir, largest=False)[1]

            for i in range(no_dir):
                for k in top:
                    for model_param, noise_param in zip(model.module.networks[i].parameters(), noise_model.module.networks[k].parameters()):
                        model_param.add_(noise_param.data * dif_losses[k] / std)

            loss = calculate_loss(model, features, targets)[0]

            print(loss)

            tsum_loss = tsum_loss + loss.item()

        torch.cuda.empty_cache()
        train(model)



if __name__ == '__main__':

    model = nn.DataParallel(BigNetwork()).cuda()
    train(model)
