# You need to install the following python packages
# pytorch, vit_pytorch.
import torch
import torchvision
import torchvision.transforms as transforms
from vit_pytorch import ViT, ViT_equ, ViT_Haar
import time
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import wandb
import os

torch.set_printoptions(profile='full', precision=6)

def seed_everything(seed=666):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def softmax2(x, index, dim=0):
    max_x = x.max()

    x1 = x-max_x
    x2 = x[index]-max_x
    MAE1 = torch.norm(x1[index]-x2, p=1)
    Prob1 = torch.sum(x1[index]==x2)/N
    print('MAE1: %.10f, Prob1: %.10f'%(MAE1, Prob1))

    ex1 = torch.exp(x1)
    ex2 = torch.exp(x2)
    MAE2 = torch.norm(ex1[index]-ex2, p=1)
    Prob2 = torch.sum(ex1[index]==ex2)/N
    print('MAE2: %.10f, Prob2: %.10f'%(MAE2, Prob2))

    sum_ex1 = ex1.sum()
    sum_ex2 = ex2.sum()
    # sum_ex1 = torch.sum(ex1)
    # sum_ex2 = torch.sum(ex2)
    MAE3 = torch.norm(sum_ex1-sum_ex2, p=1)
    Prob3 = torch.sum(sum_ex1==sum_ex2)
    print('MAE3: %.10f, Prob3: %.10f'%(MAE3, Prob3))

    y1 = x1 / sum_ex1
    y2 = x2 / sum_ex2
    MAE4 = torch.norm(y1[index]-y2, p=1)
    Prob4 = torch.sum(y1[index]==y2)/N
    print('MAE4: %.10f, Prob4: %.10f'%(MAE4, Prob4))

    return MAE1, Prob1, MAE2, Prob2, MAE3, Prob3, MAE4, Prob4

def softmax(x, dim=0):
    max_x = x.max()
    x = x-max_x
    sum_ex = torch.exp(x).sum()
    y = x / sum_ex
    return y


seed_everything(seed=42)

N = int(1000)
K = 100

total_MAE1, total_Prob1, total_MAE2, total_Prob2, total_MAE3, total_Prob3, total_MAE4, total_Prob4 = 0, 0, 0, 0, 0, 0, 0, 0
for k in range(K):
    x = torch.randn(N)
    index = torch.randperm(N)
    MAE1, Prob1, MAE2, Prob2, MAE3, Prob3, MAE4, Prob4 = softmax2(x, index, dim=0)
    total_MAE1 += MAE1
    total_MAE2 += MAE2
    total_MAE3 += MAE3
    total_MAE4 += MAE4
    total_Prob1 += Prob1
    total_Prob2 += Prob2
    total_Prob3 += Prob3
    total_Prob4 += Prob4

print('total_MAE1: %.10f, total_Prob1: %.10f'%(total_MAE1/K, total_Prob1/K))
print('total_MAE2: %.10f, total_Prob2: %.10f'%(total_MAE2/K, total_Prob2/K))
print('total_MAE3: %.10f, total_Prob3: %.10f'%(total_MAE3/K, total_Prob3/K))
print('total_MAE4: %.10f, total_Prob4: %.10f'%(total_MAE4/K, total_Prob4/K))

# total_std, total_MAE, total_Prob = 0, 0, 0
# for k in range(K):
#     x = torch.randn(N)
#     index = torch.randperm(N)
#     y1 = softmax(x, dim=0)[index]
#     y2 = softmax(x[index], dim=0)
#     std = torch.std(y1)
#     MAE = torch.norm(y1-y2, p=1)
#     Prob = torch.sum(y1==y2)/N
#     print('Std: %.10f, MAE: %.10f, Prob: %.10f'%(std, MAE, Prob))
#     total_std += std
#     total_MAE += MAE
#     total_Prob += Prob
# print('total_Std: %.10f, total_MAE: %.10f, total_Prob: %.10f'%(total_std/K, total_MAE/K, total_Prob/K))




