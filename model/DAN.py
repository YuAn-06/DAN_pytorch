"""
Copyright (C) 2024
@ Name: DAN.py
@ Time: 2024/1/12 15:57
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    def __init__(self,input_dim):
        super(Extractor,self).__init__()
        self.input_dim = input_dim
        # self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=5)
        self.dense1 = nn.Linear(self.input_dim,128)
        self.bn1 = nn.BatchNorm1d(128)
        # self.conv2 = nn.Conv2d(64, 50, kernel_size=5)
        self.dense2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)

    def forward(self, input):
        # x = F.max_pool2d(F.relu((self.bn1(self.conv1(input)))), 2)
        # x = F.max_pool2d(F.relu((self.conv2_drop(self.bn2(self.conv2(x))))), 2)
        # x = x.view(-1, 50 * 4 * 4)
        x = self.dense1(input)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.fc2(x)
        x = self.bn4(x)

        return x


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc3 = nn.Linear(16, 1)

    def forward(self, input):
        y = self.fc3(input)

        return y
