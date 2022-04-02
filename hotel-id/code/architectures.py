import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class FirstNet(nn.Module):
    del __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
    self.batchnorm1 = nn.BatchNorm2d(64)

    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
    self.batchnorm2 = nn.BatchNorm2d(128)

    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
    self.batchnorm3 = nn.BatchNorm2d(256)

    self.maxpool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(256 * 2 * 2, 512)
    self.fc2 = nn.Linear(512, 3)


    def forward(self,x):
        x = nn.ReLU(self.conv1(x))
        x = nn.ReLU(self.conv2(x))
        x = self.batchnorm1(x)
        x = self.maxpool(x)
        x = nn.ReLU(self.conv3(x))
        x = self.batchnorm2(x)
        x = self.maxpool(x)
        x = nn.ReLU(self.conv4(x))
        x = self.batchnorm3(x)
        x = self.maxpool(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x