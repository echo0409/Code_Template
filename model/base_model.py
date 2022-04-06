# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Model structure
class SubModel(nn.Module):
    def __init__(self):
        # 在构造函数中，实例化不同的layer组件，并赋给类成员变量
        super(SubModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(200, 50)

    def forward(self, x):
        # 在前馈函数中，利用实例化的组件对网络进行搭建，并对输入Tensor进行操作，并返回Tensor类型的输出结果
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 200)
        x = F.relu(self.fc1(x))
        return x


class Net(nn.Module):
    def __init__(self):
        # 在构造函数中，实例化不同的layer组件，并赋给类成员变量
        super(Net, self).__init__()
        self.submodel = SubModel()
        self.fc = nn.Linear(100, 10)

    def forward(self, x1, x2):
        # 在前馈函数中，利用实例化的组件对网络进行搭建，并对输入Tensor进行操作，并返回Tensor类型的输出结果
        x1 = self.submodel(x1)
        x2 = self.submodel(x2)
        x = torch.cat((x1,x2), 1)
        x = self.fc(x)
        return x1,x2,x

