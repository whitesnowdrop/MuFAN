''' Network structure of MuFAN in PyTorch.

Mainly adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py and https://github.com/XingangPan/IBN-Net/blob/master/ibnnet/resnet_ibn.py

Reference:
[1] Jung, Dahuin, et al. "New Insights for the Stability-Plasticity Dilemma in Online Continual Learning."
    International Conference on Learning Representations 2023.
'''

import torch.nn as nn
import torch.nn.functional as F
from .spnorm import SPNorm

class BasicBlock_MuFAN(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, stride=1):
        super(BasicBlock_MuFAN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.spnorm1 = SPNorm(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.spnorm2 = SPNorm(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                SPNorm(out_planes)
            )

    def forward(self, x):
        out = self.spnorm1(self.conv1(x))
        out = F.relu(out)
        out = self.spnorm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_MuFAN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_stride=1):

        super(ResNet_MuFAN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=init_stride, padding=1, bias=False)
        self.spnorm1 = SPNorm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, planes, stride))
        self.in_planes = planes

        for i in range(1,num_blocks):
            layers.append(block(self.in_planes, planes, planes, 1))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.spnorm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out_features = out.view(out.size(0), -1)
        out = self.linear(out_features)
        return out


def ResNet18_MuFAN(num_classes=10, init_stride=1):
    return ResNet_MuFAN(BasicBlock_MuFAN, [2, 2, 2, 2], num_classes=num_classes, init_stride=init_stride)

def ResNet34_MuFAN(num_classes=10, init_stride=1):
    return ResNet_MuFAN(BasicBlock_MuFAN, [3,4,6,3],  num_classes=num_classes, init_stride=init_stride)

