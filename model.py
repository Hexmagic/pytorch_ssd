import torch
import torch.nn as nn
import torch.nn.init as init
from torch import nn as nn
from torch.autograd import Function, Variable
from typing import List


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.cfg = [
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512
        ]
        self.features = self.create_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        first = self.features[:23](x)
        second = self.features[23:](first)
        return first, second

    def create_layers(self, batch_norm=False):
        layers = []
        in_channels = 3
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
                ]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        layers += [pool5]
        return nn.Sequential(*layers)


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        #x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(
            x) * x
        return out


class Block(nn.Module):
    def __init__(self, first, second, third, stride=2, padding=1,norm=True):
        super(Block, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(first, second, 1, 1),
                                 nn.BatchNorm2d(second) if norm else nn.Sequential(), nn.ReLU(True),
                                 nn.Conv2d(second, third, 3, stride, padding),
                                 nn.BatchNorm2d(third) if norm else
                                      nn.Sequential(), nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class SSD(nn.Module):
    def __init__(self, vgg):
        super(SSD, self).__init__()
        self.vgg = vgg
        self.l2_norm = L2Norm(512, 20)
        self.conv6_7 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )
        self.down1 = Block(1024, 256, 512)
        self.down2 = Block(512, 128, 256)
        self.down3 = Block(256, 128, 256, stride=1, padding=0)
        self.down4 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feature_maps: List = []
        first, second = self.vgg(x)
        x = self.l2_norm(first)
        feature_maps.append(x)
        x = self.conv6_7(second)
        feature_maps.append(x)
        x = self.down1(x)
        feature_maps.append(x)
        x = self.down2(x)
        feature_maps.append(x)
        x = self.down3(x)
        feature_maps.append(x)
        x = self.down4(x)
        feature_maps.append(x)
        print(feature_maps)


sdd = SSD(VGG())
data = torch.rand((1, 3, 300, 300))
sdd(data)
