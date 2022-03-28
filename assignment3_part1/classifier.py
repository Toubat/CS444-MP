import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import Tensor, optim
import numpy as np

NUM_CLASSES = 21


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers: int, bn: bool, skip_connect: bool):
        super().__init__()

        self.num_layers = num_layers
        self.skip_connect = skip_connect
        self.bn = bn

        self.downsample = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(out_channels)
        )

        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)] +
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) for _ in range(num_layers-1)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(num_layers)])

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        for i in range(self.num_layers):
            x = self.convs[i](x)

            if self.bn: # BatchNorm
                x = self.bns[i](x)

            x = self.relu(x)

        if self.skip_connect:
            identity = self.downsample(identity)
            x = x + identity

        x = self.maxpool(x)
        return x


class ConvNet(nn.Module):

    def __init__(self, num_layers, bn=False, skip_connect=False, num_classes=21):
        super().__init__()

        self.in_channels = 3

        self.layer1 = ConvBlock(self.in_channels, 64, num_layers[0], bn, skip_connect)
        self.layer2 = ConvBlock(64, 128, num_layers[1], bn, skip_connect)
        self.layer3 = ConvBlock(128, 256, num_layers[2], bn, skip_connect)
        self.layer4 = ConvBlock(256, 512, num_layers[3], bn, skip_connect)
        self.layer5 = ConvBlock(512, 512, num_layers[4], bn, skip_connect)

        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 5 * 5, 4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )


    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        print("ASDADAS")

        x = self.classifier(x)

        return x
