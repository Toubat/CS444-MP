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

        self.convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) for _ in range(num_layers-1)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(num_layers-1)])

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        identity = x

        for i in range(self.num_layers-1):
            x = self.convs[i](x)

            if self.bn: # BatchNorm
                x = self.bns[i](x)

            x = self.relu(x)


        if self.skip_connect:
            x += identity

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
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
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
        x = self.classifier(x)

        return x


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()

        self.downsample = downsample

        # Conv 3 x 3
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv 3 x 3
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = F.relu(x)

        return x


class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()

        self.downsample = downsample

        # Conv 1 x 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv 3 x 3
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Conv 1 x 1 with expansion
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = F.relu(x)

        return x


class DropoutBlock(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()

        self.downsample = downsample

        # Conv 1 x 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv 3 x 3
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Conv 1 x 1 with expansion
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.dropout = nn.Dropout(p=0.7)


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = F.relu(self.bn1(self.dropout(self.conv1(x))))
        x = F.relu(self.bn2(self.dropout(self.conv2(x))))
        x = self.bn3(self.conv3(x))

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = F.relu(x)

        return x


class ResNet(nn.Module):

    def __init__(self, block, num_layers, input_channels, num_classes):
        super().__init__()

        self.in_channels = 64
        self.block = block

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(num_layers[0], out_channels=64, stride=1)
        self.layer2 = self.make_layer(num_layers[1], out_channels=128, stride=2)
        self.layer3 = self.make_layer(num_layers[2], out_channels=256, stride=2)
        self.layer4 = self.make_layer(num_layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


    def make_layer(self, num_blocks, out_channels, stride):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.block.expansion)
            )

        layers.append(self.block(self.in_channels, out_channels, downsample, stride))
        self.in_channels = out_channels * self.block.expansion

        for i in range(1, num_blocks):
            layers.append(self.block(self.in_channels, out_channels))

        return nn.Sequential(*layers)