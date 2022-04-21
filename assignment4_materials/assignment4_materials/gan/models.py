import torch
from .spectral_normalization import SpectralNorm
from torch import nn
import torch.nn.functional as F

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=4, stride=2, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1))
        self.bn1 = nn.BatchNorm2d(256)
        self.conv3 = SpectralNorm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1))
        self.bn2 = nn.BatchNorm2d(512)
        self.conv4 = SpectralNorm(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1))
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv5 = SpectralNorm(nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=1))

        self.leaky_relu = nn.LeakyReLU(0.2)
        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)

        x = self.conv5(x)
        ##########       END      ##########

        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super().__init__()
        self.noise_dim = noise_dim

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.t_conv1 = nn.ConvTranspose2d(self.noise_dim, 1024, kernel_size=4, stride=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.t_conv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.t_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.t_conv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.t_conv5 = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.t_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.t_conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.t_conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.t_conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.t_conv5(x)
        x = self.tanh(x)
        ##########       END      ##########

        return x