import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms


class DiscConvNet(nn.Module):
  def __init__(self, input_length = 16):
    super(DiscConvNet, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 128), stride=(2,2))
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2))
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2))
    self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2))
    self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 2))
    # self.fc1 = nn.Linear(64 * 2, 128)
    # self.fc2 = nn.Linear(128, 128)
    self.out = nn.Linear(16 * 1, 1)
    self.prelu = nn.PReLU()
    self.sigmoid = nn.Sigmoid()
    self.batch_norm_2d = nn.BatchNorm2d(16)
    self.batch_norm_1d = nn.BatchNorm1d(128)
    self.dropout = nn.Dropout(0.3)

  def forward(self, input): # Input size: [batch_size x 1 x 16 x 128]
    x = self.prelu(self.batch_norm_2d(self.conv1(input)))  # [batch_size x 64 x 8 x 1]
    fm = x.clone()
    x = self.dropout(x)
    x = self.prelu(self.batch_norm_2d(self.conv2(x)))  # [batch_size x 16 x 4 x 1]
    x = self.dropout(x)
    x = self.prelu(self.batch_norm_2d(self.conv3(x)))  # [batch_size x 16 x 2 x 1]
    x = self.dropout(x)
    x = self.prelu(self.batch_norm_2d(self.conv4(x)))  # [batch_size x 16 x 2 x 1]
    x = self.dropout(x)
    # x = self.prelu(self.batch_norm_2d(self.conv5(x))) # [batch_size x 16 x 1 x 1]
    x = x.flatten(1, -1) # [batch_size x 128]
    # x = self.prelu(self.batch_norm_1d(self.fc1(x))) # [batch_size x 128]
    # x = self.prelu(self.batch_norm_1d(self.fc2(x))) # [batch_size x 128]
    x = self.out(x)
    x_sigmoid = self.sigmoid(x)  # [batch_size x 1]
    return x, x_sigmoid, fm


class GenConvNet(nn.Module):
  def __init__(self, z_dim=100, input_length=16):
    super(GenConvNet, self).__init__()
    self.z_dim = z_dim

    # self.fc1 = nn.Linear(z_dim, 128)
    # self.fc2 = nn.Linear(128, 128)
    self.transpose_conv1 = nn.ConvTranspose2d(in_channels=z_dim, out_channels=64, kernel_size=(2, 1), stride=(2, 2))
    self.transpose_conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(2, 1), stride=(2, 2))
    self.transpose_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(2, 1), stride=(2, 2))
    self.transpose_conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(2, 1), stride=(2, 2))
    self.transpose_conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(2, 128), stride=(2, 2))
    self.prelu = nn.PReLU()
    self.sigmoid = nn.Sigmoid()
    self.batch_norm_2d = nn.BatchNorm2d(64)
    self.batch_norm_1d = nn.BatchNorm1d(128)

  def forward(self, input):
    # x = self.prelu(self.batch_norm_1d(self.fc1(input)))
    # x = self.prelu(self.batch_norm_1d(self.fc2(x)))
    x = input.view(-1, self.z_dim, 1, 1)
    x = self.prelu(self.batch_norm_2d(self.transpose_conv1(x)))
    x = self.prelu(self.batch_norm_2d(self.transpose_conv2(x)))
    x = self.prelu(self.batch_norm_2d(self.transpose_conv3(x)))
    x = self.sigmoid(self.transpose_conv5(x))
    return x





