#!/usr/bin/env python3

# Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda_is_available() else 'cpu')

# Hyper-parameters
input_size = 784  # 28 x 28 pixels
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
  root='./MNIST-data/',
  train=True,
  transform=transforms.ToTensor,
  download=True
)

test_dataset = torchvision.datasets.MNIST(
  root='./MNIST-data/',
  train=False,
  transform=transforms.ToTensor
)

# Data loader
train_loader = torch.utils.data.DataLoader(
  dataset=train_dataset,
  batch_size=batch_size,
  shuffle=True
)

test_loader = torch.utils.data.DataLoader(
  dataset=test_dataset,
  shuffle=False
)

def binarize(tensor):
  # Returns the sign of the vector (standard binarization)
  return tensor.sign()

class BinarizeLinear(nn.Linear):
  def __init__(self, in_features, out_features):
    super(BinarizeLinear, self).__init__(in_features, out_features)

  def forward(self, input):
    # Matrices default moltiplication
    # input * weight

    # binarize input
    input.data = binarize(input.data)

    # Binarize weight AND save original ones for STE phase
    if not hasattr(self.weight, 'org'):
      self.weight.org = self.weight.data.clone()

    self.weight.data = binarize(self.weight.org)

    res = nn.functional.linear(input, self.weight)

    return res



