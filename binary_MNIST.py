#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
  transform=transforms.ToTensor(),
  download=True
)

test_dataset = torchvision.datasets.MNIST(
  root='./MNIST-data/',
  train=False,
  transform=transforms.ToTensor()
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
  # Returns the sign of the tensor (standard binarization)
  return tensor.sign()

class BinarizeLinear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super(BinarizeLinear, self).__init__(in_features, out_features, bias)
    # Save a copy of the original weights and biases for STE
    if not hasattr(self.weight, 'org'):
      self.weight.org = self.weight.data.clone()
    if self.bias is not None and not hasattr(self.bias, 'org'):
      self.bias.org = self.bias.data.clone()

  def forward(self, input):
    # Binarize input
    input.data = binarize(input.data)

    # Binarize weights and biases
    self.weight.data = binarize(self.weight.org)
    if self.bias is not None:
      self.bias.data = binarize(self.bias.org)

    output = nn.functional.linear(input, self.weight, self.bias)
    return output

class NeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNetwork, self).__init__()

    # Replace nn.Linear with BinarizeLinear for all layers
    self.l1 = BinarizeLinear(input_size, hidden_size)
    self.bn1 = nn.BatchNorm1d(hidden_size)
    self.htanh1 = nn.Hardtanh()

    self.l2 = BinarizeLinear(hidden_size, 400)
    self.bn2 = nn.BatchNorm1d(400)
    self.htanh2 = nn.Hardtanh()

    self.l3 = BinarizeLinear(400, 300)
    self.bn3 = nn.BatchNorm1d(300)
    self.htanh3 = nn.Hardtanh()

    self.l4 = BinarizeLinear(300, num_classes)

  def forward(self, x):
    out = self.l1(x)
    out = self.bn1(out)
    out = self.htanh1(out)

    out = self.l2(out)
    out = self.bn2(out)
    out = self.htanh2(out)

    out = self.l3(out)
    out = self.bn3(out)
    out = self.htanh3(out)

    out = self.l4(out)
    return out

# Model, loss function and optimizer
model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
model.train()

for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()

    # Straight Through Estimator
    # Restore original weights before optimizer step
    for p in list(model.parameters()):
      if hasattr(p, 'org'):
        p.data.copy_(p.org)

    optimizer.step()

    # Clamping weights and biases
    for p in list(model.parameters()):
      if hasattr(p, 'org'):
        p.org.copy_(p.data.clamp_(-1, 1))

    if (i + 1) % 100 == 0:
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
          .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

def save_weights(model, directory='binarized-weights'):
  # Create folder if it does not exist
  if not os.path.exists(directory):
    os.makedirs(directory)

  layer_idx = 1
  for name, layer in model.named_children():
    # Check if the layer is a BinarizeLinear layer
    if isinstance(layer, BinarizeLinear):
      # Names of the files for weights and biases
      weight_file = os.path.join(directory, f'layer{layer_idx}_weights.txt')
      bias_file = os.path.join(directory, f'layer{layer_idx}_biases.txt')

      # Extract and binarize weights
      if hasattr(layer, 'weight'):
        weights = binarize(layer.weight.org).cpu().numpy()
        fmt = '%d'
        np.savetxt(weight_file, weights, fmt=fmt)

      # Extract and binarize biases
      if hasattr(layer, 'bias') and layer.bias is not None:
        biases = binarize(layer.bias.org).cpu().numpy()
        fmt = '%d'
        np.savetxt(bias_file, biases, fmt=fmt)

      layer_idx += 1

  print('Weights and biases saved in directory:', directory)

# Finally, test the model
model.eval()

with torch.no_grad():
  correct = 0
  total = 0

  for images, labels in test_loader:
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  print('Accuracy of the network on the 10000 test images: {} %.'
      .format(100 * correct / total))

# Save binarized weights and biases
save_weights(model)
