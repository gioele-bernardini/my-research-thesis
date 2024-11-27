#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# To save weights
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
  # Returns the sign of the tensor (2d-matrix) (standard binarization)
  return tensor.sign()

class BinarizeLinear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super(BinarizeLinear, self).__init__(in_features, out_features, bias)
    
    # Save a copy of the original biases for STE
    if self.bias is not None:
      self.bias.org = self.bias.data.clone()

  def forward(self, input):
    # Binarize *input*
    input.data = binarize(input.data)

    # Binarize *weights* and save original ones for STE
    if not hasattr(self.weight, 'org'):
      self.weight.org = self.weight.data.clone()
    self.weight.data = binarize(self.weight.org)

    # Binarize *biases* and save original ones for STE
    if self.bias is not None:
      if not hasattr(self.bias, 'org'):
        self.bias.org = self.bias.data.clone()
      self.bias.data = binarize(self.bias.org)

    output = nn.functional.linear(input, self.weight, self.bias)

    return output

class NeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNetwork, self).__init__()

    self.l1 = nn.Linear(input_size, hidden_size)
    self.bn1 = nn.BatchNorm1d(hidden_size)
    self.htanh1 = nn.Hardtanh()

    self.l2 = BinarizeLinear(hidden_size, 400)
    self.bn2 = nn.BatchNorm1d(400)
    self.htanh2 = nn.Hardtanh()

    self.l3 = BinarizeLinear(400, 300)
    self.bn3 = nn.BatchNorm1d(300)
    self.htanh3 = nn.Hardtanh()

    self.l4 = nn.Linear(300, num_classes)

  def forward(self, x):
    out = self.l1(x)
    out = self.bn1(out)
    out = self.htanh1(out)

    # out = self.l2.forward(out)
    out = self.l2(out)
    out = self.bn2(out)
    out = self.htanh2(out)

    out = self.l3(out)
    out = self.bn3(out)
    out = self.htanh3(out)

    out = self.l4(out)

    # Return logits
    return out

# Model, loss function and optimizer
model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
# Does *not* train, but sets the mode for the model
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
    # optimizer.step() -> this leads to error

    # Straight through estimator
    # This is used to update the weights before binarizing them
    for p in list (model.parameters()):
      if hasattr(p, 'org'):
        p.data.copy_(p.org)

    optimizer.step()

    for p in list(model.parameters()):
      if hasattr(p, 'org'):
        p.org.copy_(p.data.clamp_(-1, 1))

    if (i + 1) % 100 == 0:
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

def save_weights(model, directory='binarized-weights'):
  # Create folder if does not exist
  if not os.path.exists(directory):
    os.makedirs(directory)

  layer_idx = 1
  for name, layer in model.name_modules():
    # Check if the layer is not a batch normalization one
    if isinstance(layer, (nn.Linear, BinarizeLinear)):
      # Name of the files for weights and biases
      weight_file = os.path.join(directory, f'layer{layer_idx}_weights.txt')
      bias_file = os.path.join(directory, f'layer{layer_idx}_biases.txt')

      # Extract and binarize weights
      if hasattr(layer, 'weight'):
        if isinstance(layer, BinarizeLinear):
          #
          weights = binarize(layer.weight.org).cpu().numpy()
          fmt = '%d'
        else:
          # Linear layer, keep real weights
          weights = layer.weight.data.cpu().numpy()
          fmt = '%.6f'
        
        np.savetxt(weight_file, weights, fmt=fmt)

        # Estrazione e binarizzazione dei bias
    if hasattr(layer, 'bias') and layer.bias is not None:
      if isinstance(layer, BinarizeLinear):
        # I bias sono binarizzati, accediamo a layer.bias.org
        biases = binarize(layer.bias.org).cpu().numpy()
        fmt = '%d'  # Formato per valori binari (-1, 1)
      else:
        # Per nn.Linear, mantieni i bias reali
        biases = layer.bias.data.cpu().numpy()
        fmt = '%.6f'  # Formato per valori reali
      # Salva i bias nel file
      np.savetxt(bias_file, biases, fmt=fmt)

    layer_idx += 1

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

save_weights()
print('Weights and Biases correctly saved')

