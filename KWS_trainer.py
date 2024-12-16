#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_classes = None  # Will be determined based on the commands to load
num_epochs = 100
batch_size = 512
learning_rate = 0.001

# Dataset and commands directory
dataset_dir = './speech-commands'
commands_file = './commands_list.txt'

# Loading commands to consider for training
with open(commands_file, 'r') as f:
  commands = f.read().splitlines()

num_classes = len(commands)

# Creating a mapping between commands and classes
command_to_index = {command: idx for idx, command in enumerate(commands)}

# Definition of audio transformations
sample_rate = 16000  # Standard sampling rate for the dataset
num_mel_bins = 64  # Number of Mel bands for the Mel-Spectrogram

melkwargs = {"n_mels": 64, "n_fft": 400, "hop_length": 256, "center": False}
transform = torchaudio.transforms.MFCC(
 n_mfcc = 13,
 sample_rate=sample_rate,
 melkwargs = melkwargs,
 log_mels = True,
)

def binarize(tensor):
  # Returns the sign of the tensor (standard binarization)
  return tensor.sign()

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

class BinarizeLinear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super(BinarizeLinear, self).__init__(in_features, out_features, bias)
    # Save a copy of the original weights and biases for STE
    self.weight.org = self.weight.data.clone()
    self.bias.org = self.bias.data.clone()

  def forward(self, input):
    # Binarize input
    input.data = binarize(input.data)

    # Binarize weights and biases
    self.weight.data = binarize(self.weight.org).to(self.weight.device)
    if self.bias is not None:
      self.bias.data = binarize(self.bias.org).to(self.bias.device)

    output = nn.functional.linear(input, self.weight, self.bias)
    return output
# Binarization function with STE
class BinarizeSTE(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    return torch.where(input >= 0, torch.ones_like(input), -torch.ones_like(input))

  @staticmethod
  def backward(ctx, grad_output):
    # STE gradient: passes the gradient without modification
    return grad_output

binarize_ste = BinarizeSTE.apply

# Custom Dataset class
class SpeechCommandsDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_dir, commands, transform=None):
    self.dataset_dir = dataset_dir
    self.commands = commands
    self.transform = transform
    self.samples = []

    for command in self.commands:
      command_dir = os.path.join(self.dataset_dir, command)
      if os.path.isdir(command_dir):
        for filename in os.listdir(command_dir):
          if filename.endswith('.wav'):
            filepath = os.path.join(command_dir, filename)
            self.samples.append((filepath, command_to_index[command]))
      else:
        print(f'Warning: The folder {command_dir} does not exist.')

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    filepath, label = self.samples[idx]
    waveform, sr = torchaudio.load(filepath)

    # Resample if necessary
    if sr != sample_rate:
      resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
      waveform = resampler(waveform)

    # Standardize duration to 1 second (16000 samples)
    if waveform.size(1) < sample_rate:
      padding = sample_rate - waveform.size(1)
      waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
      waveform = waveform[:, :sample_rate]

    # Apply transformation
    if self.transform:
      features = self.transform(waveform)
    else:
      features = waveform

    # Normalization
    features = (features - features.mean()) / (features.std() + 1e-5)
    if torch.isnan(features.flatten()).sum():
        assert 'oh no'
        print('oh no 3 ')
        x = torch.where(torch.isnan(features.flatten()))[0]
        print(features[x[0]])
        print(features.mean())
        print(features.std())
        exit()

    return features, label, filepath

# Splitting dataset into training and test sets
full_dataset = SpeechCommandsDataset(dataset_dir, commands, transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# Data loaders
train_loader = torch.utils.data.DataLoader(
  dataset=train_dataset,
  batch_size=batch_size,
  shuffle=True
)

test_loader = torch.utils.data.DataLoader(
  dataset=test_dataset,
  batch_size=batch_size,
  shuffle=False
)

# Compute input size
# The Mel-Spectrogram will have dimensions: (batch_size, channels, n_mels, time_steps)
example_data, _, _= next(iter(train_loader))
input_size = example_data.shape[1] * example_data.shape[2] * example_data.shape[3]  # Correction here

class BinarizeLinear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super(BinarizeLinear, self).__init__(in_features, out_features, bias)
    # Save a copy of the original weights and biases for STE
    self.weight.org = self.weight.data.clone()
    self.bias.org = self.bias.data.clone()

  def forward(self, input):
    # Binarize input
    # input.data = binarize(input.data)

    # Binarize weights and biases
    self.weight.data = binarize(self.weight.org).to(self.weight.device)
    if self.bias is not None:
      self.bias.data = binarize(self.bias.org).to(self.bias.device)

    output = nn.functional.linear(input, self.weight, self.bias)
    return output

# BinarizeLinearSTE class with STE
class BinarizeLinearSTE(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super(BinarizeLinearSTE, self).__init__(in_features, out_features, bias)
    # Xavier initialization
    nn.init.xavier_uniform_(self.weight)
    if self.bias is not None:
      nn.init.constant_(self.bias, 0)

  def forward(self, input):
    weight_bin = binarize_ste(self.weight)
    bias_bin = binarize_ste(self.bias) if self.bias is not None else None
    input_bin = binarize_ste(input)
    output = F.linear(input_bin, weight_bin, bias_bin)
    return output

class NeuralNetworkSimplified(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
    super(NeuralNetworkSimplified, self).__init__()

    self.l1 = BinarizeLinear(input_size, hidden_size1)
    self.bn1 = nn.BatchNorm1d(hidden_size1)
    self.htanh1 = nn.Hardtanh()
    self.dropout1 = nn.Dropout(p=0.0)

    self.l2 = BinarizeLinear(hidden_size1, hidden_size2)
    self.bn2 = nn.BatchNorm1d(hidden_size2)
    self.htanh2 = nn.Hardtanh()
    self.dropout2 = nn.Dropout(p=0.0)

    self.l3 = BinarizeLinear(hidden_size2, hidden_size3)

    self.bn3 = nn.BatchNorm1d(hidden_size3)
    self.htanh3 = nn.Hardtanh()
    self.dropout3 = nn.Dropout(p=0.0)
    self.l4 = nn.Linear(hidden_size3, num_classes)

  def forward(self, x, y):
    x = x.view(x.size(0), -1)
    out = self.l1(x)
    out = self.bn1(out)
    out = self.htanh1(out)
    out = self.dropout1(out)

    out = self.l2(out)
    out = self.bn2(out)
    out = self.htanh2(out)
    out = self.dropout2(out)

    out = self.l3(out)
    out = self.bn3(out)
    out = self.htanh3(out)
    out = self.dropout3(out)

    out = self.l4(out)
    return out
# Definition of the simplified neural network with additional fully connected layers
class NeuralNetworkSimplifiedSTE(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
    super(NeuralNetworkSimplifiedSTE, self).__init__()

    self.l1 = BinarizeLinearSTE(input_size, hidden_size1)
    # self.l1 = BinarizeLinear(input_size, hidden_size1)
    # self.bn1 = nn.BatchNorm1d(hidden_size1)
    self.htanh1 = nn.Hardtanh()
    self.dropout1 = nn.Dropout(p=0.0)

    self.l2 = BinarizeLinearSTE(hidden_size1, hidden_size2)
    # self.l2 = BinarizeLinear(hidden_size1, hidden_size2)
    # self.bn2 = nn.BatchNorm1d(hidden_size2)
    self.htanh2 = nn.Hardtanh()
    self.dropout2 = nn.Dropout(p=0.0)

    # self.l3 = BinarizeLinear(hidden_size2, hidden_size3)
    self.l3 = BinarizeLinearSTE(hidden_size2, hidden_size3)
    # self.bn3 = nn.BatchNorm1d(hidden_size3)
    self.htanh3 = nn.Hardtanh()
    self.dropout3 = nn.Dropout(p=0.0)

    self.l4 = BinarizeLinearSTE(hidden_size3, num_classes)
    # self.l4 = BinarizeLinear(hidden_size3, num_classes)

  def forward(self, x, y):
    x = x.view(x.size(0), -1)
    out = self.l1(x)
    out = self.bn1(out)
    out = self.htanh1(out)
    out = self.dropout1(out)

    out = self.l2(out)
    out = self.bn2(out)
    out = self.htanh2(out)
    out = self.dropout2(out)

    out = self.l3(out)
    out = self.bn3(out)
    out = self.htanh3(out)
    out = self.dropout3(out)

    out = self.l4(out)
    return out

# Model hyperparameters
hidden_size1 = 512  # You can modify this if necessary
hidden_size2 = 512  # New layer
hidden_size3 = 512  # New layer
model = NeuralNetworkSimplified(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes).to(device)
# model = NeuralNetworkSimplifiedSTE(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
model.train()

for epoch in range(num_epochs):
  correct, total = 0, 0
  for i, (features, labels, fp) in enumerate(train_loader):
    # print(fp)
    features = features.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(features, fp)
    loss = criterion(outputs, labels)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    # Backward and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Acc: {100* correct/total:.2f}%')

# Evaluating the model
model.eval()

with torch.no_grad():
  correct = 0
  total = 0

for features, labels, fp in test_loader:
  features = features.to(device)
  labels = labels.to(device)

  # Passa entrambi gli argomenti richiesti dal modello
  outputs = model(features, fp)
  _, predicted = torch.max(outputs.data, 1)
  total += labels.size(0)
  correct += (predicted == labels).sum().item()

print('\nModel accuracy on the test set: {:.2f}%'
    .format(100 * correct / total))

