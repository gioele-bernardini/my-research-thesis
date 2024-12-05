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
num_epochs = 50
batch_size = 100
learning_rate = 0.0001

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

transform = torchaudio.transforms.MelSpectrogram(
  sample_rate=sample_rate,
  n_mels=num_mel_bins
)

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
      features = features.log2()  # Log-Mel Spectrogram
    else:
      features = waveform

    # Normalization
    features = (features - features.mean()) / (features.std() + 1e-5)

    return features, label

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
example_data, _ = next(iter(train_loader))
input_size = example_data.shape[1] * example_data.shape[2] * example_data.shape[3]  # Correction here

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

# Definition of the simplified neural network with additional fully connected layers
class NeuralNetworkSimplified(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
    super(NeuralNetworkSimplified, self).__init__()

    self.l1 = BinarizeLinearSTE(input_size, hidden_size1)
    self.bn1 = nn.BatchNorm1d(hidden_size1)
    self.htanh1 = nn.Hardtanh()
    self.dropout1 = nn.Dropout(p=0.0)

    self.l2 = BinarizeLinearSTE(hidden_size1, hidden_size2)
    self.bn2 = nn.BatchNorm1d(hidden_size2)
    self.htanh2 = nn.Hardtanh()
    self.dropout2 = nn.Dropout(p=0.0)

    self.l3 = BinarizeLinearSTE(hidden_size2, hidden_size3)
    self.bn3 = nn.BatchNorm1d(hidden_size3)
    self.htanh3 = nn.Hardtanh()
    self.dropout3 = nn.Dropout(p=0.0)

    self.l4 = BinarizeLinearSTE(hidden_size3, num_classes)

  def forward(self, x):
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
hidden_size1 = 500  # You can modify this if necessary
hidden_size2 = 300  # New layer
hidden_size3 = 200  # New layer
model = NeuralNetworkSimplified(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
model.train()

for epoch in range(num_epochs):
  for i, (features, labels) in enumerate(train_loader):
    features = features.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(features)
    loss = criterion(outputs, labels)

    # Backward and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the model
model.eval()

with torch.no_grad():
  correct = 0
  total = 0

  for features, labels in test_loader:
    features = features.to(device)
    labels = labels.to(device)

    outputs = model(features)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  print('Model accuracy on the test set: {:.2f}%'
      .format(100 * correct / total))
