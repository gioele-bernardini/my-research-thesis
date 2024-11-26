#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load Commands List
commands_list_file = 'commands_list.txt'
with open(commands_list_file, 'r') as f:
  commands = f.read().splitlines()
print("Commands to use:", commands)

# 2. Prepare Dataset
DATA_DIR = 'speech-commands'

# Parameters
SAMPLE_RATE = 16000
DURATION = 1  # in seconds
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION

class SpeechCommandsDataset(Dataset):
  def __init__(self, commands, data_dir, scaler=None, transform=None):
    self.X = []
    self.y = []
    self.transform = transform
    self.scaler = scaler
    label_to_index = {label: idx for idx, label in enumerate(commands)}
    for label in commands:
      label_dir = os.path.join(data_dir, label)
      if not os.path.exists(label_dir):
        print(f"Warning: Directory {label_dir} does not exist.")
        continue
      files = os.listdir(label_dir)
      for file in files:
        file_path = os.path.join(label_dir, file)
        if not file_path.endswith('.wav'):
          continue
        self.X.append(file_path)
        self.y.append(label_to_index[label])

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    file_path = self.X[idx]
    label = self.y[idx]
    # Load audio
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    # Pad or truncate
    if len(audio) < SAMPLES_PER_FILE:
      audio = np.pad(audio, (0, SAMPLES_PER_FILE - len(audio)))
    else:
      audio = audio[:SAMPLES_PER_FILE]
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)  # Shape: (40,)
    # Normalize
    if self.scaler:
      mfccs = self.scaler.transform([mfccs])[0]
    # Convert to tensor
    mfccs = torch.tensor(mfccs, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.long)
    return mfccs, label

# Initialize scaler
scaler = StandardScaler()

# Create dataset to compute scaler parameters
print("Computing scaler parameters...")
dataset_for_scaler = SpeechCommandsDataset(commands, DATA_DIR)
all_features = []
for idx in range(len(dataset_for_scaler)):
  mfccs, _ = dataset_for_scaler[idx]
  all_features.append(mfccs.numpy())
scaler.fit(all_features)
print("Scaler parameters computed.")

# Create full dataset with normalized features
print("Creating full dataset...")
full_dataset = SpeechCommandsDataset(commands, DATA_DIR, scaler=scaler)
print(f"Total samples: {len(full_dataset)}")

# Split dataset into training and testing sets
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
  full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
)

# 3. Build Model
input_dim = 40  # Number of MFCC features
num_classes = len(commands)

# Custom binarization function
def binarize(tensor):
  return tensor.sign()

# Binary Linear Layer
class BinaryLinear(nn.Linear):
  def forward(self, input):
    if self.training:
      weight = self.weight
    else:
      weight = binarize(self.weight)
    return F.linear(input, weight, self.bias)

# Define the Binary Neural Network Model
class BNN(nn.Module):
  def __init__(self, input_dim, num_classes):
    super(BNN, self).__init__()
    self.fc1 = BinaryLinear(input_dim, 256, bias=False)
    self.bn1 = nn.BatchNorm1d(256)
    self.fc2 = BinaryLinear(256, 128, bias=False)
    self.bn2 = nn.BatchNorm1d(128)
    self.fc3 = nn.Linear(128, num_classes, bias=False)

  def forward(self, x):
    x = self.bn1(self.fc1(x))
    x = binarize(x)
    x = self.bn2(self.fc2(x))
    x = binarize(x)
    x = self.fc3(x)
    return x

model = BNN(input_dim, num_classes)

# 4. Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 5. Training Loop
num_epochs = 30
for epoch in range(num_epochs):
  model.train()
  running_loss = 0.0
  correct_train = 0
  total_train = 0
  for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    # Zero the parameter gradients
    optimizer.zero_grad()
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    # Statistics
    running_loss += loss.item() * inputs.size(0)
    _, predicted = torch.max(outputs.data, 1)
    total_train += labels.size(0)
    correct_train += (predicted == labels).sum().item()
  epoch_loss = running_loss / len(train_dataset)
  epoch_acc = 100 * correct_train / total_train

  # Validation
  model.eval()
  correct_test = 0
  total_test = 0
  with torch.no_grad():
    for inputs, labels in test_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total_test += labels.size(0)
      correct_test += (predicted == labels).sum().item()
  test_acc = 100 * correct_test / total_test

  print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
      f"Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%")

print("Training complete.")

# 6. Save Weights
weights_dir = 'binary_weights'
os.makedirs(weights_dir, exist_ok=True)

# Save weights layer by layer
def save_weights(model, weights_dir):
  for idx, layer in enumerate(model.children()):
    if isinstance(layer, (BinaryLinear, nn.Linear)):
      weight = layer.weight.data.cpu().numpy()
      weight_file = os.path.join(weights_dir, f'layer_{idx}_weights.npy')
      np.save(weight_file, weight)
      print(f"Saved weights for layer {idx}.")
    elif isinstance(layer, nn.BatchNorm1d):
      # Save BatchNorm parameters if needed
      pass

save_weights(model, weights_dir)
print("All weights saved in", weights_dir)

