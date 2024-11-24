#!/usr/bin/env python3

import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import torch.optim as optim
from tqdm import tqdm

# Constants
WEIGHTS_DIR = 'weights'  # Directory to save binarized weights
COMMANDS_LIST_FILE = 'commands_list.txt'
DATA_PATH = './speech-commands/'
TRAIN_SIZE = 800  # Number of samples for training
TEST_SIZE = 200   # Number of samples for testing
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
SAMPLE_RATE = 16000  # Sampling rate used for MFCC

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the list of keywords from the saved file
def load_keywords(filepath):
  with open(filepath, 'r') as f:
    return [line.strip() for line in f]

keywords = load_keywords(COMMANDS_LIST_FILE)

# Custom Dataset for Speech Commands
class SpeechCommandsDataset(Dataset):
  def __init__(self, data_path, keywords, transform=None):
    self.data = []
    self.labels = []
    self.transform = transform
    self.keywords = keywords
    self.classes = {word: idx for idx, word in enumerate(self.keywords)}
    
    for label in os.listdir(data_path):
      if label in self.keywords:
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
          for file in os.listdir(label_path):
            if file.endswith('.wav'):
              self.data.append(os.path.join(label_path, file))
              self.labels.append(self.classes[label])
    
  def __len__(self):
    return len(self.data)
    
  def __getitem__(self, idx):
    waveform, sample_rate = torchaudio.load(self.data[idx])
    waveform = self._preprocess_audio(waveform, sample_rate)
    if self.transform:
      waveform = self.transform(waveform)
    label = self.labels[idx]
    return waveform, label
  
  def _preprocess_audio(self, waveform, sample_rate):
    # Fixed duration of 1 second
    fixed_length = 1  # in seconds
    num_samples = int(fixed_length * sample_rate)

    if waveform.size(1) > num_samples:
      waveform = waveform[:, :num_samples]
    else:
      padding = num_samples - waveform.size(1)
      waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform

# Function to extract MFCC features
def extract_features(waveform, sample_rate=SAMPLE_RATE):
  mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=sample_rate,
    n_mfcc=40,
    melkwargs={
      'n_fft': 1024,
      'hop_length': 512,
      'n_mels': 40,
    }
  )
  mfcc = mfcc_transform(waveform)
  return mfcc

# Custom Binarized Linear Layer with STE
class BinarizedLinear(nn.Module):
  def __init__(self, in_features, out_features, bias=True):
    super(BinarizedLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.fc = nn.Linear(in_features, out_features, bias)

  def forward(self, x):
    # Binarize weights using sign function with STE
    binary_weights = self.binarize(self.fc.weight)
    if self.fc.bias is not None:
      return nn.functional.linear(x, binary_weights, self.fc.bias)
    else:
      return nn.functional.linear(x, binary_weights)

  @staticmethod
  def binarize(weights):
    # Binarize weights to -1 and +1
    binary_weights = weights.sign()
    binary_weights[binary_weights == 0] = 1  # Handle zero weights
    return binary_weights

# Define the Binarized MLP Model
class BinarizedMLP(nn.Module):
  def __init__(self, input_size, hidden_sizes, num_classes):
    super(BinarizedMLP, self).__init__()
    layers = []
    in_size = input_size

    for h in hidden_sizes:
      layers.append(BinarizedLinear(in_size, h))
      layers.append(nn.ReLU())
      in_size = h

    layers.append(BinarizedLinear(in_size, num_classes))
    layers.append(nn.LogSoftmax(dim=1))  # For multi-class classification

    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

# Custom collate function to extract features
def collate_fn(batch):
  waveforms = []
  labels = []
  for waveform, label in batch:
    waveform = waveform.squeeze(0)  # Remove channel dimension if it's 1
    features = extract_features(waveform)
    waveforms.append(features)
    labels.append(label)
  waveforms = torch.stack(waveforms)
  labels = torch.tensor(labels)
  return waveforms, labels

# Function to prepare DataLoader
def prepare_dataloaders(data_path, keywords, train_size, test_size, batch_size=32):
  full_dataset = SpeechCommandsDataset(
    data_path=data_path,
    keywords=keywords,
    transform=None  # Feature extraction is handled in collate_fn
  )

  if train_size + test_size > len(full_dataset):
    raise ValueError(f"train_size + test_size ({train_size + test_size}) exceeds dataset size ({len(full_dataset)})")

  indices = torch.randperm(len(full_dataset))
  train_indices = indices[:train_size]
  test_indices = indices[train_size:train_size + test_size]

  train_subset = Subset(full_dataset, train_indices)
  test_subset = Subset(full_dataset, test_indices)

  train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
  test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

  return train_loader, test_loader

# Function to save binarized weights
def save_binarized_weights(model, weights_dir):
  os.makedirs(weights_dir, exist_ok=True)
  for name, param in model.named_parameters():
    filename = f"{name.replace('.', '_')}.txt"
    filepath = os.path.join(weights_dir, filename)
    binarized_weights = param.data.cpu().numpy()
    np.savetxt(filepath, binarized_weights, delimiter=',')
    print(f"Saved binarized weights for '{name}' to '{filepath}'")

# Training function
def train(model, train_loader, criterion, optimizer, epoch):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

  for inputs, labels in loop:
    inputs, labels = inputs.to(device), labels.to(device)

    # Debug: Print input shape before view
    print(f"Input shape before view: {inputs.shape}")  # (batch_size, n_mfcc, time_steps)

    # Ensure inputs are reshaped correctly
    inputs = inputs.view(inputs.size(0), -1)  # (batch_size, input_size)

    # Debug: Print input shape after view
    print(f"Input shape after view: {inputs.shape}")  # (batch_size, input_size)

    optimizer.zero_grad()
    outputs = model(inputs)

    # Debug: Print output shape
    print(f"Output shape: {outputs.shape}")  # (batch_size, num_classes)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Statistics
    running_loss += loss.item() * inputs.size(0)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    # Update the progress bar with current loss and accuracy
    loop.set_postfix(loss=running_loss/total, accuracy=100*correct/total)

  epoch_loss = running_loss / total
  epoch_acc = 100 * correct / total
  print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%")

# Evaluation function
def evaluate(model, test_loader, criterion):
  model.eval()
  running_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating"):
      inputs, labels = inputs.to(device), labels.to(device)

      # Debug: Print input shape before view
      print(f"Input shape before view: {inputs.shape}")  # (batch_size, n_mfcc, time_steps)

      # Ensure inputs are reshaped correctly
      inputs = inputs.view(inputs.size(0), -1)  # (batch_size, input_size)

      # Debug: Print input shape after view
      print(f"Input shape after view: {inputs.shape}")  # (batch_size, input_size)

      outputs = model(inputs)

      # Debug: Print output shape
      print(f"Output shape: {outputs.shape}")  # (batch_size, num_classes)

      loss = criterion(outputs, labels)

      running_loss += loss.item() * inputs.size(0)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  epoch_loss = running_loss / total
  epoch_acc = 100 * correct / total
  print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.2f}%")
  return epoch_acc

# Main function to run training and evaluation
def main():
  train_loader, test_loader = prepare_dataloaders(DATA_PATH, keywords, TRAIN_SIZE, TEST_SIZE, BATCH_SIZE)

  # Initialize the model
  sample_waveform, _ = train_loader.dataset.dataset[train_loader.dataset.indices[0]]
  sample_features = extract_features(sample_waveform)
  input_size = sample_features.numel()
  print(f"Calculated input size: {input_size}")  # Should be n_mfcc * time_steps, e.g., 40 * 32 = 1280
  model = BinarizedMLP(input_size=input_size, hidden_sizes=[128, 64], num_classes=len(keywords)).to(device)

  # Define loss and optimizer
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

  best_accuracy = 0.0

  for epoch in range(EPOCHS):
    train(model, train_loader, criterion, optimizer, epoch)
    acc = evaluate(model, test_loader, criterion)

    # Save the best model
    if acc > best_accuracy:
      best_accuracy = acc
      save_binarized_weights(model, WEIGHTS_DIR)

  print(f"Best Test Accuracy: {best_accuracy:.2f}%")

if __name__ == '__main__':
  main()

