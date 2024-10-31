#!/usr/bin/env python3

import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Keywords to recognize
keywords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# Path to the dataset
data_path = './speech_commands/'

# Custom Dataset class for inference
class SpeechCommandsTestDataset(Dataset):
  def __init__(self, data_path, keywords, transform=None):
    self.data = []
    self.labels = []
    self.transform = transform
    self.keywords = keywords
    self.classes = {word: idx for idx, word in enumerate(self.keywords)}
    
    for label in os.listdir(data_path):
      if label in self.keywords:
        files = os.listdir(os.path.join(data_path, label))
        for file in files:
          if file.endswith('.wav'):
            self.data.append(os.path.join(data_path, label, file))
            self.labels.append(self.classes[label])
    
  def __len__(self):
    return len(self.data)
    
  def __getitem__(self, idx):
    waveform, sample_rate = torchaudio.load(self.data[idx])
    # Preprocess waveform to have a fixed length as in training
    waveform = self._preprocess_audio(waveform, sample_rate)
    if self.transform:
      waveform = self.transform(waveform)
    label = self.labels[idx]
    return waveform, label

  def _preprocess_audio(self, waveform, sample_rate):
    # Define a fixed duration (e.g., 1 second)
    fixed_length = 1  # in seconds
    num_samples = int(fixed_length * sample_rate)

    if waveform.size(1) > num_samples:
      waveform = waveform[:, :num_samples]
    else:
      padding = num_samples - waveform.size(1)
      waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform

# Feature extraction function (MFCC) matching the training code
def extract_features(waveform):
  mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=40,
    melkwargs={
      'n_fft': 1024,
      'hop_length': 512,
      'n_mels': 40,
    }
  )
  mfcc = mfcc_transform(waveform)
  return mfcc

# MLP model definition matching the training code
class MLP(nn.Module):
  def __init__(self, input_size, hidden_sizes, num_classes):
    super(MLP, self).__init__()
    layers = []
    in_size = input_size

    for h in hidden_sizes:
      layers.append(nn.Linear(in_size, h))
      layers.append(nn.ReLU())
      in_size = h

    layers.append(nn.Linear(in_size, num_classes))
    layers.append(nn.LogSoftmax(dim=1))  # For multi-class classification

    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

# Prepare the test dataset and dataloader
# Create the test dataset
test_dataset = SpeechCommandsTestDataset(
  data_path=data_path,
  keywords=keywords,
  transform=None  # Feature extraction will be applied in the DataLoader
)

# Use a subset of the dataset for testing
test_size = 100  # Number of samples to test
test_dataset = torch.utils.data.Subset(test_dataset, range(test_size))

# Create the dataloader
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

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Initialize the model with the same architecture as in training
# Get the input size
sample_waveform, _ = test_dataset[0]
sample_features = extract_features(sample_waveform)
input_size = sample_features.numel()

# Number of classes is the number of keywords
num_classes = len(keywords)

# Define hidden layer sizes as in training
hidden_sizes = [128, 64]

model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, num_classes=num_classes)

# Function to load weights from .txt files
def load_weights_from_txt(model):
  for name, param in model.named_parameters():
    # Generate the filename based on the parameter name
    filename = f"{name.replace('.', '_')}.txt"
    if os.path.exists(filename):
      # Load the weights from the file
      weights = np.loadtxt(filename, delimiter=',')
      # Reshape the weights to match the parameter's shape
      weights = weights.reshape(param.shape)
      # Assign the weights to the parameter
      param.data = torch.from_numpy(weights).float()
      print(f"Loaded weights for '{name}' from '{filename}'")
    else:
      print(f"File '{filename}' not found. Unable to load weights for '{name}'.")

# Load the weights from the .txt files
load_weights_from_txt(model)

# Set the model to evaluation mode
model.eval()

# Mapping from class indices to labels
idx_to_class = {idx: word for idx, word in enumerate(keywords)}

# Perform inference
correct = 0
total = 0

with torch.no_grad():
  for inputs, labels in test_loader:
    # Flatten the input as expected by the model
    inputs = inputs.view(inputs.size(0), -1)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
    # Print the result for each sample
    actual_label = idx_to_class[labels.item()]
    predicted_label = idx_to_class[predicted.item()]
    print(f'Actual: {actual_label}, Predicted: {predicted_label}')

# Calculate and print overall accuracy
accuracy = 100 * correct / total
print(f'\nInference Accuracy on Test Data: {accuracy:.2f}%')

