#!/usr/bin/env python3

import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
    if self.transform:
      waveform = self.transform(waveform)
    label = self.labels[idx]
    return waveform, label

# Feature extraction function (MFCC)
def extract_features(waveform):
  mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=40,
    log_mels=True
  )
  mfcc = mfcc_transform(waveform)
  return mfcc

# MLP model definition
class MLP(nn.Module):
  def __init__(self, input_size, num_classes):
    super(MLP, self).__init__()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(input_size, 512)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(512, 256)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(256, num_classes)
  
  def forward(self, x):
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.fc3(x)
    return x

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

# Initialize the model
# Get the input size
sample_waveform, _ = test_dataset[0]
sample_features = extract_features(sample_waveform.squeeze(0))
input_size = sample_features.numel()

model = MLP(input_size=input_size, num_classes=len(keywords))

# Load the saved model weights
model.load_state_dict(torch.load('mlp_kws_weights.pth'))

# Set the model to evaluation mode
model.eval()

# Mapping from class indices to labels
idx_to_class = {idx: word for word, idx in model.classes.items()}

# Perform inference
correct = 0
total = 0

with torch.no_grad():
  for inputs, labels in test_loader:
    inputs = inputs.view(inputs.size(0), -1)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
    # Print the result for each sample
    print(f'Actual: {keywords[labels.item()]}, Predicted: {keywords[predicted.item()]}')

# Calculate and print overall accuracy
accuracy = 100 * correct / total
print(f'\nInference Accuracy on Test Data: {accuracy:.2f}%')

