#!/usr/bin/env python3

import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Definizione delle parole chiave
keywords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# Classe Dataset personalizzata
class SpeechCommandsDataset(Dataset):
  def __init__(self, root_dir, keywords, transform=None):
    self.root_dir = root_dir
    self.commands = keywords  # Utilizza solo le 10 parole chiave
    self.all_files = []
    self.labels = []
    self.transform = transform

    for idx, command in enumerate(self.commands):
      command_dir = os.path.join(root_dir, command)
      if os.path.isdir(command_dir):
        files = [f for f in os.listdir(command_dir) if f.endswith('.wav')]
        for f in files:
          self.all_files.append(os.path.join(command_dir, f))
          self.labels.append(idx)

  def __len__(self):
    return len(self.all_files)

  def __getitem__(self, idx):
    audio_path = self.all_files[idx]
    label = self.labels[idx]
    waveform, sample_rate = torchaudio.load(audio_path)

    # Preprocessing per standardizzare la durata degli audio
    waveform = self._preprocess_audio(waveform, sample_rate)

    if self.transform:
      features = self.transform(waveform)
    else:
      features = waveform

    return features, label

  def _preprocess_audio(self, waveform, sample_rate):
    # Durata fissa di 1 secondo
    fixed_length = 1  # in secondi
    num_samples = int(fixed_length * sample_rate)

    if waveform.size(1) > num_samples:
      waveform = waveform[:, :num_samples]
    else:
      padding = num_samples - waveform.size(1)
      waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform

# Trasformazione MFCC
mfcc_transform = T.MFCC(
  sample_rate=16000,
  n_mfcc=40,
  melkwargs={
    'n_fft': 1024,
    'hop_length': 512,
    'n_mels': 40,
  }
)

# Inizializzazione del dataset e DataLoader
dataset = SpeechCommandsDataset(
  root_dir='speech_commands',  # Sostituisci con il percorso corretto
  keywords=keywords,
  transform=mfcc_transform
)

# Salva la lista delle classi
with open('commands_list.txt', 'w') as f:
  for command in dataset.commands:
    f.write(command + '\n')

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Definizione del modello MLP
# Ottieni un campione per determinare la dimensione dell'input
sample_features, _ = dataset[0]
input_size = sample_features.numel()  # Numero totale di elementi
num_classes = len(dataset.commands)   # 10 classi

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
    layers.append(nn.LogSoftmax(dim=1))  # Per classificazione multi-classe

    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

model = MLP(input_size=input_size, hidden_sizes=[128, 64], num_classes=num_classes)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training del modello
num_epochs = 10

for epoch in range(num_epochs):
  total_loss = 0
  for features, labels in dataloader:
    # Appiattisci le caratteristiche
    inputs = features.view(features.size(0), -1)
    labels = labels.long()

    outputs = model(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  avg_loss = total_loss / len(dataloader)
  print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Salvataggio dei pesi del modello
torch.save(model.state_dict(), 'mlp_kws_weights.pth')

# Estrazione e salvataggio dei pesi per l'uso in C
weights = {}
for name, param in model.named_parameters():
  weights[name] = param.detach().numpy()

# Salva i pesi in file .txt
for name, weight in weights.items():
  filename = f"{name.replace('.', '_')}.txt"
  np.savetxt(filename, weight.flatten(), delimiter=',')
  print(f"Weights of layer '{name}' saved to '{filename}'")

