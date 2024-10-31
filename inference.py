#!/usr/bin/env python3

import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Carica le classi dal file salvato durante il training
with open('commands_list.txt', 'r') as f:
  keywords = [line.strip() for line in f]

# Path al dataset
data_path = './speech_commands/'

# Classe Dataset personalizzata per l'inferenza
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
    waveform = self._preprocess_audio(waveform, sample_rate)
    if self.transform:
      waveform = self.transform(waveform)
    label = self.labels[idx]
    return waveform, label

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

# Funzione per l'estrazione delle caratteristiche (MFCC)
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

# Definizione del modello MLP
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

# Preparazione del dataset di test e del DataLoader
test_dataset = SpeechCommandsTestDataset(
  data_path=data_path,
  keywords=keywords,
  transform=None  # L'estrazione delle caratteristiche verrà applicata nel DataLoader
)

# Mescola gli indici e seleziona un sottoinsieme casuale
test_size = 100  # Numero di campioni per il test
test_indices = torch.randperm(len(test_dataset))[:test_size]
test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

# Funzione di collate personalizzata
def collate_fn(batch):
  waveforms = []
  labels = []
  for waveform, label in batch:
    waveform = waveform.squeeze(0)  # Rimuovi la dimensione del canale se è 1
    features = extract_features(waveform)
    waveforms.append(features)
    labels.append(label)
  waveforms = torch.stack(waveforms)
  labels = torch.tensor(labels)
  return waveforms, labels

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Inizializzazione del modello
sample_waveform, _ = test_dataset[0]
sample_features = extract_features(sample_waveform)
input_size = sample_features.numel()

num_classes = len(keywords)  # 10 classi
hidden_sizes = [128, 64]

model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, num_classes=num_classes)

# Funzione per caricare i pesi dai file .txt
def load_weights_from_txt(model):
  for name, param in model.named_parameters():
    # Genera il nome del file basato sul nome del parametro
    filename = f"{name.replace('.', '_')}.txt"
    if os.path.exists(filename):
      # Carica i pesi dal file
      weights = np.loadtxt(filename, delimiter=',')
      # Rimodella i pesi per adattarsi alla forma del parametro
      weights = weights.reshape(param.shape)
      # Assegna i pesi al parametro
      param.data = torch.from_numpy(weights).float()
      print(f"Loaded weights for '{name}' from '{filename}'")
    else:
      print(f"File '{filename}' not found. Unable to load weights for '{name}'.")

# Carica i pesi dal file .txt
load_weights_from_txt(model)

# Imposta il modello in modalità valutazione
model.eval()

# Mappatura dagli indici alle classi
idx_to_class = {idx: word for idx, word in enumerate(keywords)}

# Esecuzione dell'inferenza
correct = 0
total = 0

with torch.no_grad():
  for inputs, labels in test_loader:
    # Appiattire l'input come richiesto dal modello
    inputs = inputs.view(inputs.size(0), -1)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
    # Stampa il risultato per ogni campione
    actual_label = idx_to_class[labels.item()]
    predicted_label = idx_to_class[predicted.item()]
    print(f'Actual: {actual_label}, Predicted: {predicted_label}')

# Calcola e stampa l'accuratezza complessiva
accuracy = 100 * correct / total
print(f'\nInference Accuracy on Test Data: {accuracy:.2f}%')

