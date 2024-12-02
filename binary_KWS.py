#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchaudio
import os
import numpy as np

# Configurazione del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparametri
num_classes = None  # Sarà determinato in base ai comandi da caricare
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Directory del dataset e dei comandi
dataset_dir = './speech-commands'
commands_file = './commands_list.txt'

# Caricamento dei comandi da considerare per l'addestramento
with open(commands_file, 'r') as f:
  commands = f.read().splitlines()

num_classes = len(commands)

# Creazione di un mapping tra comandi e classi
command_to_index = {command: idx for idx, command in enumerate(commands)}

# Definizione delle trasformazioni audio
sample_rate = 16000  # Frequenza di campionamento standard per il dataset
num_mel_bins = 64  # Numero di bande Mel per il Mel-Spectrogram

transform = torchaudio.transforms.MelSpectrogram(
  sample_rate=sample_rate,
  n_mels=num_mel_bins
)

# Funzione di binarizzazione modificata per evitare zeri
def binarize(tensor):
  return torch.where(tensor >= 0, torch.ones_like(tensor), -torch.ones_like(tensor))

# Classe Dataset personalizzata
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
        print(f'Attenzione: La cartella {command_dir} non esiste.')

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    filepath, label = self.samples[idx]
    waveform, sr = torchaudio.load(filepath)

    # Resample se necessario
    if sr != sample_rate:
      resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
      waveform = resampler(waveform)

    # Uniformare la durata a 1 secondo (16000 campioni)
    if waveform.size(1) < sample_rate:
      padding = sample_rate - waveform.size(1)
      waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
      waveform = waveform[:, :sample_rate]

    # Applicare la trasformazione
    if self.transform:
      features = self.transform(waveform)
      features = features.log2()  # Log-Mel Spectrogram
    else:
      features = waveform

    # Normalizzazione
    features = (features - features.mean()) / features.std()

    return features, label

# Suddivisione del dataset in training e test set
full_dataset = SpeechCommandsDataset(dataset_dir, commands, transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# Data loader
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

# Calcolo della dimensione dell'input
# Il Mel-Spectrogram avrà dimensioni: (batch_size, channels, n_mels, time_steps)
example_data, _ = next(iter(train_loader))
input_size = example_data.shape[1] * example_data.shape[2] * example_data.shape[3]  # Correzione qui

# Definizione delle funzioni per la binarizzazione
def binarize(tensor):
  return torch.where(tensor >= 0, torch.ones_like(tensor), -torch.ones_like(tensor))

# Classe BinarizeLinear con inizializzazione binarizzata
class BinarizeLinear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super(BinarizeLinear, self).__init__(in_features, out_features, bias)
    # Inizializzazione Xavier
    nn.init.xavier_uniform_(self.weight)
    if self.bias is not None:
      nn.init.constant_(self.bias, 0)
    
    # Binarizza i pesi e i bias dopo l'inizializzazione
    self.weight.data = binarize(self.weight.data)
    if self.bias is not None:
      self.bias.data = binarize(self.bias.data)
    
    # Conserva i pesi originali per l'aggiornamento
    if not hasattr(self.weight, 'org'):
      self.weight.org = self.weight.data.clone()
    if self.bias is not None and not hasattr(self.bias, 'org'):
      self.bias.org = self.bias.data.clone()

  def forward(self, input):
    input.data = binarize(input.data)
    self.weight.data = binarize(self.weight.org)
    if self.bias is not None:
      self.bias.data = binarize(self.bias.org)
    output = nn.functional.linear(input, self.weight, self.bias)
    return output

# Definizione della rete neurale
class NeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNetwork, self).__init__()

    self.l1 = BinarizeLinear(input_size, hidden_size)
    self.bn1 = nn.BatchNorm1d(hidden_size)
    self.htanh1 = nn.Hardtanh()

    self.l2 = BinarizeLinear(hidden_size, 400)
    self.bn2 = nn.BatchNorm1d(400)
    self.htanh2 = nn.Hardtanh()

    # **Nuovi Layer Aggiunti**
    self.l2_1 = BinarizeLinear(400, 350)
    self.bn2_1 = nn.BatchNorm1d(350)
    self.htanh2_1 = nn.Hardtanh()

    self.l2_2 = BinarizeLinear(350, 300)
    self.bn2_2 = nn.BatchNorm1d(300)
    self.htanh2_2 = nn.Hardtanh()
    # **Fine Nuovi Layer**

    self.l3 = BinarizeLinear(300, 300)
    self.bn3 = nn.BatchNorm1d(300)
    self.htanh3 = nn.Hardtanh()

    self.l4 = BinarizeLinear(300, num_classes)

  def forward(self, x):
    x = x.view(x.size(0), -1)  # Correzione qui
    out = self.l1(x)
    out = self.bn1(out)
    out = self.htanh1(out)

    out = self.l2(out)
    out = self.bn2(out)
    out = self.htanh2(out)

    # **Forward dei Nuovi Layer**
    out = self.l2_1(out)
    out = self.bn2_1(out)
    out = self.htanh2_1(out)

    out = self.l2_2(out)
    out = self.bn2_2(out)
    out = self.htanh2_2(out)
    # **Fine Forward dei Nuovi Layer**

    out = self.l3(out)
    out = self.bn3(out)
    out = self.htanh3(out)

    out = self.l4(out)
    return out

# Iperparametri del modello
hidden_size = 500  # Puoi modificarlo se necessario
model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)

# Funzione di perdita e ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Addestramento del modello
model.train()

for epoch in range(num_epochs):
  for i, (features, labels) in enumerate(train_loader):
    features = features.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(features)
    loss = criterion(outputs, labels)

    # Backward e ottimizzazione
    optimizer.zero_grad()
    loss.backward()

    # Straight Through Estimator
    for p in list(model.parameters()):
      if hasattr(p, 'org'):
        p.data.copy_(p.org)

    optimizer.step()

    # Clamping dei pesi e dei bias
    for p in list(model.parameters()):
      if hasattr(p, 'org'):
        p.org.copy_(p.data.clamp_(-1, 1))

    if (i + 1) % 10 == 0:
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
          .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# Funzione per salvare i pesi e i parametri di batch norm
def save_weights_and_bn_params(model, directory='binarized-weights-audio'):
  if not os.path.exists(directory):
    os.makedirs(directory)

  for name, layer in model.named_children():
    if isinstance(layer, BinarizeLinear):
      weight_file = os.path.join(directory, f'{name}_weights.txt')
      bias_file = os.path.join(directory, f'{name}_biases.txt')

      if hasattr(layer, 'weight'):
        weights = binarize(layer.weight.org).detach().cpu().numpy()
        fmt = '%d'
        np.savetxt(weight_file, weights, fmt=fmt)

      if hasattr(layer, 'bias') and layer.bias is not None:
        biases = binarize(layer.bias.org).detach().cpu().numpy()
        fmt = '%d'
        np.savetxt(bias_file, biases, fmt=fmt)

    elif isinstance(layer, nn.BatchNorm1d):
      mean_file = os.path.join(directory, f'{name}_running_mean.txt')
      var_file = os.path.join(directory, f'{name}_running_var.txt')
      gamma_file = os.path.join(directory, f'{name}_gamma.txt')
      beta_file = os.path.join(directory, f'{name}_beta.txt')

      running_mean = layer.running_mean.detach().cpu().numpy()
      running_var = layer.running_var.detach().cpu().numpy()
      gamma = layer.weight.detach().cpu().numpy()
      beta = layer.bias.detach().cpu().numpy()

      np.savetxt(mean_file, running_mean)
      np.savetxt(var_file, running_var)
      np.savetxt(gamma_file, gamma)
      np.savetxt(beta_file, beta)

  print('Pesi, bias e parametri di batch norm salvati nella cartella:', directory)

# Valutazione del modello
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

  print('Accuratezza del modello sul test set: {:.2f}%'
      .format(100 * correct / total))

# Salvataggio dei pesi e dei parametri di batch norm
# save_weights_and_bn_params(model)
