#!/usr/bin/env python3

"""
Programma per addestrare una Multilayer Perceptron (MLP) per il Keyword Spotting utilizzando PyTorch.

- Carica il Google Speech Commands Dataset.
- Preprocessa gli audio in coefficienti MFCC.
- Definisce un modello MLP.
- Addestra il modello.
- Applica la quantizzazione post-addestramento.
- Esporta i pesi per l'uso su un MCU.

Prerequisiti:
- Python 3.6+
- PyTorch
- Torchaudio
- NumPy
- Scikit-learn
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Configurazione del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparametri
batch_size = 64
num_epochs = 20
learning_rate = 0.001

# Lista delle parole chiave da riconoscere
keywords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# Percorso al dataset
data_path = './speech_commands/'

# Classe per il Dataset
class SpeechCommandsDataset(Dataset):
    def __init__(self, data_path, keywords, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.keywords = keywords

        for idx, keyword in enumerate(keywords):
            keyword_path = os.path.join(data_path, keyword)
            if os.path.isdir(keyword_path):
                for file_name in os.listdir(keyword_path):
                    if file_name.endswith('.wav'):
                        self.data.append(os.path.join(keyword_path, file_name))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.data[idx])
        label = self.labels[idx]

        # Converti a mono se necessario
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ridimensiona l'audio a 1 secondo (16000 campioni)
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        waveform = torchaudio.transforms.PadTrim(max_len=16000)(waveform)

        # Estrai i coefficienti MFCC
        mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=20)(waveform)
        mfcc = mfcc.squeeze(0)  # Rimuovi la dimensione del canale

        # Flatten delle feature per l'input all'MLP
        mfcc = mfcc.view(-1)

        return mfcc, label

# Caricamento del Dataset
dataset = SpeechCommandsDataset(data_path, keywords)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

# Definizione del Modello MLP
class MLPKeywordSpotting(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPKeywordSpotting, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Ottieni la dimensione dell'input
sample_mfcc, _ = dataset[0]
input_size = sample_mfcc.shape[0]
num_classes = len(keywords)

# Inizializza il modello
model = MLPKeywordSpotting(input_size, num_classes).to(device)

# Definizione della funzione di perdita e dell'ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Funzione per addestrare il modello
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Azzeramento dei gradienti
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass e ottimizzazione
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Funzione per testare il modello
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (100 * correct / total)
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

# Addestramento del modello
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Test del modello
test_model(model, test_loader)

# Salvataggio del modello
torch.save(model.state_dict(), 'mlp_kws.pth')

# Quantizzazione post-addestramento
def quantize_model(model, test_loader):
    model.eval()
    model_int8 = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return model_int8

# Quantizza il modello
model_int8 = quantize_model(model, test_loader)

# Test del modello quantizzato
test_model(model_int8, test_loader)

# Esportazione dei pesi quantizzati per l'MCU
def export_weights(model, file_path):
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()

    # Salva i pesi in un file .npz
    np.savez_compressed(file_path, **weights)
    print(f'Weights exported to {file_path}')

export_weights(model_int8, 'mlp_kws_weights_quantized.npz')
