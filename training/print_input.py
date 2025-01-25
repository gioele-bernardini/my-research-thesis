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
    n_mfcc=13,
    sample_rate=sample_rate,
    melkwargs=melkwargs,
    log_mels=True,
)

def binarize(tensor):
    # Returns the sign of the tensor (standard binarization)
    return tensor.sign()

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
        if self.transform:
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

# Definition of BinarizeLinear with STE
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

# Simplified Neural Network without multiple BinarizeLinear definitions
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

def save_weights_16bit(model, directory='weights_16bit'):
    import os
    import numpy as np
    import torch.nn as nn

    if not os.path.exists(directory):
        os.makedirs(directory)

    for name, layer in model.named_modules():
        # Save weights of Linear or BinarizeLinear layers
        if hasattr(layer, 'weight') and layer.weight is not None:
            is_binarized = isinstance(layer, (BinarizeLinearSTE))

            if is_binarized:
                # Binarized weights
                weight_org = layer.weight.data.clone()
                w_bin = binarize(weight_org).detach().cpu().numpy()  # values in {-1,+1}
                w_bin = (w_bin > 0).astype(np.uint8)  # convert to {0,1}
                packed_w = np.packbits(w_bin.flatten())

                weight_file = os.path.join(directory, f'{name}_weights_binarized.bin')
                with open(weight_file, 'wb') as f:
                    f.write(packed_w.tobytes())

                # Binarized bias (if present)
                if layer.bias is not None:
                    b_bin = binarize(layer.bias.data.clone()).detach().cpu().numpy()
                    b_bin = (b_bin > 0).astype(np.uint8)
                    packed_b = np.packbits(b_bin.flatten())

                    bias_file = os.path.join(directory, f'{name}_biases_binarized.bin')
                    with open(bias_file, 'wb') as f:
                        f.write(packed_b.tobytes())

            else:
                # Float weights in float16 format
                w_float16 = layer.weight.data.detach().cpu().numpy().astype(np.float16)
                weight_file = os.path.join(directory, f'{name}_weights_float16.bin')
                w_float16.tofile(weight_file)

                # Float bias16 (if present)
                if layer.bias is not None:
                    b_float16 = layer.bias.data.detach().cpu().numpy().astype(np.float16)
                    bias_file = os.path.join(directory, f'{name}_biases_float16.bin')
                    b_float16.tofile(bias_file)

        # Save BatchNorm1d parameters in float16
        if isinstance(layer, nn.BatchNorm1d):
            # Running mean
            running_mean = layer.running_mean.detach().cpu().numpy().astype(np.float16)
            mean_file = os.path.join(directory, f'{name}_running_mean_float16.bin')
            running_mean.tofile(mean_file)

            # Running variance
            running_var = layer.running_var.detach().cpu().numpy().astype(np.float16)
            var_file = os.path.join(directory, f'{name}_running_variance_float16.bin')
            running_var.tofile(var_file)

            # Scale (gamma)
            if layer.weight is not None:
                weight_scale = layer.weight.data.detach().cpu().numpy().astype(np.float16)
                scale_file = os.path.join(directory, f'{name}_scale_float16.bin')
                weight_scale.tofile(scale_file)

            # Shift (beta)
            if layer.bias is not None:
                bias_shift = layer.bias.data.detach().cpu().numpy().astype(np.float16)
                shift_file = os.path.join(directory, f'{name}_shift_float16.bin')
                bias_shift.tofile(shift_file)

    print(f"Weights and BN parameters saved in '{directory}'.")

# Model hyperparameters
hidden_size1 = 256
hidden_size2 = 128
hidden_size3 = 128

# Initialize the model
model = NeuralNetworkSimplified(input_size=1 * num_mel_bins * ((sample_rate // melkwargs['hop_length']) + 1), 
                                hidden_size1=hidden_size1, 
                                hidden_size2=hidden_size2, 
                                hidden_size3=hidden_size3, 
                                num_classes=num_classes).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ==================== Inspection of Raw Input ====================

# Create a dataset without transformations to get raw waveform
raw_dataset = SpeechCommandsDataset(dataset_dir, commands, transform=None)

# Get the first example
raw_waveform, label, filepath = raw_dataset[0]

print("===== Primo Esempio Prima del Preprocessing =====")
print(f"Percorso File: {filepath}")
print(f"Etichetta: {label} ({commands[label]})")
print(f"Forma del Waveform: {raw_waveform.shape}")
print(f"Dati del Waveform (prime 10 samples): {raw_waveform[:, :10]}")  # Stampa delle prime 10 samples per brevità

# Se vuoi visualizzare il waveform, puoi usare matplotlib (opzionale)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 4))
# plt.plot(raw_waveform.t().numpy())
# plt.title(f"Waveform di '{commands[label]}'")
# plt.xlabel("Indice del Campione")
# plt.ylabel("Ampiezza")
# plt.show()

# ==================== Fine Ispezione ====================

# Per prevenire che lo script continui con il training dopo l'ispezione, esci qui.
# Rimuovi o commenta la seguente riga se vuoi procedere con il training dopo l'ispezione.
exit()

# ==================== Training del Modello ====================
# Uncomment the following code block to enable training.

# model.train()

# for epoch in range(num_epochs):
#     correct, total = 0, 0
#     for i, (features, labels, fp) in enumerate(train_loader):
#         features = features.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         outputs = model(features, fp)
#         loss = criterion(outputs, labels)

#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#         # Backward and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%')

# # Evaluating the model
# model.eval()

# with torch.no_grad():
#     correct = 0
#     total = 0

#     for features, labels, fp in test_loader:
#         features = features.to(device)
#         labels = labels.to(device)

#         # Passa entrambi gli argomenti richiesti dal modello
#         outputs = model(features, fp)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('\nAccuratezza del modello sul set di test: {:.2f}%'
#       .format(100 * correct / total))

# # Salva i pesi
# save_weights_16bit(model)
