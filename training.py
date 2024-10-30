import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Training parameters
batch_size = 64
learning_rate = 0.001
num_epochs = 20

# Keywords to recognize
keywords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# Path to the dataset
data_path = './speech_commands/'

# Custom Dataset class
class SpeechCommandsDataset(Dataset):
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
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        # Normalize waveform length to 1 second (16000 samples)
        waveform = self._normalize_waveform_length(waveform, target_length=16000)
        label = self.labels[idx]
        return waveform, label
    
    def _normalize_waveform_length(self, waveform, target_length):
        current_length = waveform.size(1)
        if current_length < target_length:
            # Pad with zeros
            pad_amount = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif current_length > target_length:
            # Truncate the waveform
            waveform = waveform[:, :target_length]
        return waveform

# Feature extraction function (MFCC)
def extract_features(waveform):
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,  # Ensure n_mfcc is set to 40
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

# Prepare the dataset and dataloader
# Create the dataset
dataset = SpeechCommandsDataset(
    data_path=data_path,
    keywords=keywords,
    transform=None  # Feature extraction will be applied in the DataLoader
)

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create the dataloaders
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Initialize the model, loss function, and optimizer
# Get the input size
sample_waveform, _ = dataset[0]
sample_features = extract_features(sample_waveform.squeeze(0))
input_size = sample_features.numel()

# Initialize the model
model = MLP(input_size=input_size, num_classes=len(keywords))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Flatten inputs
        inputs = inputs.view(inputs.size(0), -1)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Calculate average loss
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%\n')

# Save the model weights
torch.save(model.state_dict(), 'mlp_kws_weights.pth')
print('Weights saved to mlp_kws_weights.pth')
