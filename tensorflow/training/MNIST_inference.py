#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to binarize tensors (not strictly necessary for inference)
def binarize(tensor):
    return tensor.sign()

# Neural Network definition (identical for inference)
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()

        # Using nn.Linear since weights are already binarized
        self.l1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.htanh1 = nn.Hardtanh()

        self.l2 = nn.Linear(hidden_size, 400)
        self.bn2 = nn.BatchNorm1d(400)
        self.htanh2 = nn.Hardtanh()

        self.l3 = nn.Linear(400, 300)
        self.bn3 = nn.BatchNorm1d(300)
        self.htanh3 = nn.Hardtanh()

        self.l4 = nn.Linear(300, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.bn1(out)
        out = self.htanh1(out)

        out = self.l2(out)
        out = self.bn2(out)
        out = self.htanh2(out)

        out = self.l3(out)
        out = self.bn3(out)
        out = self.htanh3(out)

        out = self.l4(out)
        return out

# Function to load binarized weights, biases, and batch norm parameters
def load_weights_and_bn_params(model, directory='binarized-weights'):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            # Paths for weights and biases files
            weight_file = os.path.join(directory, f'{name}_weights.txt')
            bias_file = os.path.join(directory, f'{name}_biases.txt')

            # Load binarized weights
            if os.path.exists(weight_file):
                weights = np.loadtxt(weight_file, dtype=int)    # Load as integers (-1, 1)
                weights = torch.tensor(weights, dtype=torch.float32)
                layer.weight.data = weights
                print(f'Layer {name} weights loaded from {weight_file}')

            # Load binarized biases
            if os.path.exists(bias_file):
                biases = np.loadtxt(bias_file, dtype=int)    # Load as integers (-1, 1)
                biases = torch.tensor(biases, dtype=torch.float32)
                layer.bias.data = biases
                print(f'Layer {name} biases loaded from {bias_file}')

        elif isinstance(layer, nn.BatchNorm1d):
            # Paths for batch norm parameters
            mean_file = os.path.join(directory, f'{name}_running_mean.txt')
            var_file = os.path.join(directory, f'{name}_running_var.txt')
            gamma_file = os.path.join(directory, f'{name}_gamma.txt')
            beta_file = os.path.join(directory, f'{name}_beta.txt')

            # Load running_mean
            if os.path.exists(mean_file):
                running_mean = np.loadtxt(mean_file)
                running_mean = torch.tensor(running_mean, dtype=torch.float32)
                layer.running_mean.data = running_mean
                print(f'BatchNorm Layer {name} running_mean loaded from {mean_file}')

            # Load running_var
            if os.path.exists(var_file):
                running_var = np.loadtxt(var_file)
                running_var = torch.tensor(running_var, dtype=torch.float32)
                layer.running_var.data = running_var
                print(f'BatchNorm Layer {name} running_var loaded from {var_file}')

            # Load gamma (weight)
            if os.path.exists(gamma_file):
                gamma = np.loadtxt(gamma_file)
                gamma = torch.tensor(gamma, dtype=torch.float32)
                layer.weight.data = gamma
                print(f'BatchNorm Layer {name} gamma loaded from {gamma_file}')

            # Load beta (bias)
            if os.path.exists(beta_file):
                beta = np.loadtxt(beta_file)
                beta = torch.tensor(beta, dtype=torch.float32)
                layer.bias.data = beta
                print(f'BatchNorm Layer {name} beta loaded from {beta_file}')

    print('All binarized weights, biases, and batch norm parameters have been loaded successfully.')

def print_batchnorm_params(model):
    """
    Prints the parameters of the Batch Normalization layers in the model.

    These meta-parameters are managed by PyTorch and are specific to the batch norm layers.
    They are used internally by the batch norm layers.

    Args:
        model (nn.Module): The PyTorch model to inspect.
    """
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm1d):
            print(f'\nBatchNorm Layer: {name}')
            print(f'Running Mean: {layer.running_mean}')
            print(f'Running Var: {layer.running_var}')
            print(f'Gamma (weight): {layer.weight}')
            print(f'Beta (bias): {layer.bias}')

# Definition of parameters
input_size = 784    # 28 x 28 pixels
hidden_size = 500
num_classes = 10

# Instantiate the model for inference
model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)

# Load binarized weights, biases, and batch norm parameters
load_weights_and_bn_params(model, directory='binarized-weights')

# Set the model to evaluation mode
model.eval()

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the test dataset
test_dataset = torchvision.datasets.MNIST(
    root='./MNIST-data/',
    train=False,
    transform=transform
)

# Data loader for testing
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=100,
    shuffle=False
)

# Function to perform inference and calculate accuracy
def evaluate_model(model, test_loader, device):
    model.eval()    # Ensure the model is in evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy} %')

# Perform evaluation
evaluate_model(model, test_loader, device)

# Print BatchNorm parameters
# print_batchnorm_params(model)
