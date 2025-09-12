"""
Neural network models for the Gibbs generalization bound experiments.

This module contains the neural network architectures (SynthNN, MNISTNN)
used in the experiments. All models are GPU-ready.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SynthNN(nn.Module):
    """
    Neural network for the SYNTH dataset following SGLD specifications.
    Architecture: 1 hidden layer with 50 units, input=2, output=1 (binary classification)
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 50, output_dim: int = 1):
        super(SynthNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Single output for binary classification
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation, raw logits
        return x


class MNISTNN(nn.Module):
    """
    Neural network for MNIST binary classification (classes 0 and 1 only).
    Architecture: 1 hidden layer with 50 units, input=784, output=1 (binary classification)
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 50, output_dim: int = 1):
        super(MNISTNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Single output for binary classification
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten for MNIST (batch_size, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation, raw logits
        return x


def initialize_nn_weights_gaussian(model: nn.Module, sigma: float, seed: int = 42) -> nn.Module:
    """
    Initialize neural network weights from a Gaussian distribution with mean 0 and std sigma.
    
    Args:
        model (nn.Module): The neural network model to initialize.
        sigma (float): Standard deviation of the Gaussian distribution.
    """
    for param in model.parameters():
        if param.requires_grad:
            nn.init.normal_(param.data, mean=0.0, std=sigma, generator=torch.Generator().manual_seed(seed))
    return model