"""
Neural network models for the Gibbs generalization bound experiments.

This module contains the neural network architectures (SynthNN, MNISTNN)
used in the experiments.
"""
import torch.nn as nn
import torch.nn.functional as F


class SynthNN(nn.Module):
    """
    Neural network for the SYNTH dataset following SGLD specifications.
    Architecture: 1 hidden layer with 100 units, input=4, output=1 (binary classification)
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 100):
        super(SynthNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Single output for binary classification
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation, raw logits
        return x


class MNISTNN(nn.Module):
    """
    Neural network for MNIST following SGLD specifications.
    Architecture: 3 layers with 600 units each, input=784, output=10
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 600, num_classes: int = 10):
        super(MNISTNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten for MNIST
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
