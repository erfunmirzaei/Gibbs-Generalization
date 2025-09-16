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
    
    This is a simple feedforward neural network designed for binary classification
    on synthetic 2D datasets. The architecture consists of a single hidden layer
    with ReLU activation, followed by a linear output layer that produces raw logits.
    
    Architecture:
        - Input layer: 2 features (default)
        - Hidden layer: 50 units with ReLU activation (default)
        - Output layer: 1 unit (binary classification logits)
    
    Args:
        input_dim (int, optional): Number of input features. Defaults to 2.
        hidden_dim (int, optional): Number of hidden units. Defaults to 50.
        output_dim (int, optional): Number of output units. Defaults to 1.
    
    Attributes:
        fc1 (nn.Linear): First fully connected layer (input to hidden).
        fc2 (nn.Linear): Second fully connected layer (hidden to output).
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 50, output_dim: int = 1):
        super(SynthNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Single output for binary classification
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation, raw logits
        return x


class MNISTNN1L(nn.Module):
    """
    Single-layer neural network for MNIST binary classification.
    
    This network is designed for binary classification on MNIST data, specifically
    for distinguishing between classes 0 and 1. It uses a single hidden layer
    architecture with ReLU activation and produces raw logits for binary classification.
    
    Architecture:
        - Input layer: 784 features (28×28 flattened MNIST images)
        - Hidden layer: 50 units with ReLU activation (default)
        - Output layer: 1 unit (binary classification logits)
    
    Args:
        input_dim (int, optional): Number of input features. Defaults to 784 (28×28).
        hidden_dim (int, optional): Number of hidden units. Defaults to 50.
        output_dim (int, optional): Number of output units. Defaults to 1.
    
    Attributes:
        fc1 (nn.Linear): First fully connected layer (input to hidden).
        fc2 (nn.Linear): Second fully connected layer (hidden to output).
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 50, output_dim: int = 1):
        super(MNISTNN1L, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Single output for binary classification
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 28, 28) or 
                            (batch_size, 784). Will be flattened automatically.
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        x = x.view(x.size(0), -1)  # Flatten for MNIST (batch_size, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation, raw logits
        return x

class MNISTNN2L(nn.Module):
    """
    Two-layer neural network for MNIST binary classification.
    
    This network is designed for binary classification on MNIST data, specifically
    for distinguishing between classes 0 and 1. It uses a two hidden layer
    architecture with ReLU activations and produces raw logits for binary classification.
    
    Architecture:
        - Input layer: 784 features (28×28 flattened MNIST images)
        - First hidden layer: 50 units with ReLU activation (default)
        - Second hidden layer: 50 units with ReLU activation (default)
        - Output layer: 1 unit (binary classification logits)
    
    Args:
        input_dim (int, optional): Number of input features. Defaults to 784 (28×28).
        hidden_dim (int, optional): Number of hidden units in each layer. Defaults to 50.
        output_dim (int, optional): Number of output units. Defaults to 1.
    
    Attributes:
        fc1 (nn.Linear): First fully connected layer (input to first hidden).
        fc2 (nn.Linear): Second fully connected layer (first hidden to second hidden).
        fc3 (nn.Linear): Third fully connected layer (second hidden to output).
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 50, output_dim: int = 1):
        super(MNISTNN2L, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Single output for binary classification

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 28, 28) or 
                            (batch_size, 784). Will be flattened automatically.
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        x = x.view(x.size(0), -1)  # Flatten for MNIST (batch_size, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation, raw logits
        return x


def initialize_nn_weights_gaussian(model: nn.Module, sigma: float, seed: int = 42) -> nn.Module:
    """
    Initialize neural network weights from a Gaussian distribution.
    
    This function initializes all trainable parameters in the given neural network
    model using a Gaussian (normal) distribution with mean 0 and specified standard
    deviation. This is commonly used for weight initialization in Bayesian neural
    networks and SGLD/HMC sampling procedures.
    
    Args:
        model (nn.Module): The neural network model to initialize. All parameters
                          with requires_grad=True will be initialized.
        sigma (float): Standard deviation of the Gaussian distribution used for
                      weight initialization. Must be positive.
        seed (int, optional): Random seed for reproducible initialization. 
                             Defaults to 42.
    
    Returns:
        nn.Module: The model with initialized weights (modified in-place and returned).
    
    Example:
        >>> model = SynthNN()
        >>> initialized_model = initialize_nn_weights_gaussian(model, sigma=0.1)
        >>> # All weights are now initialized from N(0, 0.1²)
    
    Note:
        This function modifies the model in-place and also returns it for convenience.
        Only parameters with requires_grad=True are initialized.
    """
    for param in model.parameters():
        if param.requires_grad:
            nn.init.normal_(param.data, mean=0.0, std=sigma, generator=torch.Generator().manual_seed(seed))
    return model