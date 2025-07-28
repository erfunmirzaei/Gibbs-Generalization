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


def initialize_kaiming_and_get_prior_sigma(model, activation='relu', mode='fan_in', verbose=True):
    """
    Initialize model with Kaiming initialization and compute appropriate prior sigma.
    
    Args:
        model: PyTorch model to initialize
        activation: Activation function ('relu', 'linear', etc.)
        mode: Fan mode ('fan_in', 'fan_out', 'fan_avg')
        verbose: Whether to print initialization details
        
    Returns:
        float: Computed sigma_prior that should be used for weight decay
    """
    sigma_values = []
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:  # Only for weight matrices
                # Apply Kaiming initialization
                fan_in = param.size(1)
                fan_out = param.size(0)
                
                if mode == 'fan_in':
                    fan = fan_in
                elif mode == 'fan_out':
                    fan = fan_out
                else:  # fan_avg
                    fan = (fan_in + fan_out) / 2
                
                # For ReLU: gain = sqrt(2), for other activations adjust accordingly
                gain = np.sqrt(2.0) if activation == 'relu' else 1.0
                std = gain / np.sqrt(fan)
                
                # Initialize the parameter
                param.normal_(0, std)
                sigma_values.append(std)
                
                if verbose:
                    print(f"Layer {name}: fan_in={fan_in}, fan_out={fan_out}, std={std:.6f}")
            elif 'bias' in name:
                # Initialize biases to zero
                param.zero_()
    
    # Use the average std as the prior sigma
    sigma_prior = np.mean(sigma_values) if sigma_values else 0.1
    
    if verbose:
        print(f"Computed sigma_prior from Kaiming init: {sigma_prior:.6f}")
        print(f"Corresponding weight_decay: {1/(2*sigma_prior**2):.6f}")
    
    return sigma_prior
