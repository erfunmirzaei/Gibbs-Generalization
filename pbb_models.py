"""
PBB-compatible neural network models for comparison with the PAC-Bayes with Backprop paper.

This module implements the standard (non-Bayesian) NNet4l and CNNet4l architectures
from the PBB paper (https://github.com/mperezortiz/PBB) for MNIST classification.

The architectures and hyperparameters are designed to be identical to the original PBB
implementation for fair comparison of generalization bounds.

Architectures:
- NNet4l: 4-layer fully connected network designed for MNIST
  * Input: 784 (28x28 flattened)
  * Hidden layers: 600-600-600 units with ReLU
  * Output: 10 classes
  * Uses log_softmax activation (NLL loss compatible)

- CNNet4l: 4-layer convolutional network designed for MNIST
  * Input: 1 channel, 28x28
  * Conv layers: 1->32, 32->64 with kernel 3x3
  * Fully connected: 9216->128->10
  * Uses log_softmax activation (NLL loss compatible)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NNet4l(nn.Module):
    """
    4-layer Fully Connected Neural Network for MNIST (from PBB paper).
    
    This is a standard (non-probabilistic) version that matches the architecture
    used in the PAC-Bayes with Backprop paper by Perez-Ortiz et al. (2020).
    
    Architecture:
        - Input layer: 28x28 MNIST images (784 features)
        - Hidden layer 1: 600 units with ReLU activation
        - Hidden layer 2: 600 units with ReLU activation  
        - Hidden layer 3: 600 units with ReLU activation
        - Output layer: 10 units with log_softmax activation
    
    Args:
        num_classes (int): Number of output classes. Default: 10 (MNIST)
        dropout_prob (float): Dropout probability. Default: 0.0 (no dropout)
    
    Attributes:
        l1, l2, l3, l4 (nn.Linear): Fully connected layers
        dropout (nn.Dropout): Dropout layer (if dropout_prob > 0)
    
    Example:
        >>> model = NNet4l(num_classes=10, dropout_prob=0.2)
        >>> x = torch.randn(32, 784)  # Batch of 32 flattened MNIST images
        >>> output = model(x)  # Shape: (32, 10)
        >>> loss = F.nll_loss(output, targets)
    """
    
    def __init__(self, num_classes: int = 10, dropout_prob: float = 0.0):
        super(NNet4l, self).__init__()
        
        # Fully connected layers: 784 -> 600 -> 600 -> 600 -> num_classes
        self.l1 = nn.Linear(28 * 28, 600)
        self.l2 = nn.Linear(600, 600)
        self.l3 = nn.Linear(600, 600)
        self.l4 = nn.Linear(600, num_classes)
        
        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28) or
                            (batch_size, 784). Will be flattened automatically.
        
        Returns:
            torch.Tensor: Output log probabilities of shape (batch_size, num_classes).
                         Ready for use with F.nll_loss().
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Forward through layers with ReLU activation
        x = F.relu(self.l1(x))
        if self.dropout:
            x = self.dropout(x)
        
        x = F.relu(self.l2(x))
        if self.dropout:
            x = self.dropout(x)
        
        x = F.relu(self.l3(x))
        if self.dropout:
            x = self.dropout(x)
        
        # Output layer with log_softmax activation (for NLL loss)
        x = self.l4(x)
        return F.log_softmax(x, dim=1)


class CNNet4l(nn.Module):
    """
    4-layer Convolutional Neural Network for MNIST (from PBB paper).
    
    This is a standard (non-probabilistic) convolutional network that matches the
    architecture used in the PAC-Bayes with Backprop paper by Perez-Ortiz et al. (2020).
    
    Architecture:
        - Input: 1 channel, 28x28 MNIST images
        - Conv1: 1->32 channels, kernel 3x3, stride 1
        - Conv2: 32->64 channels, kernel 3x3, stride 1
        - Max pooling: 2x2
        - FC1: 9216 -> 128 with ReLU
        - FC2: 128 -> num_classes with log_softmax
    
    Args:
        num_classes (int): Number of output classes. Default: 10 (MNIST)
        dropout_prob (float): Dropout probability. Default: 0.0 (no dropout)
    
    Attributes:
        conv1, conv2 (nn.Conv2d): Convolutional layers
        fc1, fc2 (nn.Linear): Fully connected layers
        dropout (nn.Dropout2d): Dropout layer (if dropout_prob > 0)
    
    Example:
        >>> model = CNNet4l(num_classes=10, dropout_prob=0.2)
        >>> x = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
        >>> output = model(x)  # Shape: (32, 10)
        >>> loss = F.nll_loss(output, targets)
    """
    
    def __init__(self, num_classes: int = 10, dropout_prob: float = 0.0):
        super(CNNet4l, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        
        # Fully connected layers
        # After conv layers and max_pool2d: (28-3+1)//2 = 12, then (12-3+1)//2 = 5
        # So feature maps: 64 * 5 * 5 = 1600... but original PBB uses 9216
        # This suggests they may not be using padding or different kernel setup
        # Let me check: with no padding, stride 1: 28 -> 26 -> 13 -> 11 -> 5
        # 64 * 5 * 5 = 1600. But PBB says 9216. Let me use what matches:
        # 9216 = 64 * 144 = 64 * 12 * 12 (if input is 24x24 after pooling?)
        # Or maybe they pad differently. Let me use the exact number from PBB.
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout_prob = dropout_prob
        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)  # Use standard Dropout instead of Dropout2d
        else:
            self.dropout = None
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).
        
        Returns:
            torch.Tensor: Output log probabilities of shape (batch_size, num_classes).
                         Ready for use with F.nll_loss().
        """
        # Convolutional layers with dropout and ReLU
        x = self.conv1(x)
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(x)
        
        # Max pooling
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.fc1(x)
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(x)
        
        # Output layer with log_softmax activation
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Alias names for compatibility with different naming conventions
NNet4L = NNet4l
CNNet4L = CNNet4l


if __name__ == "__main__":
    # Test the architectures
    print("Testing NNet4l architecture...")
    model_fc = NNet4l(num_classes=10, dropout_prob=0.2)
    x_fc = torch.randn(32, 784)
    y_fc = model_fc(x_fc)
    print(f"  Input shape: {x_fc.shape}")
    print(f"  Output shape: {y_fc.shape}")
    print(f"  Output is log probabilities: {(y_fc.exp().sum(dim=1) - 1).abs().max().item() < 1e-5}")
    
    print("\nTesting CNNet4l architecture...")
    model_cnn = CNNet4l(num_classes=10, dropout_prob=0.2)
    x_cnn = torch.randn(32, 1, 28, 28)
    y_cnn = model_cnn(x_cnn)
    print(f"  Input shape: {x_cnn.shape}")
    print(f"  Output shape: {y_cnn.shape}")
    print(f"  Output is log probabilities: {(y_cnn.exp().sum(dim=1) - 1).abs().max().item() < 1e-5}")
