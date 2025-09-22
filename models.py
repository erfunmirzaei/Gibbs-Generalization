"""
Neural network models for the Gibbs generalization bound experiments.

This module contains the neural network architectures (SynthNN, MN 
used in the experiments. All models are GPU-ready.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN1L(nn.Module):
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
        super(FCN1L, self).__init__()
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

class FCN2L(nn.Module):
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
        super(FCN2L, self).__init__()
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

    
class FCN3L(nn.Module):
    """
    Three-layer neural network for MNIST binary classification.
    
    This network is designed for binary classification on MNIST data, specifically
    for distinguishing between classes 0 and 1. It uses a three hidden layer
    architecture with ReLU activations and produces raw logits for binary classification.
    
    Architecture:
        - Input layer: 784 features (28×28 flattened MNIST images)
        - First hidden layer: 50 units with ReLU activation (default)
        - Second hidden layer: 50 units with ReLU activation (default)
        - Third hidden layer: 50 units with ReLU activation (default)
        - Output layer: 1 unit (binary classification logits)
    
    Args:
        input_dim (int, optional): Number of input features. Defaults to 784 (28×28).
        hidden_dim (int, optional): Number of hidden units in each layer. Defaults to 50.
        output_dim (int, optional): Number of output units. Defaults to 1.
    
    Attributes:
        fc1 (nn.Linear): First fully connected layer (input to first hidden).
        fc2 (nn.Linear): Second fully connected layer (first hidden to second hidden).
        fc3 (nn.Linear): Third fully connected layer (second hidden to third hidden).
        fc4 (nn.Linear): Fourth fully connected layer (third hidden to output).

    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 50, output_dim: int = 1):
        super(FCN3L, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Hidden layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Hidden layer
        self.fc4 = nn.Linear(hidden_dim, output_dim)  # Single output for binary classification

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
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation, raw logits
        return x

    
class LeNet5(nn.Module):
    """
    LeNet-5 architecture for MNIST digit classification (10 classes).
    
    Architecture:
        - Input: grayscale images, size 28×28
        - (Optionally) pad or resize to 32×32, since original LeNet-5 expected 32×32.
        - C1: Conv2d(1 in channel, 6 out channels, kernel_size 5, stride 1, padding 0)
        - S2: AvgPool2d (or MaxPool2d) 2×2
        - C3: Conv2d(6, 16, kernel_size 5, stride 1, padding 0)
        - S4: AvgPool2d (2×2)
        - C5: Conv2d(16, 120, kernel_size 5, stride 1, padding 0) → gives 120 feature maps of size 1×1 (if input properly sized)
        - F6: Fully connected layer from 120 → 84
        - Output: Fully connected layer from 84 → 10 (for 10 digit classes)
        
    Notes:
        - Using ReLU activations
        - Could use MaxPool instead of the original average pooling for better performance in practice
        - If input is 28×28, may need to adjust padding or resize to 32×32 so that conv/pool layers give correct sizes
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super(LeNet5, self).__init__()
        
        # Convolution + pooling layers
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=0)  # C1
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # S2
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # C3
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # S4
        
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)  # C5
        
        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)  # F6
        self.fc2 = nn.Linear(84, num_classes)  # Output layer
    
    def forward(self, x):
        if x.dim() == 2:
            # If input is flattened (batch_size, 784), reshape to (batch_size, 1, 28, 28)
            x = x.view(x.size(0), 1, 28, 28)
        elif x.dim() == 3:
            # If input is (batch_size, 28, 28), add channel dimension
            x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, 28, 28)

        # If input is 28×28, we can pad it to 32×32
        # Alternative: resize before feeding. Here’s padding approach:
        if x.shape[-1] == 28:
            # pad 2 pixels on each side (left, right, top, bottom)
            x = F.pad(x, (2, 2, 2, 2))  # pad = (left, right, top, bottom)

        # Handle both flattened input (batch_size, 1024) and image input (batch_size, 1, 32, 32)
        # Layer1
        x = F.relu(self.conv1(x))  # → [batch, 6, 28, 28] if input was 32×32
        x = self.pool1(x)          # → [batch, 6, 14, 14]
        
        # Layer2
        x = F.relu(self.conv2(x))  # → [batch, 16, 10, 10]
        x = self.pool2(x)          # → [batch, 16, 5, 5]
        
        # Layer3 (C5)
        x = F.relu(self.conv3(x))  # → [batch, 120, 1, 1] if correct size
        
        # Flatten
        x = x.view(x.size(0), -1)   # → [batch, 120]
        
        # Fully connected
        x = F.relu(self.fc1(x))     # → [batch, 84]
        x = self.fc2(x)             # → [batch, num_classes] raw logits
        
        return x

class VGG16_CIFAR(nn.Module):
    """
    VGG-16 style network adapted for CIFAR-10 (32x32 RGB images).
    """

    def __init__(self, num_classes=10, batch_norm: bool = True):
        super(VGG16_CIFAR, self).__init__()
        self.batch_norm = batch_norm
        # Configuration for VGG-16: conv layers per block
        # Blocks: (2 convs) + pool, (2 convs) + pool, (3 convs) + pool, (3 convs) + pool, (3 convs) + pool
        # Number of channels: 64, 128, 256, 512, 512
        self.features = self._make_layers()

        # Classifier head
        # After conv/pool, feature map size is reduced; with CIFAR10 32×32 → after 5 pools of stride 2 you get 1x1 
        # (but often fewer pools are used; here I'll do 5 pools to match original VGG)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
            # Handle both flattened input (batch_size, 3072) and image input (batch_size, 3, 32, 32)
        if x.dim() == 2:
            # If input is flattened (batch_size, 3072), reshape to (batch_size, 3, 32, 32)
            x = x.view(x.size(0), 3, 32, 32)
        elif x.dim() == 3:
            # If input is (batch_size, 32, 32), add channel dimension
            x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 3, 32, 32)

        x = self.features(x)
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self):
        layers = []
        in_channels = 3
        cfg = [64, 64, 'M', 
               128, 128, 'M', 
               256, 256, 256, 'M', 
               512, 512, 512, 'M',
               512, 512, 512, 'M']
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    expansion = 1  # no expansion, output channels same as block's "planes"

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # first conv in block
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # second conv in block
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut (identity or downsampling)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # when changing dimension, use 1×1 conv to match shape
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # No pooling, as CIFAR-10 is small

        # Layers / stages
        self.layer1 = self._make_layer(block=BasicBlock, planes=16, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(block=BasicBlock, planes=32, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(block=BasicBlock, planes=64, num_blocks=3, stride=2)

        # Classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Create one stage (sequence of residual blocks).
        The first block in the stage may downsample (if stride != 1).
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)  # shape: (batch, 64, 1, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        
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