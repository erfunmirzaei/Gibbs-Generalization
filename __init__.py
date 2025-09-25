"""
Gibbs Generalization Bound Experiments

A comprehensive implementation of SGLD/ULA experiments for computing PAC-Bayesian
generalization bounds on MNIST and CIFAR-10 datasets.

Modules:
    - dataset: MNIST and CIFAR-10 dataset creation and data loading utilities
    - models: Neural network architectures (FCN, LeNet, VGG)
    - sgld: SGLD/ULA optimizer implementation
    - losses: Loss functions (BBCE, Savage, Tangent, Zero-one)
    - training: Training procedures and experiment orchestration
    - plot: Plotting, visualization, and PAC-Bayesian bound computation
    - main: Main experiment script
    - table_MNIST: MNIST results analysis and table generation
    - table_CIFAR: CIFAR-10 results analysis and table generation
"""

__version__ = "1.0.0"
__author__ = "Erfan Mirzaei"

# Import main components for easy access
try:
    from .dataset import (
        get_mnist_binary_dataloaders,
        get_cifar10_binary_dataloaders,
        get_mnist_binary_dataloaders_random_labels,
        get_cifar10_binary_dataloaders_random_labels
    )
    from .models import FCN1L, FCN2L, FCN3L, LeNet5, VGG16_CIFAR, initialize_nn_weights_gaussian
    from .sgld import SGLD
    from .losses import BoundedCrossEntropyLoss, ZeroOneLoss, TangentLoss, SavageLoss
    from .training import run_beta_experiments, save_moving_average_losses_to_csv
    from .plot import create_plots_from_csv, klbounds, invert_kl
except ImportError:
    # Handle case where package is imported from outside directory
    pass

__all__ = [
    # Dataset utilities
    'get_mnist_binary_dataloaders',
    'get_cifar10_binary_dataloaders', 
    'get_mnist_binary_dataloaders_random_labels',
    'get_cifar10_binary_dataloaders_random_labels',
    
    # Models and optimizers
    'FCN1L', 'FCN2L', 'FCN3L',
    'LeNet5', 'VGG16_CIFAR',
    'initialize_nn_weights_gaussian',
    'SGLD',
    
    # Loss functions
    'BoundedCrossEntropyLoss',
    'ZeroOneLoss', 
    'TangentLoss',
    'SavageLoss',
    
    # Training functions
    'run_beta_experiments',
    'save_moving_average_losses_to_csv',
    
    # Analysis and plotting
    'create_plots_from_csv',
    'klbounds',
    'invert_kl',
]
