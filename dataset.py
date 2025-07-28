"""
Dataset creation module for SGLD experiments.

This module contains functions and classes for creating synthetic datasets
and MNIST binary classification datasets used in the SGLD generalization bound experiments.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple


def create_synth_dataset(
    n_train: int = 50,
    n_test: int = 100,
    input_dim: int = 4,
    random_seed: int = None
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Create the SYNTH dataset as described in the paper.
    
    The SYNTH dataset consists of:
    - 50 training data and 100 heldout data
    - Each input is a 4-dimensional vector sampled independently from a 
      zero-mean Gaussian distribution with an identity covariance matrix
    - The true classifier is linear
    - The norm of the separating hyperplane is sampled from a standard normal
    
    Args:
        n_train (int): Number of training samples (default: 50)
        n_test (int): Number of test samples (default: 100)
        input_dim (int): Dimensionality of input vectors (default: 4)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Tuple[TensorDataset, TensorDataset]: Training and test datasets
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Generate the separating hyperplane
    # Sample hyperplane direction uniformly from unit sphere
    w_direction = np.random.randn(input_dim)
    w_direction = w_direction / np.linalg.norm(w_direction)
    
    # Sample the norm from standard normal distribution
    w_norm = np.abs(np.random.randn())  # Taking absolute to ensure positive norm
    
    # Create the weight vector
    w = w_norm * w_direction
    
    # Generate training data
    X_train = np.random.randn(n_train, input_dim)  # Zero-mean Gaussian with identity covariance
    y_train_scores = X_train @ w  # Linear scores
    y_train = (y_train_scores > 0).astype(np.float32)  # Binary classification
    
    # Generate test data
    X_test = np.random.randn(n_test, input_dim)
    y_test_scores = X_test @ w
    y_test = (y_test_scores > 0).astype(np.float32)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, test_dataset


def get_synth_dataloaders(
    batch_size: int = 10,  # SYNTH uses batch size 10
    random_seed: int = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for the SYNTH dataset.
    
    Args:
        batch_size (int): Batch size for DataLoaders (default: 10 for SYNTH)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test DataLoaders
    """
    train_dataset, test_dataset = create_synth_dataset(random_seed=random_seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader


def create_synth_dataset_random_labels(
    n_train: int = 50,
    n_test: int = 100,
    input_dim: int = 4,
    random_seed: int = None
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Create the SYNTH dataset with random labels (no linear relationship).
    
    This dataset has the same structure as create_synth_dataset but with completely
    random labels, breaking the linear relationship between inputs and outputs.
    This is useful for testing generalization bounds in the presence of label noise
    or when there's no learnable pattern.
    
    The dataset consists of:
    - 50 training data and 100 heldout data
    - Each input is a 4-dimensional vector sampled independently from a 
      zero-mean Gaussian distribution with an identity covariance matrix
    - Labels are randomly assigned (50% probability for each class)
    - No linear separating hyperplane exists
    
    Args:
        n_train (int): Number of training samples (default: 50)
        n_test (int): Number of test samples (default: 100)
        input_dim (int): Dimensionality of input vectors (default: 4)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Tuple[TensorDataset, TensorDataset]: Training and test datasets with random labels
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Generate training data with same input distribution as original
    X_train = np.random.randn(n_train, input_dim)  # Zero-mean Gaussian with identity covariance
    # Generate completely random labels (50% probability for each class)
    y_train = np.random.randint(0, 2, size=n_train).astype(np.float32)
    
    # Generate test data with random labels
    X_test = np.random.randn(n_test, input_dim)
    y_test = np.random.randint(0, 2, size=n_test).astype(np.float32)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, test_dataset


def get_synth_dataloaders_random_labels(
    batch_size: int = 10,  # SYNTH uses batch size 10
    random_seed: int = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for the SYNTH dataset with random labels.
    
    Args:
        batch_size (int): Batch size for DataLoaders (default: 10 for SYNTH)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test DataLoaders with random labels
    """
    train_dataset, test_dataset = create_synth_dataset_random_labels(random_seed=random_seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader


def create_mnist_binary_dataset(
    classes=[0, 1],
    n_train_per_class: int = 1000,
    n_test_per_class: int = 500,
    random_seed: int = None,
    normalize: bool = True
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Create a binary MNIST dataset with only specified classes.
    
    Args:
        classes: List of two MNIST classes to use (default: [0, 1])
        n_train_per_class: Number of training samples per class
        n_test_per_class: Number of test samples per class
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Tuple[TensorDataset, TensorDataset]: Training and test datasets
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Define transforms
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    
    # Download MNIST datasets
    train_mnist = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_mnist = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Filter for only the specified classes
    def filter_classes(dataset, classes, n_per_class):
        data_list = []
        target_list = []
        class_counts = {cls: 0 for cls in classes}
        
        for data, target in dataset:
            if target in classes and class_counts[target] < n_per_class:
                data_list.append(data.flatten())  # Flatten 28x28 to 784
                # Convert to binary: class[0] -> 0, class[1] -> 1
                binary_target = 0 if target == classes[0] else 1
                target_list.append(binary_target)
                class_counts[target] += 1
                
                # Stop when we have enough samples of each class
                if all(count >= n_per_class for count in class_counts.values()):
                    break
        
        return torch.stack(data_list), torch.tensor(target_list, dtype=torch.float32)
    
    # Create filtered datasets
    train_data, train_targets = filter_classes(train_mnist, classes, n_train_per_class)
    test_data, test_targets = filter_classes(test_mnist, classes, n_test_per_class)
    
    # Shuffle the data
    if random_seed is not None:
        train_perm = torch.randperm(len(train_data))
        test_perm = torch.randperm(len(test_data))
        train_data, train_targets = train_data[train_perm], train_targets[train_perm]
        test_data, test_targets = test_data[test_perm], test_targets[test_perm]
    
    # Create TensorDatasets
    train_dataset = TensorDataset(train_data, train_targets)
    test_dataset = TensorDataset(test_data, test_targets)
    
    print(f"MNIST Binary Dataset Created:")
    print(f"  Classes: {classes[0]} (label=0) and {classes[1]} (label=1)")
    print(f"  Training samples: {len(train_dataset)} ({n_train_per_class} per class)")
    print(f"  Test samples: {len(test_dataset)} ({n_test_per_class} per class)")
    print(f"  Input dimension: {train_data.shape[1]} (flattened 28x28)")
    print(f"  Normalized: {normalize}")
    
    return train_dataset, test_dataset


def get_mnist_binary_dataloaders(
    classes=[0, 1],
    n_train_per_class: int = 1000,
    n_test_per_class: int = 500,
    batch_size: int = 32,
    random_seed: int = None,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for binary MNIST dataset.
    
    Args:
        classes: List of two MNIST classes to use
        n_train_per_class: Number of training samples per class
        n_test_per_class: Number of test samples per class
        batch_size: Batch size for data loaders
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize pixel values
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test data loaders
    """
    train_dataset, test_dataset = create_mnist_binary_dataset(
        classes=classes,
        n_train_per_class=n_train_per_class,
        n_test_per_class=n_test_per_class,
        random_seed=random_seed,
        normalize=normalize
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader
