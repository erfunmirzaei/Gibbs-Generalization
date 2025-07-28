"""
Dataset creation module for SGLD experiments.

This module contains functions and classes for creating synthetic datasets
used in the SGLD generalization bound experiments.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
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
