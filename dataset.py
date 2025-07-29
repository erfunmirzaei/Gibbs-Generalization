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
    random_seed: int = None,
    num_workers: int = 2,  # Add multiprocessing
    pin_memory: bool = True  # Speed up GPU transfer
) -> Tuple[DataLoader, DataLoader]:
    """
    Create optimized DataLoaders for the SYNTH dataset.
    
    Args:
        batch_size (int): Batch size for DataLoaders (default: 10 for SYNTH)
        random_seed (int): Random seed for reproducibility
        num_workers (int): Number of worker processes for data loading
        pin_memory (bool): Pin memory for faster GPU transfer
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test DataLoaders
    """
    train_dataset, test_dataset = create_synth_dataset(random_seed=random_seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size * 2,  # Larger batch for evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
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
    classes=[[0], [1]],
    n_train_per_group: int = 1000,
    n_test_per_group: int = 500,
    random_seed: int = None,
    normalize: bool = True
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Create a binary MNIST dataset with specified class groups.
    
    This function supports both simple binary classification (two individual classes)
    and grouped binary classification (e.g., odd vs even digits).
    
    Args:
        classes: Can be either:
                - List of two individual classes: [0, 1] (backward compatibility)
                - List of two class groups: [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]] (even vs odd)
                - List of two class groups: [[0], [1]] (equivalent to [0, 1])
        n_train_per_group: Number of training samples per group (distributed among classes in the group)
        n_test_per_group: Number of test samples per group (distributed among classes in the group)
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize pixel values
        
    Returns:
        Tuple[TensorDataset, TensorDataset]: Training and test datasets
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Handle backward compatibility: convert [0, 1] to [[0], [1]]
    if len(classes) == 2 and isinstance(classes[0], int):
        classes = [[classes[0]], [classes[1]]]
    
    # Validate input
    if len(classes) != 2:
        raise ValueError("classes must contain exactly 2 groups")
    
    group_0_classes = classes[0] if isinstance(classes[0], list) else [classes[0]]
    group_1_classes = classes[1] if isinstance(classes[1], list) else [classes[1]]
    
    # Ensure no overlap between groups
    if set(group_0_classes) & set(group_1_classes):
        raise ValueError("Class groups cannot have overlapping classes")
    
    all_classes = group_0_classes + group_1_classes
    
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
    
    # Filter for the specified class groups
    def filter_grouped_classes(dataset, group_0_classes, group_1_classes, n_per_group):
        """Filter dataset to get samples from two class groups with binary labels."""
        data_list = []
        target_list = []
        
        # Count samples for each group
        group_0_count = 0
        group_1_count = 0
        
        # Calculate samples per class within each group
        n_per_class_g0 = max(1, n_per_group // len(group_0_classes))
        n_per_class_g1 = max(1, n_per_group // len(group_1_classes))
        
        # Track counts for each individual class
        class_counts = {}
        for cls in group_0_classes + group_1_classes:
            class_counts[cls] = 0
        
        for data, target in dataset:
            # Check if we have enough samples for both groups
            if group_0_count >= n_per_group and group_1_count >= n_per_group:
                break
                
            # Process group 0 classes (label = 0)
            if target in group_0_classes and group_0_count < n_per_group:
                target_n_per_class = n_per_class_g0
                # Allow some flexibility to reach target group size
                if group_0_count >= n_per_group - len(group_0_classes):
                    target_n_per_class = n_per_group  # Allow more samples to fill group
                    
                if class_counts[target] < target_n_per_class:
                    data_list.append(data.flatten())  # Flatten 28x28 to 784
                    target_list.append(0)  # Group 0 -> label 0
                    class_counts[target] += 1
                    group_0_count += 1
            
            # Process group 1 classes (label = 1)
            elif target in group_1_classes and group_1_count < n_per_group:
                target_n_per_class = n_per_class_g1
                # Allow some flexibility to reach target group size
                if group_1_count >= n_per_group - len(group_1_classes):
                    target_n_per_class = n_per_group  # Allow more samples to fill group
                    
                if class_counts[target] < target_n_per_class:
                    data_list.append(data.flatten())  # Flatten 28x28 to 784
                    target_list.append(1)  # Group 1 -> label 1
                    class_counts[target] += 1
                    group_1_count += 1
        
        return torch.stack(data_list), torch.tensor(target_list, dtype=torch.float32), class_counts
    
    # Create filtered datasets
    train_data, train_targets, train_class_counts = filter_grouped_classes(
        train_mnist, group_0_classes, group_1_classes, n_train_per_group
    )
    test_data, test_targets, test_class_counts = filter_grouped_classes(
        test_mnist, group_0_classes, group_1_classes, n_test_per_group
    )
    
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
    print(f"  Group 0 classes: {group_0_classes} (label=0)")
    print(f"  Group 1 classes: {group_1_classes} (label=1)")
    print(f"  Training samples: {len(train_dataset)} ({n_train_per_group} per group)")
    print(f"  Test samples: {len(test_dataset)} ({n_test_per_group} per group)")
    print(f"  Input dimension: {train_data.shape[1]} (flattened 28x28)")
    print(f"  Normalized: {normalize}")
    print(f"  Train class distribution: {train_class_counts}")
    print(f"  Test class distribution: {test_class_counts}")
    
    return train_dataset, test_dataset


def get_mnist_binary_dataloaders(
    classes=[[0], [1]],
    n_train_per_group: int = 1000,
    n_test_per_group: int = 500,
    batch_size: int = 128,  # Increase default batch size for better GPU utilization
    random_seed: int = None,
    normalize: bool = True,
    num_workers: int = 4,  # More workers for MNIST (larger dataset)
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create optimized data loaders for binary MNIST dataset.
    
    Args:
        classes: Can be either:
                - List of two individual classes: [0, 1] (backward compatibility)
                - List of two class groups: [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]] (even vs odd)
        n_train_per_group: Number of training samples per group
        n_test_per_group: Number of test samples per group
        batch_size: Batch size for data loaders (increased default for efficiency)
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize pixel values
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test data loaders
    """
    train_dataset, test_dataset = create_mnist_binary_dataset(
        classes=classes,
        n_train_per_group=n_train_per_group,
        n_test_per_group=n_test_per_group,
        random_seed=random_seed,
        normalize=normalize
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size * 2,  # Larger batch for evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    return train_loader, test_loader


def create_mnist_binary_dataset_random_labels(
    classes=[[0], [1]],
    n_train_per_group: int = 1000,
    n_test_per_group: int = 500,
    random_seed: int = None,
    normalize: bool = True
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Create a binary MNIST dataset with random labels (no relationship to actual digits).
    
    This function creates the same input data as create_mnist_binary_dataset but with
    completely random labels, breaking the relationship between image content and labels.
    This is useful for testing generalization bounds when there's no learnable pattern.
    
    Args:
        classes: Can be either:
                - List of two individual classes: [0, 1] (backward compatibility)
                - List of two class groups: [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]] (even vs odd)
        n_train_per_group: Number of training samples per group
        n_test_per_group: Number of test samples per group
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Tuple[TensorDataset, TensorDataset]: Training and test datasets with random labels
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # First create the regular MNIST dataset to get the images
    regular_train_dataset, regular_test_dataset = create_mnist_binary_dataset(
        classes=classes,
        n_train_per_group=n_train_per_group,
        n_test_per_group=n_test_per_group,
        random_seed=random_seed,
        normalize=normalize
    )
    
    # Extract the data (images) but replace labels with random ones
    train_data = regular_train_dataset.tensors[0]  # Get images
    test_data = regular_test_dataset.tensors[0]    # Get images
    
    # Generate completely random labels (50% probability for each class)
    train_random_labels = torch.randint(0, 2, (len(train_data),), dtype=torch.float32)
    test_random_labels = torch.randint(0, 2, (len(test_data),), dtype=torch.float32)
    
    # Create new datasets with random labels
    train_dataset_random = TensorDataset(train_data, train_random_labels)
    test_dataset_random = TensorDataset(test_data, test_random_labels)
    
    # Handle backward compatibility for printing
    if len(classes) == 2 and isinstance(classes[0], int):
        group_0_classes = [classes[0]]
        group_1_classes = [classes[1]]
    else:
        group_0_classes = classes[0]
        group_1_classes = classes[1]
    
    print(f"MNIST Binary Dataset with RANDOM LABELS Created:")
    print(f"  Original group 0 classes: {group_0_classes} (images only, labels randomized)")
    print(f"  Original group 1 classes: {group_1_classes} (images only, labels randomized)")
    print(f"  Training samples: {len(train_dataset_random)} ({n_train_per_group} per group)")
    print(f"  Test samples: {len(test_dataset_random)} ({n_test_per_group} per group)")
    print(f"  Input dimension: {train_data.shape[1]} (flattened 28x28)")
    print(f"  Normalized: {normalize}")
    print(f"  Labels: Completely random (50% each class)")
    
    return train_dataset_random, test_dataset_random


def get_mnist_binary_dataloaders_random_labels(
    classes=[[0], [1]],
    n_train_per_group: int = 1000,
    n_test_per_group: int = 500,
    batch_size: int = 128,
    random_seed: int = None,
    normalize: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for MNIST binary dataset with random labels.
    
    Args:
        classes: Can be either:
                - List of two individual classes: [0, 1] (backward compatibility)
                - List of two class groups: [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]] (even vs odd)
        n_train_per_group: Number of training samples per group
        n_test_per_group: Number of test samples per group
        batch_size: Batch size for DataLoaders
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize pixel values
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test DataLoaders with random labels
    """
    train_dataset, test_dataset = create_mnist_binary_dataset_random_labels(
        classes=classes,
        n_train_per_group=n_train_per_group,
        n_test_per_group=n_test_per_group,
        random_seed=random_seed,
        normalize=normalize
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # Larger batch for evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    return train_loader, test_loader
