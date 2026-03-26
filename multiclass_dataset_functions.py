"""
Multi-class MNIST dataset functions.

This module contains functions for creating multi-class MNIST datasets,
extending the binary classification functions to support any number of classes.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, List


def create_mnist_multiclass_dataset(
    classes: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    n_train_per_class: int = 1000,
    n_test_per_class: int = 500,
    random_seed: int = None,
    normalize: bool = True
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Create a multi-class MNIST dataset with specified classes.
    
    This function supports multi-class classification with any number of classes
    from the MNIST dataset.
    
    Args:
        classes: List of class indices to include (0-9)
        n_train_per_class: Number of training samples per class
        n_test_per_class: Number of test samples per class
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize pixel values
        
    Returns:
        Tuple[TensorDataset, TensorDataset]: Training and test datasets
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Validate input
    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError("classes must be a non-empty list")
    if any(c not in range(10) for c in classes):
        raise ValueError("All classes must be between 0 and 9")
    if len(classes) != len(set(classes)):
        raise ValueError("classes must not contain duplicates")
    
    # Define transforms
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    
    # Download MNIST datasets with error handling
    import os
    import shutil
    
    data_root = './data'
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            train_mnist = torchvision.datasets.MNIST(
                root=data_root, train=True, download=True, transform=transform
            )
            test_mnist = torchvision.datasets.MNIST(
                root=data_root, train=False, download=True, transform=transform
            )
            # Test that we can access the data
            _ = len(train_mnist)
            _ = len(test_mnist)
            break
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: MNIST download/loading failed: {e}")
            if attempt < max_retries - 1:
                # Remove potentially corrupted data and retry
                mnist_path = os.path.join(data_root, 'MNIST')
                if os.path.exists(mnist_path):
                    print(f"Removing corrupted MNIST data at {mnist_path}")
                    shutil.rmtree(mnist_path)
                print("Retrying download...")
            else:
                raise RuntimeError(f"Failed to download MNIST data after {max_retries} attempts: {e}")
    
    # Filter for the specified classes
    def filter_multiclass(dataset, target_classes, n_per_class):
        """Filter dataset to get samples from specified classes with original labels."""
        data_list = []
        target_list = []
        
        # Track counts for each class
        class_counts = {cls: 0 for cls in target_classes}
        
        for data, target in dataset:
            # Check if we have enough samples for all classes
            if all(class_counts[cls] >= n_per_class for cls in target_classes):
                break
                
            # Process if target class is in our desired classes
            if target in target_classes and class_counts[target] < n_per_class:
                data_list.append(data.flatten())  # Flatten 28x28 to 784
                target_list.append(target)  # Keep original label
                class_counts[target] += 1
        
        return torch.stack(data_list), torch.tensor(target_list, dtype=torch.long), class_counts
    
    # Create filtered datasets
    train_data, train_targets, train_class_counts = filter_multiclass(
        train_mnist, classes, n_train_per_class
    )
    test_data, test_targets, test_class_counts = filter_multiclass(
        test_mnist, classes, n_test_per_class
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
    
    print(f"MNIST Multi-class Dataset Created:")
    print(f"  Classes: {classes}")
    print(f"  Training samples: {len(train_dataset)} ({n_train_per_class} per class)")
    print(f"  Test samples: {len(test_dataset)} ({n_test_per_class} per class)")
    print(f"  Input dimension: {train_data.shape[1]} (flattened 28x28)")
    print(f"  Normalized: {normalize}")
    print(f"  Train class distribution: {train_class_counts}")
    print(f"  Test class distribution: {test_class_counts}")
    
    return train_dataset, test_dataset


def get_mnist_multiclass_dataloaders(
    classes: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    n_train_per_class: int = 1000,
    n_test_per_class: int = 500,
    batch_size: int = 128,
    random_seed: int = None,
    normalize: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for MNIST multi-class dataset.
    
    Args:
        classes: List of class indices to include (0-9)
        n_train_per_class: Number of training samples per class
        n_test_per_class: Number of test samples per class
        batch_size: Batch size for DataLoaders
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize pixel values
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test DataLoaders
    """
    train_dataset, test_dataset = create_mnist_multiclass_dataset(
        classes=classes,
        n_train_per_class=n_train_per_class,
        n_test_per_class=n_test_per_class,
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


def create_mnist_multiclass_dataset_partial_random_labels(
    classes: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    p: float = 0.5,
    n_train_per_class: int = 1000,
    n_test_per_class: int = 500,
    random_seed: int = None,
    normalize: bool = True
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Create a multi-class MNIST dataset with partial random labels.
    
    This function creates the same input data as create_mnist_multiclass_dataset but with
    partially random labels, breaking the relationship between image content and labels.
    This is useful for testing generalization bounds when there's little learnable pattern.
    
    Args:
        classes: List of class indices to include (0-9)
        p: The percentage of labels to randomize (0.5 means 50% of labels are random, 50% are original)
        n_train_per_class: Number of training samples per class
        n_test_per_class: Number of test samples per class
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Tuple[TensorDataset, TensorDataset]: Training and test datasets with partial random labels
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # First create the regular MNIST dataset to get the images
    regular_train_dataset, regular_test_dataset = create_mnist_multiclass_dataset(
        classes=classes,
        n_train_per_class=n_train_per_class,
        n_test_per_class=n_test_per_class,
        random_seed=random_seed,
        normalize=normalize
    )
    
    # Extract the data (images) but replace labels with random ones
    train_data = regular_train_dataset.tensors[0]  # Get images
    test_data = regular_test_dataset.tensors[0]    # Get images
    
    # Get original labels
    train_original_labels = regular_train_dataset.tensors[1]
    test_original_labels = regular_test_dataset.tensors[1]

    # Flip labels with probability p
    n_classes = len(classes)
    train_flip_mask = torch.rand(len(train_data)) < p
    test_flip_mask = torch.rand(len(test_data)) < p
    train_final_labels = torch.where(
        train_flip_mask, 
        torch.randint(0, n_classes, train_original_labels.shape), 
        train_original_labels
    )
    test_final_labels = torch.where(
        test_flip_mask, 
        torch.randint(0, n_classes, test_original_labels.shape), 
        test_original_labels
    )

    # Create new datasets with random labels
    train_dataset_random = TensorDataset(train_data, train_final_labels)
    test_dataset_random = TensorDataset(test_data, test_final_labels)

    print(f"MNIST Multi-class Dataset with PARTIAL RANDOM LABELS Created:")
    print(f"  Classes: {classes}")
    print(f"  Training samples: {len(train_dataset_random)} ({n_train_per_class} per class)")
    print(f"  Test samples: {len(test_dataset_random)} ({n_test_per_class} per class)")
    print(f"  Input dimension: {train_data.shape[1]} (flattened 28x28)")
    print(f"  Normalized: {normalize}")
    print(f"  Randomization probability p: {p}")
    print(f"  Number of classes: {n_classes}")
    
    return train_dataset_random, test_dataset_random


def get_mnist_multiclass_dataloaders_partial_random_labels(
    classes: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    p: float = 0.5,
    n_train_per_class: int = 1000,
    n_test_per_class: int = 500,
    batch_size: int = 128,
    random_seed: int = None,
    normalize: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for MNIST multi-class dataset with partial random labels.
    
    Args:
        classes: List of class indices to include (0-9)
        p: The percentage of labels to randomize (0.5 means 50% of labels are random, 50% are original)
        n_train_per_class: Number of training samples per class
        n_test_per_class: Number of test samples per class
        batch_size: Batch size for DataLoaders
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize pixel values
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test DataLoaders with random labels
    """
    train_dataset, test_dataset = create_mnist_multiclass_dataset_partial_random_labels(
        classes=classes,
        p=p,
        n_train_per_class=n_train_per_class,
        n_test_per_class=n_test_per_class,
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
