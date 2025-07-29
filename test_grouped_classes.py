#!/usr/bin/env python3
"""
Test script to verify MNIST binary dataset creation with grouped classes.
This tests the new functionality for binary classification with grouped classes
(e.g., odd vs even digits).
"""

import numpy as np
import torch
from dataset import create_mnist_binary_dataset, get_mnist_binary_dataloaders
from dataset import create_mnist_binary_dataset_random_labels, get_mnist_binary_dataloaders_random_labels

def test_backward_compatibility():
    """Test that the old API still works."""
    print("=== Testing Backward Compatibility ===")
    
    # Test with old-style individual classes
    train_dataset, test_dataset = create_mnist_binary_dataset(
        classes=[0, 1],
        n_train_per_group=200,
        n_test_per_group=100,
        random_seed=42,
        normalize=True
    )
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    print(f"Input shape: {train_dataset[0][0].shape}")
    print()

def test_grouped_classes():
    """Test the new grouped classes functionality."""
    print("=== Testing Grouped Classes (Odd vs Even) ===")
    
    # Define odd vs even groups
    even_digits = [0, 2, 4, 6, 8]
    odd_digits = [1, 3, 5, 7, 9]
    
    train_dataset, test_dataset = create_mnist_binary_dataset(
        classes=[even_digits, odd_digits],
        n_train_per_group=500,
        n_test_per_group=200,
        random_seed=42,
        normalize=True
    )
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    print(f"Input shape: {train_dataset[0][0].shape}")
    
    # Check label distribution
    train_labels = [int(train_dataset[i][1].item()) for i in range(len(train_dataset))]
    test_labels = [int(test_dataset[i][1].item()) for i in range(len(test_dataset))]
    
    print(f"Train label distribution: {np.bincount(train_labels)}")
    print(f"Test label distribution: {np.bincount(test_labels)}")
    print()

def test_grouped_classes_random_labels():
    """Test grouped classes with random labels."""
    print("=== Testing Grouped Classes with Random Labels ===")
    
    # Define odd vs even groups
    even_digits = [0, 2, 4, 6, 8]
    odd_digits = [1, 3, 5, 7, 9]
    
    train_dataset, test_dataset = create_mnist_binary_dataset_random_labels(
        classes=[even_digits, odd_digits],
        n_train_per_group=300,
        n_test_per_group=150,
        random_seed=42,
        normalize=True
    )
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    print(f"Input shape: {train_dataset[0][0].shape}")
    
    # Check label distribution (should be roughly 50/50 due to randomness)
    train_labels = [int(train_dataset[i][1].item()) for i in range(len(train_dataset))]
    test_labels = [int(test_dataset[i][1].item()) for i in range(len(test_dataset))]
    
    print(f"Train label distribution: {np.bincount(train_labels)}")
    print(f"Test label distribution: {np.bincount(test_labels)}")
    print()

def test_dataloader_functionality():
    """Test that the dataloaders work with grouped classes."""
    print("=== Testing DataLoader Functionality ===")
    
    # Test with a different grouping: low digits vs high digits
    low_digits = [0, 1, 2, 3, 4]
    high_digits = [5, 6, 7, 8, 9]
    
    train_loader, test_loader = get_mnist_binary_dataloaders(
        classes=[low_digits, high_digits],
        n_train_per_group=100,
        n_test_per_group=50,
        batch_size=32,
        random_seed=42,
        normalize=True
    )
    
    # Test that we can iterate through the dataloaders
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))
    
    print(f"Train batch - Data shape: {train_batch[0].shape}, Labels shape: {train_batch[1].shape}")
    print(f"Test batch - Data shape: {test_batch[0].shape}, Labels shape: {test_batch[1].shape}")
    
    # Check label distribution in first batch
    print(f"Train batch label distribution: {np.bincount(train_batch[1].int().numpy())}")
    print(f"Test batch label distribution: {np.bincount(test_batch[1].int().numpy())}")
    print()

def test_mixed_groups():
    """Test with groups of different sizes."""
    print("=== Testing Mixed Group Sizes ===")
    
    # Group with different numbers of classes
    small_group = [0]
    large_group = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    train_dataset, test_dataset = create_mnist_binary_dataset(
        classes=[small_group, large_group],
        n_train_per_group=200,
        n_test_per_group=100,
        random_seed=42,
        normalize=True
    )
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Check label distribution
    train_labels = [int(train_dataset[i][1].item()) for i in range(len(train_dataset))]
    test_labels = [int(test_dataset[i][1].item()) for i in range(len(test_dataset))]
    
    print(f"Train label distribution: {np.bincount(train_labels)}")
    print(f"Test label distribution: {np.bincount(test_labels)}")
    print()

if __name__ == "__main__":
    print("Testing MNIST Binary Dataset with Grouped Classes")
    print("=" * 60)
    
    test_backward_compatibility()
    test_grouped_classes()
    test_grouped_classes_random_labels()
    test_dataloader_functionality()
    test_mixed_groups()
    
    print("All tests completed successfully!")
