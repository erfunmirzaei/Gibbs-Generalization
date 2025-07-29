#!/usr/bin/env python3
"""
Demonstration script showing different ways to use MNIST binary classification
with grouped classes.
"""

from dataset import create_mnist_binary_dataset, get_mnist_binary_dataloaders
import numpy as np

def demo_simple_binary():
    """Demo: Simple binary classification (backward compatible)."""
    print("=== Simple Binary Classification (0 vs 1) ===")
    
    train_dataset, test_dataset = create_mnist_binary_dataset(
        classes=[0, 1],  # Will be converted to [[0], [1]]
        n_train_per_group=100,
        n_test_per_group=50,
        random_seed=42
    )
    print()

def demo_odd_vs_even():
    """Demo: Odd vs Even digits."""
    print("=== Odd vs Even Digits ===")
    
    even_digits = [0, 2, 4, 6, 8]
    odd_digits = [1, 3, 5, 7, 9]
    
    train_dataset, test_dataset = create_mnist_binary_dataset(
        classes=[even_digits, odd_digits],
        n_train_per_group=200,
        n_test_per_group=100,
        random_seed=42
    )
    
    # Check that we actually get balanced labels
    train_labels = [int(train_dataset[i][1].item()) for i in range(len(train_dataset))]
    print(f"Label distribution: {np.bincount(train_labels)} (even=0, odd=1)")
    print()

def demo_low_vs_high():
    """Demo: Low digits (0-4) vs High digits (5-9)."""
    print("=== Low vs High Digits ===")
    
    low_digits = [0, 1, 2, 3, 4]
    high_digits = [5, 6, 7, 8, 9]
    
    train_dataset, test_dataset = create_mnist_binary_dataset(
        classes=[low_digits, high_digits],
        n_train_per_group=150,
        n_test_per_group=75,
        random_seed=42
    )
    print()

def demo_custom_groups():
    """Demo: Custom groupings."""
    print("=== Custom Groups: {0,1,2} vs {7,8,9} ===")
    
    group_a = [0, 1, 2]
    group_b = [7, 8, 9]
    
    train_dataset, test_dataset = create_mnist_binary_dataset(
        classes=[group_a, group_b],
        n_train_per_group=120,
        n_test_per_group=60,
        random_seed=42
    )
    print()

def demo_single_vs_multiple():
    """Demo: Single digit vs multiple digits."""
    print("=== Single Digit (0) vs Multiple Digits (1,2,3,4,5) ===")
    
    single_digit = [0]
    multiple_digits = [1, 2, 3, 4, 5]
    
    train_dataset, test_dataset = create_mnist_binary_dataset(
        classes=[single_digit, multiple_digits],
        n_train_per_group=100,
        n_test_per_group=50,
        random_seed=42
    )
    print()

def demo_dataloader_usage():
    """Demo: Using with DataLoaders."""
    print("=== DataLoader Usage (Even vs Odd) ===")
    
    even_digits = [0, 2, 4, 6, 8]
    odd_digits = [1, 3, 5, 7, 9]
    
    train_loader, test_loader = get_mnist_binary_dataloaders(
        classes=[even_digits, odd_digits],
        n_train_per_group=200,
        n_test_per_group=100,
        batch_size=32,
        random_seed=42
    )
    
    # Demonstrate usage
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx+1}: Data shape {data.shape}, Target shape {target.shape}")
        print(f"Target distribution in batch: {np.bincount(target.int().numpy())}")
        if batch_idx >= 2:  # Show only first 3 batches
            break
    print()

if __name__ == "__main__":
    print("MNIST Binary Classification with Grouped Classes - Demonstrations")
    print("=" * 70)
    print()
    
    demo_simple_binary()
    demo_odd_vs_even()
    demo_low_vs_high()
    demo_custom_groups()
    demo_single_vs_multiple()
    demo_dataloader_usage()
    
    print("=" * 70)
    print("All demonstrations completed!")
    print("\nUsage Summary:")
    print("- Use [class1, class2] for simple binary classification")
    print("- Use [[group1_classes], [group2_classes]] for grouped classification")
    print("- Examples:")
    print("  * Even vs Odd: [[0,2,4,6,8], [1,3,5,7,9]]")
    print("  * Low vs High: [[0,1,2,3,4], [5,6,7,8,9]]")
    print("  * Custom: [[0,1], [8,9]]")
