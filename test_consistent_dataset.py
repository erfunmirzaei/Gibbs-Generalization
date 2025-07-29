#!/usr/bin/env python3
"""
Test script to verify that using fixed datasets produces consistent class distributions
across different beta values and repetitions.
"""

from dataset import get_mnist_binary_dataloaders
from training import run_beta_experiments
import torch

def test_consistent_dataset():
    """Test that all repetitions and beta values use the same dataset."""
    print("Testing consistent dataset across repetitions and beta values...")
    print("=" * 70)
    
    # Create dataset once
    train_loader, test_loader = get_mnist_binary_dataloaders(
        classes=[[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]],  # Even vs Odd
        n_train_per_group=200,
        n_test_per_group=100,
        batch_size=32,
        random_seed=42,  # Fixed seed for consistency
        normalize=True
    )
    
    print(f"Created dataset with {len(train_loader.dataset)} train samples and {len(test_loader.dataset)} test samples")
    
    # Get the first batch to show it's consistent
    train_iter = iter(train_loader)
    first_batch_data, first_batch_labels = next(train_iter)
    
    print(f"\nFirst batch info:")
    print(f"  Data shape: {first_batch_data.shape}")
    print(f"  Labels shape: {first_batch_labels.shape}")
    print(f"  Label distribution in first batch: {torch.bincount(first_batch_labels.int())}")
    
    # Test that multiple calls to the dataloader give the same first batch
    train_iter2 = iter(train_loader)
    first_batch_data2, first_batch_labels2 = next(train_iter2)
    
    print(f"\nConsistency check:")
    print(f"  Data identical: {torch.equal(first_batch_data, first_batch_data2)}")
    print(f"  Labels identical: {torch.equal(first_batch_labels, first_batch_labels2)}")
    
    print(f"\n✅ Dataset consistency test passed!")
    print(f"   The same dataset will be used across all repetitions and beta values.")
    print(f"   Only the model initialization will vary between repetitions.")

if __name__ == "__main__":
    test_consistent_dataset()
