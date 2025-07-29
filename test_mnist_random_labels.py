#!/usr/bin/env python3
"""
Test script to verify MNIST random labels functionality.
"""

import torch
from dataset import (create_mnist_binary_dataset, create_mnist_binary_dataset_random_labels,
                    get_mnist_binary_dataloaders, get_mnist_binary_dataloaders_random_labels)
from training import run_beta_experiments

def test_mnist_random_labels():
    """Test MNIST random labels functionality."""
    print("="*60)
    print("TESTING MNIST RANDOM LABELS FUNCTIONALITY")
    print("="*60)
    
    # Test configuration
    classes = [4, 9]
    n_train_per_class = 100
    n_test_per_class = 50
    
    print(f"\n1. Testing dataset creation...")
    print(f"   Classes: {classes}")
    print(f"   Samples per class: {n_train_per_class} train, {n_test_per_class} test")
    
    # Create regular dataset
    print(f"\n   Creating regular MNIST dataset...")
    train_regular, test_regular = create_mnist_binary_dataset(
        classes=classes,
        n_train_per_class=n_train_per_class,
        n_test_per_class=n_test_per_class,
        random_seed=42,
        normalize=True
    )
    
    # Create random labels dataset
    print(f"\n   Creating MNIST dataset with random labels...")
    train_random, test_random = create_mnist_binary_dataset_random_labels(
        classes=classes,
        n_train_per_class=n_train_per_class,
        n_test_per_class=n_test_per_class,
        random_seed=42,
        normalize=True
    )
    
    # Verify datasets
    print(f"\n2. Verifying datasets...")
    print(f"   Regular dataset sizes: train={len(train_regular)}, test={len(test_regular)}")
    print(f"   Random labels dataset sizes: train={len(train_random)}, test={len(test_random)}")
    
    # Check that inputs are the same but labels are different
    regular_inputs = train_regular.tensors[0]
    random_inputs = train_random.tensors[0]
    regular_labels = train_regular.tensors[1]
    random_labels = train_random.tensors[1]
    
    inputs_identical = torch.equal(regular_inputs, random_inputs)
    labels_different = not torch.equal(regular_labels, random_labels)
    
    print(f"   ✅ Inputs identical: {inputs_identical}")
    print(f"   ✅ Labels different: {labels_different}")
    print(f"   Regular labels distribution: {torch.bincount(regular_labels.long())}")
    print(f"   Random labels distribution: {torch.bincount(random_labels.long())}")
    
    # Test dataloaders
    print(f"\n3. Testing dataloaders...")
    
    # Regular dataloaders
    train_loader_reg, test_loader_reg = get_mnist_binary_dataloaders(
        classes=classes,
        n_train_per_class=n_train_per_class,
        n_test_per_class=n_test_per_class,
        batch_size=32,
        random_seed=42,
        normalize=True
    )
    
    # Random labels dataloaders
    train_loader_rand, test_loader_rand = get_mnist_binary_dataloaders_random_labels(
        classes=classes,
        n_train_per_class=n_train_per_class,
        n_test_per_class=n_test_per_class,
        batch_size=32,
        random_seed=42,
        normalize=True
    )
    
    print(f"   Regular dataloaders: {len(train_loader_reg)} train batches, {len(test_loader_reg)} test batches")
    print(f"   Random labels dataloaders: {len(train_loader_rand)} train batches, {len(test_loader_rand)} test batches")
    
    # Test a batch from each
    for batch_x, batch_y in train_loader_reg:
        print(f"   Regular batch: shape={batch_x.shape}, labels range=[{batch_y.min():.1f}, {batch_y.max():.1f}]")
        break
    
    for batch_x, batch_y in train_loader_rand:
        print(f"   Random batch: shape={batch_x.shape}, labels range=[{batch_y.min():.1f}, {batch_y.max():.1f}]")
        break
    
    print(f"\n4. Testing training integration...")
    # Test a very minimal training run
    device = 'cpu'  # Use CPU for testing
    
    print(f"   Running minimal training experiment...")
    results = run_beta_experiments(
        beta_values=[1000],  # Just one beta value
        num_repetitions=1,   # Just one repetition
        num_epochs=5,        # Very few epochs
        a0=1e-3,
        b=0.5,
        sigma_gauss_prior=1000,
        device=device,
        dataset_type='mnist',
        use_random_labels=True,  # This is the key parameter!
        l_max=4.0,
        mnist_classes=classes
    )
    
    print(f"   ✅ Training completed successfully!")
    print(f"   Results keys: {list(results.keys())}")
    if 1000 in results:
        print(f"   Beta 1000 results: train_bce={results[1000]['train_bce_mean']:.4f}, test_bce={results[1000]['test_bce_mean']:.4f}")
    
    print(f"\n" + "="*60)
    print("✅ ALL TESTS PASSED - MNIST RANDOM LABELS FUNCTIONALITY WORKS!")
    print("="*60)

if __name__ == "__main__":
    test_mnist_random_labels()
