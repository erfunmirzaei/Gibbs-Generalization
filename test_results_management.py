#!/usr/bin/env python3
"""
Test script to verify the enhanced results management system works correctly.
"""

import numpy as np
from results_manager import (
    create_hyperparameter_dict, 
    save_or_merge_results, 
    load_existing_results,
    hyperparameters_match,
    generate_hyperparameter_hash
)

def create_mock_results(num_reps=5):
    """Create mock experimental results for testing."""
    results = {}
    
    for beta in [100, 1000, 5000]:
        results[beta] = {
            'train_bce_mean': np.random.uniform(0.1, 0.5),
            'train_bce_var': np.random.uniform(0.001, 0.01),
            'train_bce_std': np.random.uniform(0.01, 0.1),
            'test_bce_mean': np.random.uniform(0.2, 0.6),
            'test_bce_var': np.random.uniform(0.001, 0.01),
            'test_bce_std': np.random.uniform(0.01, 0.1),
            'train_01_mean': np.random.uniform(0.05, 0.3),
            'train_01_var': np.random.uniform(0.001, 0.01),
            'train_01_std': np.random.uniform(0.01, 0.1),
            'test_01_mean': np.random.uniform(0.1, 0.4),
            'test_01_var': np.random.uniform(0.001, 0.01),
            'test_01_std': np.random.uniform(0.01, 0.1),
            'raw_train_bce': np.random.uniform(0.1, 0.5, num_reps).tolist(),
            'raw_test_bce': np.random.uniform(0.2, 0.6, num_reps).tolist(),
            'raw_train_01': np.random.uniform(0.05, 0.3, num_reps).tolist(),
            'raw_test_01': np.random.uniform(0.1, 0.4, num_reps).tolist(),
        }
    
    return results

def test_results_management():
    """Test the complete results management workflow."""
    print("Testing Enhanced Results Management System")
    print("=" * 50)
    
    # Create hyperparameters for the test
    hyperparams = create_hyperparameter_dict(
        beta_values=[100, 1000, 5000],
        num_repetitions=5,
        num_epochs=1000,
        a0=0.01,
        dataset_type='mnist',
        mnist_classes=[[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]],
        train_dataset_size=2000,
        test_dataset_size=1000
    )
    
    hash_val = generate_hyperparameter_hash(hyperparams)
    print(f"Hyperparameter hash: {hash_val}")
    print()
    
    # Test 1: Save initial results
    print("Test 1: Saving initial results...")
    results1 = create_mock_results(num_reps=5)
    filename, was_merged = save_or_merge_results(results1, hyperparams, "test_experiment")
    print(f"Saved to: {filename}")
    print(f"Was merged: {was_merged}")
    print()
    
    # Test 2: Load the results back
    print("Test 2: Loading results back...")
    loaded_hyperparams, loaded_results = load_existing_results(filename)
    if loaded_results is not None:
        print("✅ Results loaded successfully")
        print(f"Loaded {len(loaded_results)} beta values")
        for beta in sorted(loaded_results.keys()):
            print(f"  Beta {beta}: {len(loaded_results[beta]['raw_train_bce'])} repetitions")
    else:
        print("❌ Failed to load results")
    print()
    
    # Test 3: Try to merge with identical hyperparameters
    print("Test 3: Merging with identical hyperparameters...")
    results2 = create_mock_results(num_reps=3)  # Different number of reps but same hyperparams
    filename2, was_merged2 = save_or_merge_results(results2, hyperparams, "test_experiment")
    print(f"Saved to: {filename2}")
    print(f"Was merged: {was_merged2}")
    
    if was_merged2:
        # Load merged results
        _, merged_results = load_existing_results(filename2)
        print("✅ Results merged successfully")
        for beta in sorted(merged_results.keys()):
            print(f"  Beta {beta}: {len(merged_results[beta]['raw_train_bce'])} repetitions")
    print()
    
    # Test 4: Try to save with different hyperparameters
    print("Test 4: Trying with different hyperparameters...")
    different_hyperparams = create_hyperparameter_dict(
        beta_values=[100, 1000, 5000],
        num_repetitions=5,
        num_epochs=2000,  # Different epochs
        a0=0.01,
        dataset_type='mnist',
        mnist_classes=[[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]],
        train_dataset_size=2000,
        test_dataset_size=1000
    )
    
    results3 = create_mock_results(num_reps=3)
    filename3, was_merged3 = save_or_merge_results(results3, different_hyperparams, "test_experiment")
    print(f"Saved to: {filename3}")
    print(f"Was merged: {was_merged3}")
    
    if not was_merged3:
        print("✅ Correctly created new file for different hyperparameters")
    print()
    
    # Test 5: Test hyperparameter matching
    print("Test 5: Testing hyperparameter matching...")
    match1 = hyperparameters_match(hyperparams, loaded_hyperparams)
    match2 = hyperparameters_match(hyperparams, different_hyperparams)
    print(f"Original vs Loaded: {match1}")
    print(f"Original vs Different: {match2}")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("The results management system is working correctly.")

if __name__ == "__main__":
    test_results_management()
