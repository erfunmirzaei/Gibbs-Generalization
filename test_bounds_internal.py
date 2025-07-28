#!/usr/bin/env python3
"""
Test script to verify that the bounds computation now handles beta=0 internally.
"""

import numpy as np
from training import run_beta_experiments
from bounds import compute_generalization_bound, compute_generalization_errors

def test_bounds_internal_beta_zero():
    """Test that bounds computation handles beta=0 internally."""
    print("Testing bounds computation with internal beta=0 handling...")
    
    # Test with beta values that don't include 0
    original_beta_values = [0.1, 0.2, 0.5]
    print(f"Original beta values: {original_beta_values}")
    
    # Run experiments (this will still add beta=0 to results)
    results = run_beta_experiments(
        beta_values=original_beta_values,
        num_repetitions=3,  # Small number for testing
        num_epochs=100,     # Small number for testing
        a0=0.1,
        device='cpu'
    )
    
    print(f"Results contain beta values: {sorted(results.keys())}")
    
    # Now test that bounds computation works with just the original beta values
    print("\nTesting bounds computation with original beta values only...")
    
    try:
        # This should work - bounds function handles beta=0 internally
        n_train = 50  # Default SYNTH dataset size
        bounds = compute_generalization_bound(original_beta_values, results, n_train, loss_type='bce')
        print(f"✓ BCE bounds computed successfully for betas: {sorted(bounds.keys())}")
        
        zero_one_bounds = compute_generalization_bound(original_beta_values, results, n_train, loss_type='zero_one')
        print(f"✓ Zero-one bounds computed successfully for betas: {sorted(zero_one_bounds.keys())}")
        
        gen_errors = compute_generalization_errors(original_beta_values, results)
        print(f"✓ Generalization errors computed successfully for betas: {sorted(gen_errors.keys())}")
        
        # Verify bounds are reasonable
        for beta in original_beta_values:
            bce_bound = bounds[beta]['generalization_bound']
            bce_empirical = bounds[beta]['empirical_loss']
            print(f"  Beta {beta}: BCE bound = {bce_bound:.4f}, empirical loss = {bce_empirical:.4f}")
            
            if np.isnan(bce_bound) or np.isnan(bce_empirical):
                print(f"  ✗ WARNING: NaN values for beta {beta}")
            elif bce_bound < 0:
                print(f"  ✗ WARNING: Negative bound for beta {beta}")
            else:
                print(f"  ✓ Valid bounds for beta {beta}")
        
    except Exception as e:
        print(f"✗ Error in bounds computation: {e}")
        return False
    
    return True

def test_bounds_without_beta_zero():
    """Test that bounds computation fails gracefully when beta=0 is not in results."""
    print("\nTesting bounds computation without beta=0 in results...")
    
    # Create fake results without beta=0
    fake_results = {
        0.1: {
            'train_bce_mean': 0.5,
            'test_bce_mean': 0.6,
            'train_bce_std': 0.05,
            'test_bce_std': 0.06,
            'train_01_mean': 0.3,
            'test_01_mean': 0.4,
            'train_01_std': 0.03,
            'test_01_std': 0.04,
            'raw_train_bce': [0.5, 0.5, 0.5],
            'raw_train_01': [0.3, 0.3, 0.3]
        },
        0.2: {
            'train_bce_mean': 0.45,
            'test_bce_mean': 0.58,
            'train_bce_std': 0.05,
            'test_bce_std': 0.06,
            'train_01_mean': 0.28,
            'test_01_mean': 0.38,
            'train_01_std': 0.03,
            'test_01_std': 0.04,
            'raw_train_bce': [0.45, 0.45, 0.45],
            'raw_train_01': [0.28, 0.28, 0.28]
        }
    }
    
    try:
        n_train = 50  # Test training set size
        bounds = compute_generalization_bound([0.1, 0.2], fake_results, n_train, loss_type='bce')
        print("✗ Expected error but bounds computation succeeded")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
        return True
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Testing bounds computation with internal beta=0 handling")
    print("="*60)
    
    test1_passed = test_bounds_internal_beta_zero()
    test2_passed = test_bounds_without_beta_zero()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Test 1 (bounds with beta=0 in results): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Test 2 (bounds without beta=0 in results): {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
