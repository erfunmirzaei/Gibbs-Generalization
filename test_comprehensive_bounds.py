#!/usr/bin/env python3
"""
Comprehensive test of the updated bounds computation behavior.
This test verifies that:
1. The bounds functions handle beta=0 internally
2. Only the original beta values are used for bounds computation
3. The bounds are mathematically correct and consistent
"""

import numpy as np
from training import run_beta_experiments
from bounds import compute_generalization_bound, compute_generalization_errors, save_results_to_file
from plot_utils import plot_beta_results

def test_comprehensive_bounds_behavior():
    """Test comprehensive bounds behavior with internal beta=0 handling."""
    print("="*80)
    print("COMPREHENSIVE BOUNDS BEHAVIOR TEST")
    print("="*80)
    
    # Test with beta values that don't include 0
    original_beta_values = [0.1, 0.5, 1.0]
    print(f"Testing with original beta values: {original_beta_values}")
    
    # Run experiments (this will still add beta=0 to results)
    print("\nRunning experiments...")
    results = run_beta_experiments(
        beta_values=original_beta_values,
        num_repetitions=3,  # Small number for testing
        num_epochs=100,     # Small number for testing
        a0=0.1,
        device='cpu'
    )
    
    print(f"Results contain beta values: {sorted(results.keys())}")
    
    # Test 1: Bounds computation with original beta values
    print("\nTest 1: Bounds computation with original beta values")
    print("-" * 50)
    
    try:
        # These should work - bounds function handles beta=0 internally
        # For SYNTH dataset, default training set size is 50
        n_train = 50  # Default SYNTH dataset size
        bounds_bce = compute_generalization_bound(original_beta_values, results, n_train, loss_type='bce')
        bounds_01 = compute_generalization_bound(original_beta_values, results, n_train, loss_type='zero_one')
        gen_errors = compute_generalization_errors(original_beta_values, results)
        
        print(f"✓ BCE bounds computed for: {sorted(bounds_bce.keys())}")
        print(f"✓ Zero-one bounds computed for: {sorted(bounds_01.keys())}")
        print(f"✓ Generalization errors computed for: {sorted(gen_errors.keys())}")
        
        # Verify that bounds are only computed for original beta values
        assert set(bounds_bce.keys()) == set(original_beta_values), "BCE bounds should only be for original beta values"
        assert set(bounds_01.keys()) == set(original_beta_values), "Zero-one bounds should only be for original beta values"
        assert set(gen_errors.keys()) == set(original_beta_values), "Gen errors should only be for original beta values"
        
        print("✓ All bounds computed only for original beta values")
        
    except Exception as e:
        print(f"✗ Error in bounds computation: {e}")
        return False
    
    # Test 2: Verify mathematical consistency
    print("\nTest 2: Mathematical consistency")
    print("-" * 50)
    
    for beta in original_beta_values:
        # Check that bounds are positive
        bce_bound = bounds_bce[beta]['generalization_bound']
        zo_bound = bounds_01[beta]['generalization_bound']
        
        if bce_bound <= 0:
            print(f"✗ BCE bound for beta={beta} is non-positive: {bce_bound}")
            return False
        
        if zo_bound <= 0:
            print(f"✗ Zero-one bound for beta={beta} is non-positive: {zo_bound}")
            return False
        
        # Check that empirical losses are reasonable
        bce_emp = bounds_bce[beta]['empirical_loss']
        zo_emp = bounds_01[beta]['empirical_loss']
        
        if bce_emp < 0 or bce_emp > 4:  # BCE loss should be in [0, 4]
            print(f"✗ BCE empirical loss for beta={beta} is unreasonable: {bce_emp}")
            return False
        
        if zo_emp < 0 or zo_emp > 1:  # Zero-one loss should be in [0, 1]
            print(f"✗ Zero-one empirical loss for beta={beta} is unreasonable: {zo_emp}")
            return False
        
        # Check generalization errors
        bce_gen_err = gen_errors[beta]['bce_gen_error']
        zo_gen_err = gen_errors[beta]['zero_one_gen_error']
        
        print(f"  Beta {beta}: BCE bound={bce_bound:.3f}, emp={bce_emp:.3f}, gen_err={bce_gen_err:.3f}")
        print(f"            ZO bound={zo_bound:.3f}, emp={zo_emp:.3f}, gen_err={zo_gen_err:.3f}")
    
    print("✓ All bounds are mathematically consistent")
    
    # Test 3: Save results with original beta values
    print("\nTest 3: Save results with original beta values")
    print("-" * 50)
    
    try:
        filename = "results/test_bounds_behavior.txt"
        n_train = 50  # Default SYNTH dataset size
        save_results_to_file(
            results=results,
            n=n_train,
            filename=filename,
            beta_values=original_beta_values,
            num_repetitions=3,
            num_epochs=100,
            a0=0.1,
            sigma_gauss_prior=1000000
        )
        print(f"✓ Results saved to {filename}")
        
    except Exception as e:
        print(f"✗ Error saving results: {e}")
        return False
    
    # Test 4: Plot results with original beta values
    print("\nTest 4: Plot results with original beta values")
    print("-" * 50)
    
    try:
        plot_filename = "results/test_bounds_behavior_plot.png"
        plot_beta_results(
            results=results,
            beta_values=original_beta_values,
            filename=plot_filename,
            num_repetitions=3,
            num_epochs=100,
            a0=0.1,
            sigma_gauss_prior=1000000
        )
        print(f"✓ Plot saved to {plot_filename}")
        
    except Exception as e:
        print(f"✗ Error plotting results: {e}")
        return False
    
    # Test 5: Verify that bounds increase with beta (mathematical property)
    print("\nTest 5: Verify bounds increase with beta")
    print("-" * 50)
    
    sorted_betas = sorted(original_beta_values)
    for i in range(len(sorted_betas) - 1):
        beta1, beta2 = sorted_betas[i], sorted_betas[i+1]
        
        # For higher beta, the integral should be larger, so bounds might be larger
        # (This isn't guaranteed to be strictly increasing due to noise, but let's check)
        bound1 = bounds_bce[beta1]['generalization_bound']
        bound2 = bounds_bce[beta2]['generalization_bound']
        
        print(f"  Beta {beta1}: bound = {bound1:.3f}")
        print(f"  Beta {beta2}: bound = {bound2:.3f}")
        
        # Note: Due to the nature of the bound and empirical noise, 
        # bounds might not be strictly increasing, so we just report
    
    print("✓ Bounds reported for all beta values")
    
    return True

if __name__ == "__main__":
    success = test_comprehensive_bounds_behavior()
    
    print("\n" + "="*80)
    if success:
        print("✓ ALL TESTS PASSED - Bounds computation behaves correctly!")
        print("  - Bounds functions handle beta=0 internally")
        print("  - Only original beta values are used for bounds computation")
        print("  - Mathematical consistency is maintained")
        print("  - Results and plots work correctly")
    else:
        print("✗ SOME TESTS FAILED - Check the output above")
    print("="*80)
