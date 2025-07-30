#!/usr/bin/env python3
"""Test script to verify bounds.py is working correctly."""

import sys
import os
sys.path.append('/home/emirzaei/Gibbs-Generalization')

import numpy as np
from bounds import (
    compute_generalization_bound, 
    compute_generalization_errors, 
    compute_individual_generalization_bounds,
    compute_kl_divergence_analysis
)

def create_mock_results():
    """Create mock experimental results for testing."""
    beta_values = [0, 1000, 4000]
    
    # Create mock results dictionary
    results = {}
    
    for beta in beta_values:
        # Simulate 5 repetitions for each beta
        n_reps = 5
        
        # Generate realistic mock data
        if beta == 0:
            # Beta=0 should have lower training error but higher test error (overfitting)
            raw_train_bce = np.random.uniform(0.1, 0.2, n_reps).tolist()
            raw_test_bce = np.random.uniform(0.3, 0.4, n_reps).tolist()
            raw_train_01 = np.random.uniform(0.05, 0.15, n_reps).tolist()
            raw_test_01 = np.random.uniform(0.25, 0.35, n_reps).tolist()
        else:
            # Higher beta should have better generalization
            raw_train_bce = np.random.uniform(0.15, 0.25, n_reps).tolist()
            raw_test_bce = np.random.uniform(0.18, 0.28, n_reps).tolist()
            raw_train_01 = np.random.uniform(0.1, 0.2, n_reps).tolist()
            raw_test_01 = np.random.uniform(0.12, 0.22, n_reps).tolist()
        
        results[beta] = {
            'raw_train_bce': raw_train_bce,
            'raw_test_bce': raw_test_bce,
            'raw_train_01': raw_train_01,
            'raw_test_01': raw_test_01,
            'train_bce_mean': np.mean(raw_train_bce),
            'train_bce_std': np.std(raw_train_bce, ddof=1),
            'train_bce_var': np.var(raw_train_bce, ddof=1),
            'test_bce_mean': np.mean(raw_test_bce),
            'test_bce_std': np.std(raw_test_bce, ddof=1),
            'test_bce_var': np.var(raw_test_bce, ddof=1),
            'train_01_mean': np.mean(raw_train_01),
            'train_01_std': np.std(raw_train_01, ddof=1),
            'train_01_var': np.var(raw_train_01, ddof=1),
            'test_01_mean': np.mean(raw_test_01),
            'test_01_std': np.std(raw_test_01, ddof=1),
            'test_01_var': np.var(raw_test_01, ddof=1),
        }
    
    return results, beta_values

def test_bounds_computation():
    """Test all bounds computation functions."""
    print("🧪 Testing bounds computation functions...")
    
    # Create mock data
    results, beta_values = create_mock_results()
    n_train = 2000  # Mock training set size
    
    # Test regular beta values (excluding beta=0)
    test_betas = [1000, 4000]
    
    try:
        print("\n1. Testing compute_generalization_bound for BCE...")
        bce_bounds = compute_generalization_bound(test_betas, results, n_train, loss_type='bce')
        print(f"✅ BCE bounds computed successfully for betas: {list(bce_bounds.keys())}")
        
        print("\n2. Testing compute_generalization_bound for zero-one...")
        zo_bounds = compute_generalization_bound(test_betas, results, n_train, loss_type='zero_one')
        print(f"✅ Zero-one bounds computed successfully for betas: {list(zo_bounds.keys())}")
        
        print("\n3. Testing compute_generalization_errors...")
        gen_errors = compute_generalization_errors(test_betas, results)
        print(f"✅ Generalization errors computed successfully for betas: {list(gen_errors.keys())}")
        
        print("\n4. Testing compute_individual_generalization_bounds for BCE...")
        indiv_bce_bounds = compute_individual_generalization_bounds(test_betas, results, n_train, loss_type='bce')
        print(f"✅ Individual BCE bounds computed successfully for betas: {list(indiv_bce_bounds.keys())}")
        
        print("\n5. Testing compute_individual_generalization_bounds for zero-one...")
        indiv_zo_bounds = compute_individual_generalization_bounds(test_betas, results, n_train, loss_type='zero_one')
        print(f"✅ Individual zero-one bounds computed successfully for betas: {list(indiv_zo_bounds.keys())}")
        
        print("\n6. Testing compute_kl_divergence_analysis for BCE...")
        kl_analysis_bce = compute_kl_divergence_analysis(test_betas, results, n_train, loss_type='bce')
        print(f"✅ KL divergence analysis (BCE) computed successfully for betas: {list(kl_analysis_bce.keys())}")
        
        print("\n7. Testing compute_kl_divergence_analysis for zero-one...")
        kl_analysis_zo = compute_kl_divergence_analysis(test_betas, results, n_train, loss_type='zero_one')
        print(f"✅ KL divergence analysis (zero-one) computed successfully for betas: {list(kl_analysis_zo.keys())}")
        
        # Print sample results
        print("\n📊 Sample Results:")
        print("="*60)
        for beta in test_betas:
            print(f"\nBeta = {beta}:")
            print(f"  Train BCE: {results[beta]['train_bce_mean']:.4f}")
            print(f"  Test BCE:  {results[beta]['test_bce_mean']:.4f}")
            print(f"  BCE Gen Error: {gen_errors[beta]['bce_gen_error']:.4f}")
            print(f"  BCE Bound: {bce_bounds[beta]['generalization_bound']:.4f}")
            print(f"  Individual BCE Bound (mean): {indiv_bce_bounds[beta]['bound_mean']:.4f} ± {indiv_bce_bounds[beta]['bound_std']:.4f}")
            
            print(f"  Train 0-1: {results[beta]['train_01_mean']:.4f}")
            print(f"  Test 0-1:  {results[beta]['test_01_mean']:.4f}")
            print(f"  0-1 Gen Error: {gen_errors[beta]['zero_one_gen_error']:.4f}")
            print(f"  0-1 Bound: {zo_bounds[beta]['generalization_bound']:.4f}")
            print(f"  Individual 0-1 Bound (mean): {indiv_zo_bounds[beta]['bound_mean']:.4f} ± {indiv_zo_bounds[beta]['bound_std']:.4f}")
            
            print(f"  KL Mean (BCE): {kl_analysis_bce[beta]['kl_mean']:.4f}")
            print(f"  KL Mean (0-1): {kl_analysis_zo[beta]['kl_mean']:.4f}")
        
        print("\n🎉 All bounds computation tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in bounds computation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bounds_computation()
    if success:
        print("\n✅ bounds.py is working correctly!")
    else:
        print("\n❌ There are issues with bounds.py that need to be fixed.")
