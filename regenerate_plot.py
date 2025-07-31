#!/usr/bin/env python3
"""
Regenerate the MNIST plot with updated bounds computation.
This script loads the existing results and re-plots them with the current bounds.py implementation.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Add the project directory to path
sys.path.append('/home/emirzaei/Gibbs-Generalization')

from bounds import (
    compute_generalization_bound, 
    compute_generalization_errors,
    compute_individual_generalization_bounds
)
from plot_utils import plot_beta_results

def regenerate_mnist_plot():
    """Load the MNIST results and regenerate the plot with updated bounds."""
    
    print("🔄 Regenerating MNIST plot with updated bounds...")
    
    # Load the latest MNIST JSON results
    json_file = '/home/emirzaei/Gibbs-Generalization/results/sgld_mnist_5clsv5cls_hce1b10d8d09f.json'
    
    if not os.path.exists(json_file):
        print(f"❌ Results file not found: {json_file}")
        return False
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract results and hyperparameters
        string_results = data['results']
        hyperparams = data['hyperparameters']
        
        # Convert string keys to numeric for bounds computation
        results = {}
        for k, v in string_results.items():
            try:
                numeric_key = float(k)
                results[numeric_key] = v
            except ValueError:
                results[k] = v
        
        # Get the original beta values and training set size
        original_beta_values = hyperparams['beta_values']  # [250, 500, 1000, 1500, 2000, 4000, 8000, 16000]
        n_train = hyperparams['train_dataset_size']
        
        print(f"✅ Loaded results successfully!")
        print(f"📊 Original beta values: {original_beta_values}")
        print(f"📈 Training set size: {n_train}")
        print(f"🔢 Results keys: {sorted(results.keys())}")
        
        # Test the bounds computation with current bounds.py
        print(f"\n🧮 Computing bounds with updated bounds.py...")
        
        # Compute bounds for both loss types
        bce_bounds = compute_generalization_bound(original_beta_values, results, n_train, loss_type='bce')
        zo_bounds = compute_generalization_bound(original_beta_values, results, n_train, loss_type='zero_one')
        gen_errors = compute_generalization_errors(original_beta_values, results)
        
        print(f"✅ Bounds computed successfully!")
        
        # Display updated bounds
        print(f"\n📊 UPDATED BOUNDS SUMMARY:")
        print("="*90)
        print(f"{'Beta':<8} {'Train BCE':<10} {'Test BCE':<10} {'BCE Gen':<10} {'BCE Bound':<10} {'BCE Gap':<10} {'ZO Gen':<10} {'ZO Bound':<10} {'ZO Gap':<10}")
        print("-" * 90)
        
        for beta in sorted(original_beta_values):
            train_bce = results[beta]['train_bce_mean']
            test_bce = results[beta]['test_bce_mean']
            bce_gen = gen_errors[beta]['bce_gen_error']
            bce_bound = bce_bounds[beta]['generalization_bound']
            bce_gap = bce_bound - bce_gen
            
            zo_gen = gen_errors[beta]['zero_one_gen_error']
            zo_bound = zo_bounds[beta]['generalization_bound']
            zo_gap = zo_bound - zo_gen
            
            print(f"{beta:<8.0f} {train_bce:<10.4f} {test_bce:<10.4f} {bce_gen:<10.4f} {bce_bound:<10.4f} {bce_gap:<10.4f} {zo_gen:<10.4f} {zo_bound:<10.4f} {zo_gap:<10.4f}")
        
        # Prepare experiment parameters for plotting
        experiment_params = {
            'beta_values': original_beta_values,
            'num_repetitions': hyperparams['num_repetitions'],
            'num_epochs': hyperparams['num_epochs'],
            'a0': hyperparams['a0'],
            'sigma_gauss_prior': hyperparams['sigma_gauss_prior'],
            'dataset_type': hyperparams['dataset_type'],
        }
        
        print(f"\n🎨 Regenerating plot with updated bounds...")
        
        # Generate the new plot
        plot_beta_results(results, n_train, **experiment_params)
        
        # Generate expected filename
        from bounds import generate_filename
        plot_filename = generate_filename(
            file_type='plot', 
            extension='png', 
            **experiment_params
        )
        
        print(f"\n🎉 Plot regeneration completed!")
        print(f"📁 New plot saved as: results/{plot_filename}")
        print(f"📊 This plot includes the updated bounds computation from your modified bounds.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error regenerating plot: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = regenerate_mnist_plot()
    if success:
        print(f"\n✅ Successfully regenerated MNIST plot with updated bounds!")
    else:
        print(f"\n❌ Failed to regenerate plot.")
