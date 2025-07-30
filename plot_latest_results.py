#!/usr/bin/env python3
"""Script to re-plot the latest MNIST results."""

import sys
import os
sys.path.append('/home/emirzaei/Gibbs-Generalization')

import json
import numpy as np
from plot_utils import plot_beta_results
from bounds import compute_generalization_bound, compute_generalization_errors

def load_and_plot_latest_results():
    """Load the latest MNIST results and create plots."""
    
    # Load the latest JSON results file
    json_file = '/home/emirzaei/Gibbs-Generalization/results/sgld_mnist_5clsv5cls_hce1b10d8d09f.json'
    
    if not os.path.exists(json_file):
        print(f"❌ Results file not found: {json_file}")
        return False
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        results = data['results']
        hyperparams = data['hyperparameters']
        
        print("✅ Loaded results successfully!")
        print(f"📊 Hyperparameters: {hyperparams['experiment_description']}")
        print(f"📈 Beta values in results: {sorted(results.keys())}")
        
        # Convert string keys to numeric for proper processing
        numeric_results = {}
        for k, v in results.items():
            try:
                numeric_key = float(k)
                numeric_results[numeric_key] = v
            except ValueError:
                numeric_results[k] = v
        
        # Extract the original beta values (excluding beta=0 which was added automatically)
        all_betas = sorted([float(k) for k in results.keys() if k != 'hyperparameters'])
        original_beta_values = [b for b in all_betas if b != 0.0]
        
        print(f"🎯 Original beta values: {original_beta_values}")
        
        # Get the training set size from hyperparameters
        n_train = hyperparams.get('train_dataset_size', 2000)  # Default fallback
        
        # Create experiment parameters for plotting
        experiment_params = {
            'beta_values': original_beta_values,
            'num_repetitions': hyperparams.get('num_repetitions', 10),
            'num_epochs': hyperparams.get('num_epochs', {}),
            'a0': hyperparams.get('a0', {}),
            'sigma_gauss_prior': hyperparams.get('sigma_gauss_prior', 1000),
            'dataset_type': hyperparams.get('dataset_type', 'mnist'),
        }
        
        print(f"📝 Experiment parameters:")
        for k, v in experiment_params.items():
            if isinstance(v, dict) and len(v) > 3:
                print(f"  {k}: {type(v).__name__} with {len(v)} entries")
            else:
                print(f"  {k}: {v}")
        
        # Generate plots
        print(f"\n🎨 Generating plots...")
        plot_beta_results(numeric_results, n_train, **experiment_params)
        
        # Print some key results
        print(f"\n📊 KEY RESULTS:")
        print("="*80)
        print(f"{'Beta':<8} {'Train BCE':<12} {'Test BCE':<12} {'Gen Error':<12} {'0-1 Train':<12} {'0-1 Test':<12} {'0-1 Gen Err':<12}")
        print("-" * 80)
        
        for beta in sorted(original_beta_values):
            train_bce = numeric_results[beta]['train_bce_mean']
            test_bce = numeric_results[beta]['test_bce_mean']
            gen_err_bce = test_bce - train_bce
            train_01 = numeric_results[beta]['train_01_mean'] 
            test_01 = numeric_results[beta]['test_01_mean']
            gen_err_01 = test_01 - train_01
            
            print(f"{beta:<8.0f} {train_bce:<12.4f} {test_bce:<12.4f} {gen_err_bce:<12.4f} {train_01:<12.4f} {test_01:<12.4f} {gen_err_01:<12.4f}")
        
        # Test bounds computation
        print(f"\n🧮 Testing bounds computation...")
        try:
            bounds = compute_generalization_bound(original_beta_values, numeric_results, n_train, loss_type='bce')
            zo_bounds = compute_generalization_bound(original_beta_values, numeric_results, n_train, loss_type='zero_one')
            gen_errors = compute_generalization_errors(original_beta_values, numeric_results)
            
            print("✅ Bounds computed successfully!")
            print(f"\n📈 BOUNDS SUMMARY:")
            print("="*80)
            print(f"{'Beta':<8} {'BCE Bound':<12} {'BCE Gap':<12} {'0-1 Bound':<12} {'0-1 Gap':<12}")
            print("-" * 80)
            
            for beta in sorted(original_beta_values):
                bce_bound = bounds[beta]['generalization_bound']
                bce_gap = bce_bound - gen_errors[beta]['bce_gen_error']
                zo_bound = zo_bounds[beta]['generalization_bound']
                zo_gap = zo_bound - gen_errors[beta]['zero_one_gen_error']
                
                print(f"{beta:<8.0f} {bce_bound:<12.4f} {bce_gap:<12.4f} {zo_bound:<12.4f} {zo_gap:<12.4f}")
            
        except Exception as e:
            print(f"❌ Error computing bounds: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n🎉 Plotting completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = load_and_plot_latest_results()
    if success:
        print(f"\n✅ Results loaded and plotted successfully!")
        print(f"📁 Check the results/ directory for the updated plots.")
    else:
        print(f"\n❌ Failed to process results.")
