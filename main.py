"""
Main experim# Test mode flag - set to False for full experiment
TEST_MODE = True

# Random labels flag - set to True to use random labels instead of linear relationship
USE_RANDOM_LABELS = Falsecript for the Gibbs generalization bound experiments.

This script orchestrates the complete SGLD experiment, testing different beta values
and computing PAC-Bayesian generalization bounds for the SYNTH dataset.
"""

import torch
import numpy as np
import time
from dataset import (create_synth_dataset, get_synth_dataloaders, create_synth_dataset_random_labels, get_synth_dataloaders_random_labels,
                    create_mnist_binary_dataset, get_mnist_binary_dataloaders,
                    create_mnist_binary_dataset_random_labels, get_mnist_binary_dataloaders_random_labels)
from training import run_beta_experiments
from bounds import compute_generalization_bound, compute_generalization_errors, save_results_to_file
from plot_utils import plot_beta_results

# Test mode flag - set to False for full experiment
TEST_MODE =  True

# Random labels flag - set to True to use random labels instead of linear relationship
USE_RANDOM_LABELS = False

# Dataset selection - set to 'mnist' for MNIST binary classification or 'synth' for synthetic
DATASET_TYPE = 'mnist'  # 'synth' or 'mnist'

# MNIST classes for binary classification (only used when DATASET_TYPE='mnist')
# Can be either:
# - Individual classes: [0, 1] 
# - Grouped classes: [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]] for even vs odd
MNIST_CLASSES = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]  # Even vs Odd digits

def main():
    """Main experiment function."""
    # GPU diagnostic and setup
    # device = 'cuda:5'
    device = 'cpu'  # Force CPU for compatibility in this environment
    # Define beta values to test
    if TEST_MODE:
        print("\n" + "="*50)
        print("RUNNING IN TEST MODE")
        print("For full experiment, set TEST_MODE = False")
        print("="*50)

        if DATASET_TYPE == 'mnist':
            # MNIST needs fewer epochs typically - FAST TEST MODE
            beta_values = [125]  # Minimal set for testing
            num_repetitions = 1  # Very fast testing
            # num_epochs = {0: 1, 1: 3000}  # Much fewer epochs
            a0 = {0: 1e-10, 125: 0.01}
        else:
            # SYNTH dataset configuration - FAST TEST MODE
            beta_values = [1, 10]  # Minimal set for testing  
            num_repetitions = 2  # Very fast testing
            num_epochs = {0: 10, 1: 20, 10: 50}  # Much fewer epochs
            a0 = {0: 0.01, 1: 0.01, 10: 0.1}
        
    else:
        if DATASET_TYPE == 'mnist':
            beta_values = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]  # Full MNIST experiment
            # beta_values = [1000, 2000, 4000, 8000, 16000, 32000, 64000]  # Extended MNIST experiment
            num_repetitions = 1  # Full experiment
            # num_epochs = {0: 1, 1:10000, 250: 10000, 500: 10000, 1000: 10000, 2000: 40000, 4000: 40000, 8000: 40000, 16000: 40000}
            # a0 = {0: 1e-10, 125:0.2, 250: 0.1, 500: 0.05, 1000: 0.025, 2000: 0.0125, 4000: 0.00625, 8000: 0.003125, 16000: 0.0015625}
            a0 = {0: 1e-10, 125: 0.01, 250: 0.01, 500: 0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01}
            # a0 = {0: 1e-10,500:0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01, 32000: 0.01, 64000: 0.01}

        else:
            beta_values = [0, 1, 10, 30, 50, 70, 100, 200]  # Full SYNTH experiment
            num_repetitions = 30  # Full experiment
            num_epochs = {0: 1, 1: 100, 10: 5000, 30: 10000, 50: 15000, 70: 20000, 100: 30000, 200: 30000}
            a0 = {0: 1e-7, 1: 1e-7, 10: 1e-1, 30: 1e-1, 50: 1e-1, 70: 1e-1, 100: 1e-1, 200: 1e-1}
    
    print(f"\n{'='*70}")
    print(f"SGLD BETA EXPERIMENTS")
    print(f"Dataset: {DATASET_TYPE.upper()}")
    print(f"Beta values: {beta_values}")
    print(f"Repetitions per beta: {num_repetitions}")
    # if isinstance(num_epochs, dict):
    #     print(f"Epochs per training: Variable by beta")
    #     for beta in sorted(set(list(num_epochs.keys()) + beta_values)):
    #         if beta in num_epochs:
    #             print(f"  Beta {beta}: {num_epochs[beta]} epochs")
    #         else:
    #             print(f"  Beta {beta}: {num_epochs.get(beta, 1000)} epochs (default)")
    # elif callable(num_epochs):
    #     print(f"Epochs per training: Adaptive function")
    #     for beta in [0] + beta_values:
    #         print(f"  Beta {beta}: {num_epochs(beta)} epochs")
    # else:
    #     print(f"Epochs per training: {num_epochs}")
    
    if isinstance(a0, dict):
        print(f"Learning rate (a0): Variable by beta")
        for beta in sorted(set(list(a0.keys()) + beta_values)):
            if beta in a0:
                print(f"  Beta {beta}: a0 = {a0[beta]}")
            else:
                print(f"  Beta {beta}: a0 = {a0.get(beta, 0.1)} (default)")
    elif callable(a0):
        print(f"Learning rate (a0): Adaptive function")
        for beta in [0] + beta_values:
            print(f"  Beta {beta}: a0 = {a0(beta)}")
    else:
        print(f"Learning rate (a0): {a0}")
    print(f"{'='*70}")

    # Create dataloaders once (same dataset for all repetitions and beta values)
    print("\nCreating dataset and dataloaders...")
    if DATASET_TYPE == 'mnist':
        if USE_RANDOM_LABELS:
            train_loader, test_loader = get_mnist_binary_dataloaders_random_labels(
                classes=MNIST_CLASSES,
                n_train_per_group=1000,
                n_test_per_group=1000,
                batch_size=2000,
                random_seed=42001,  # Fixed seed for consistent dataset
                normalize=True
            )
        else:
            train_loader, test_loader = get_mnist_binary_dataloaders(
                classes=MNIST_CLASSES,
                n_train_per_group=1000,
                n_test_per_group=1000,
                batch_size=2000,
                random_seed=42001,  # Fixed seed for consistent dataset
                normalize=True
            )
    else:
        if USE_RANDOM_LABELS:
            train_loader, test_loader = get_synth_dataloaders_random_labels(
                batch_size=10,
                random_seed=42  # Fixed seed for consistent dataset
            )
        else:
            train_loader, test_loader = get_synth_dataloaders(
                batch_size=10,
                random_seed=42  # Fixed seed for consistent dataset
            )
    
    print(f"Dataset created with fixed random seed (42) for consistency across all experiments")

    
    # Run the experiment with optimizations
    results = run_beta_experiments(
        beta_values=beta_values,
        num_repetitions=num_repetitions,
        a0=a0,  # Now supports dict, callable, or float
        b=0.5,
        sigma_gauss_prior=5,
        device=device,
        dataset_type=DATASET_TYPE,  # 'synth' or 'mnist'
        use_random_labels=USE_RANDOM_LABELS,
        l_max=4.0,
        mnist_classes=MNIST_CLASSES if DATASET_TYPE == 'mnist' else None,
        train_loader=train_loader,  # Pass the pre-created dataloaders
        test_loader=test_loader,
        save_output_products_csv=False, 
        moving_average_outputs=True,
        alpha=0.00025, eta=0.1, eps=1e-7
    )
    
    # Get training set size for bounds computation
    n_train = len(train_loader.dataset)
    

    # Create comprehensive hyperparameter dictionary
    from results_manager import create_hyperparameter_dict, save_or_merge_results
    
    hyperparams = create_hyperparameter_dict(
        beta_values=beta_values,
        num_repetitions=num_repetitions,
        a0=a0,
        b=0.55,
        sigma_gauss_prior=1000,
        device=device,
        dataset_type=DATASET_TYPE,
        use_random_labels=USE_RANDOM_LABELS,
        l_max=4.0,
        mnist_classes=MNIST_CLASSES if DATASET_TYPE == 'mnist' else None,
        train_dataset_size=len(train_loader.dataset),
        test_dataset_size=len(test_loader.dataset),
        batch_size=train_loader.batch_size,
        random_seed=42000,  # The seed used for dataset creation
        alpha=0.1,
        eta=0.1,
        eps=1e-6,
        normalize=True
    )
    
    # Save results with hyperparameter tracking and potential merging
    print(f"\n{'='*70}")
    print("SAVING RESULTS WITH HYPERPARAMETER TRACKING")
    print(f"{'='*70}")
    
    results_filename, was_merged = save_or_merge_results(results, hyperparams)
    
    if was_merged:
        print("✅ Results merged with existing data!")
        print("The saved file now contains results from multiple experiment runs.")
        
        # Load the merged results for plotting
        from results_manager import load_existing_results
        _, merged_results = load_existing_results(results_filename)
        plot_results = merged_results
    else:
        print("💾 New results file created.")
        plot_results = results
    
    # Plot the results (now with potentially merged data)
    experiment_params = {
        'beta_values': beta_values,
        'num_repetitions': num_repetitions,
        'a0': a0,
        'sigma_gauss_prior': 1000,
        'dataset_type': DATASET_TYPE,
        'use_random_labels': USE_RANDOM_LABELS,
        'hyperparams': hyperparams,

    }
    
    plot_beta_results(plot_results, n_train, **experiment_params)
    
    # Also save in the old format for backward compatibility
    print("\n📄 Also saving in legacy format for backward compatibility...")
    save_results_to_file(plot_results, n_train, **experiment_params)
    
    # Get the generated filenames for display
    from bounds import generate_filename
    legacy_results_filename = generate_filename(
        file_type='results', extension='txt', 
        **experiment_params
    )
    plot_filename = generate_filename(
        file_type='plot', extension='png', 
        **experiment_params
    )
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED!")
    print(f"Files generated:")
    print(f"  - {results_filename} (enhanced results with hyperparameter tracking)")
    print(f"  - results/{legacy_results_filename} (legacy format)")
    print(f"  - results/{plot_filename} (plots)")
    
    if was_merged:
        print(f"\n🔄 RESULTS MERGED:")
        print(f"  Your new results have been combined with existing results")
        print(f"  that had identical hyperparameters.")
        print(f"  The plots now show data from multiple experiment runs.")
    
    print(f"\n💡 HYPERPARAMETER TRACKING:")
    print(f"  Future experiments with identical hyperparameters will be")
    print(f"  automatically merged with these results.")
    
    from results_manager import generate_hyperparameter_hash
    print(f"  Hyperparameter hash: {generate_hyperparameter_hash(hyperparams)}")

if __name__ == "__main__":
    
    # Run main experiment
    main()
