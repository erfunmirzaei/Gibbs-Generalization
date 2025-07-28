"""
Main experim# Test mode flag - set to False for full experiment
TEST_MODE = False

# Random labels flag - set to True to use random labels instead of linear relationship
USE_RANDOM_LABELS = Falsecript for the Gibbs generalization bound experiments.

This script orchestrates the complete SGLD experiment, testing different beta values
and computing PAC-Bayesian generalization bounds for the SYNTH dataset.
"""

import torch
import numpy as np
from dataset import (create_synth_dataset, get_synth_dataloaders, create_synth_dataset_random_labels,
                    create_mnist_binary_dataset, get_mnist_binary_dataloaders)
from models import SynthNN, MNISTNN, initialize_kaiming_and_get_prior_sigma
from losses import BoundedCrossEntropyLoss, ZeroOneLoss
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
MNIST_CLASSES = [0, 1]

def main():
    """Main experiment function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test the dataset creation (quick check)
    if DATASET_TYPE == 'mnist':
        train_dataset, test_dataset = create_mnist_binary_dataset(
            classes=MNIST_CLASSES,
            n_train_per_class=1000,
            n_test_per_class=500,
            random_seed=42,
            normalize=True
        )
        dataset_name = f"MNIST Binary (classes {MNIST_CLASSES[0]} vs {MNIST_CLASSES[1]})"
        dataset_type_str = f'mnist_{MNIST_CLASSES[0]}v{MNIST_CLASSES[1]}'
    else:
        if USE_RANDOM_LABELS:
            train_dataset, test_dataset = create_synth_dataset_random_labels(random_seed=42)
            dataset_name = "SYNTH dataset with RANDOM LABELS"
            dataset_type_str = 'synth_random_labels'
        else:
            train_dataset, test_dataset = create_synth_dataset(random_seed=42)
            dataset_name = "SYNTH dataset with LINEAR LABELS"
            dataset_type_str = 'synth'
    
    print(f"Using {dataset_name}")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Define beta values to test
    if TEST_MODE:
        print("\n" + "="*50)
        print("RUNNING IN TEST MODE")
        print("For full experiment, set TEST_MODE = False")
        print("="*50)

        
        if DATASET_TYPE == 'mnist':
            # MNIST needs fewer epochs typically
            beta_values = [1000, 2000]  # Reduced set for testing
            num_repetitions = 3  # Reduced for testing
            num_epochs = {0: 1, 1000: 1000, 2000: 1000}
            a0 = {0: 1e-5, 1000: 1e-3, 2000: 0.001}
        else:
            # SYNTH dataset configuration
            beta_values = [1, 10, 50]  # Reduced set for testing
            num_repetitions = 3  # Reduced for testing
            num_epochs = {0: 100, 1: 100, 10: 1000, 50: 1000}
            a0 = {0: 0.01, 1: 0.01, 10: 0.1, 50: 0.1}
        
    else:
        if DATASET_TYPE == 'mnist':
            beta_values = [0, 1, 10, 30, 50]  # Full MNIST experiment
            num_repetitions = 10  # Full experiment
            num_epochs = {0: 1, 1: 100, 10: 500, 30: 1000, 50: 1500}
            a0 = {0: 1e-7, 1: 1e-3, 10: 1e-2, 30: 1e-2, 50: 1e-2}
        else:
            beta_values = [0, 1, 10, 30, 50, 70, 100, 200]  # Full SYNTH experiment
            num_repetitions = 30  # Full experiment
            num_epochs = {0: 1, 1: 100, 10: 5000, 30: 10000, 50: 15000, 70: 20000, 100: 30000, 200: 30000}
            a0 = {0: 1e-7, 1: 1e-7, 10: 1e-1, 30: 1e-1, 50: 1e-1, 70: 1e-1, 100: 1e-1, 200: 1e-1}
    
    print(f"\n{'='*70}")
    print(f"SGLD BETA EXPERIMENTS")
    print(f"Dataset: {dataset_name}")
    print(f"Beta values: {beta_values}")
    print(f"Repetitions per beta: {num_repetitions}")
    if isinstance(num_epochs, dict):
        print(f"Epochs per training: Variable by beta")
        for beta in sorted(set(list(num_epochs.keys()) + beta_values)):
            if beta in num_epochs:
                print(f"  Beta {beta}: {num_epochs[beta]} epochs")
            else:
                print(f"  Beta {beta}: {num_epochs.get(beta, 1000)} epochs (default)")
    elif callable(num_epochs):
        print(f"Epochs per training: Adaptive function")
        for beta in [0] + beta_values:
            print(f"  Beta {beta}: {num_epochs(beta)} epochs")
    else:
        print(f"Epochs per training: {num_epochs}")
    
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
    
    # Run the experiment
    results = run_beta_experiments(
        beta_values=beta_values,
        num_repetitions=num_repetitions,
        num_epochs=num_epochs,
        a0=a0,  # Now supports dict, callable, or float
        b=0.5,
        sigma_gauss_prior=1000,
        device=device,
        dataset_type=DATASET_TYPE,  # 'synth' or 'mnist'
        use_random_labels=USE_RANDOM_LABELS,
        l_max=4.0,
        mnist_classes=MNIST_CLASSES if DATASET_TYPE == 'mnist' else None
    )
    
    # Print final summary
    print(f"\n{'='*90}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*90}")
    print("Beta\tTrain_BCE\tTest_BCE\tBCE_GenErr\tBCE_Bound\tBound_Gap\tTrain_01\tTest_01\tZO_GenErr\tZO_Bound\tZO_Gap")
    print("-" * 110)
    
    # Compute bounds and generalization errors for summary
    # The bounds functions will handle the need for beta=0 internally
    bounds = compute_generalization_bound(beta_values, results, loss_type='bce')
    zero_one_bounds = compute_generalization_bound(beta_values, results, loss_type='zero_one')
    gen_errors = compute_generalization_errors(beta_values, results)
    
    # Display results only for the originally requested beta values
    for beta in sorted(beta_values):
        train_bce = results[beta]['train_bce_mean']
        test_bce = results[beta]['test_bce_mean']
        bce_gen_error = gen_errors[beta]['bce_gen_error']
        theo_bound = bounds[beta]['generalization_bound']
        
        # Bound gap: how much larger the theoretical bound is compared to actual generalization error
        # Positive means bound is valid, negative means bound is violated
        bound_gap = theo_bound - bce_gen_error
        
        train_01 = results[beta]['train_01_mean']
        test_01 = results[beta]['test_01_mean']
        zo_gen_error = gen_errors[beta]['zero_one_gen_error']
        zo_theo_bound = zero_one_bounds[beta]['generalization_bound']
        zo_bound_gap = zo_theo_bound - zo_gen_error
        
        print(f"{beta}\t{train_bce:.4f}\t\t{test_bce:.4f}\t\t{bce_gen_error:.4f}\t\t{theo_bound:.4f}\t\t{bound_gap:.4f}\t\t{train_01:.4f}\t\t{test_01:.4f}\t\t{zo_gen_error:.4f}\t\t{zo_theo_bound:.4f}\t\t{zo_bound_gap:.4f}")
    
    print(f"\nGeneralization Error = Test Loss - Train Loss")
    print(f"Bound Gap = Theoretical Bound - Actual Generalization Error")
    print(f"  > 0: Bound is valid (theoretical bound > actual generalization error)")
    print(f"  ≈ 0: Bound is tight")
    print(f"  < 0: Bound is violated (should not happen with high probability)")
    
    # Define experimental parameters for saving/plotting
    # Use original beta values for filename generation and plotting (excluding auto-added beta=0)
    experiment_params = {
        'beta_values': beta_values,
        'num_repetitions': num_repetitions,
        'num_epochs': num_epochs,
        'a0': a0,  # Now supports variable a0
        'sigma_gauss_prior': 10,
        'dataset_type': dataset_type_str
    }
    
    # Save results to file with descriptive filename (using original beta values for filename)
    save_results_to_file(results, **experiment_params)
    
    # Plot the results with descriptive filename (only plot original beta values, not beta=0)
    plot_beta_results(results, **experiment_params)
    
    # Get the generated filenames for display
    from bounds import generate_filename
    results_filename = generate_filename(file_type='results', extension='txt', **experiment_params)
    plot_filename = generate_filename(file_type='plot', extension='png', **experiment_params)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED!")
    print(f"Files generated:")
    print(f"  - results/{results_filename} (numerical results)")
    print(f"  - results/{plot_filename} (plots)")
    if TEST_MODE:
        print(f"\nTo run the full experiment:")
        print(f"  1. Set TEST_MODE = False in the script")
        print(f"  2. Re-run the script")
        print(f"  3. The full experiment will take much longer!")
    print(f"{'='*70}")


def test_loss_functions():
    """Test the loss functions with sample data."""
    print("\n" + "="*50)
    print("Testing Bounded Cross Entropy Loss (Quick Demo)")
    print("="*50)
    
    criterion = BoundedCrossEntropyLoss(ell_max=4.0)
    zero_one_criterion = ZeroOneLoss()
    
    # Test with single output (SYNTH style)
    test_logits_single = torch.randn(5, 1)  # 5 samples, 1 output
    test_targets = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32)
    
    bounded_loss_single = criterion(test_logits_single, test_targets)
    zero_one_loss_single = zero_one_criterion(test_logits_single, test_targets)
    
    print(f"Bounded CE Loss (single output): {bounded_loss_single.item():.4f}")
    print(f"Zero-One Loss (single output): {zero_one_loss_single.item():.4f}")
    print(f"Loss bounded in [0, {criterion.ell_max}]: {0 <= bounded_loss_single.item() <= criterion.ell_max}")
    
    # Test with binary output (2 classes)
    test_logits_binary = torch.randn(5, 2)
    bounded_loss_binary = criterion(test_logits_binary, test_targets)
    zero_one_loss_binary = zero_one_criterion(test_logits_binary, test_targets)
    
    print(f"Bounded CE Loss (binary output): {bounded_loss_binary.item():.4f}")
    print(f"Zero-One Loss (binary output): {zero_one_loss_binary.item():.4f}")


if __name__ == "__main__":
    # Test loss functions first
    test_loss_functions()
    
    # Run main experiment
    main()
