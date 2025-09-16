"""
Main experim# Test mode flag - set to False for full experiment
TEST_MODE = True

# Random labels flag - set to True to use random labels instead of linear relationship
USE_RANDOM_LABELS = Falsecript for the Gibbs generalization bound experiments.

This script orchestrates the complete SGLD experiment, testing different beta values
and computing PAC-Bayesian generalization bounds for the SYNTH dataset.
"""

from dataset import (create_synth_dataset, get_synth_dataloaders, create_synth_dataset_random_labels, get_synth_dataloaders_random_labels,
                    create_mnist_binary_dataset, get_mnist_binary_dataloaders,
                    create_mnist_binary_dataset_random_labels, get_mnist_binary_dataloaders_random_labels)
from training import run_beta_experiments

# Questions: 
# 1. Bounded loss function: Could relax it by decreasing the prior variance (sigma). what is the role of prior.
# 2. Why for SGLD we don't see the shift anymore? At least not as strong as ULA.
# 3. For deeper networks, the plot looks different - why? We are visiting a phase transition?
#  finally, the bounds are working without any calibration. I guess this is excellent news for the paper, and also, it is very interesting to think about why making the neural net deeper and more overparametrized has such an effect.
# 4. What is the the role of stopping criterion to be close or far from the optimum?
# TODO: Future: 
# 0. 1 hidden layer with 1000 units for MNIST
# 1. Savage loss, ULA/SGLD?, 1L/2L?, n=2k/8k?

# 2. Burn-in phase with larger lr and then SGLD with smaller learning rate 
# 3. SGLD with scheduler
# 4. CIFAR-10 binary classification

# Test mode flag - set to False for full experiment
TEST_MODE =  False

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
            beta_values = [500]  # Minimal set for testing
            num_repetitions = 1  # Very fast testing
            # num_epochs = {0: 1, 1: 3000}  # Much fewer epochs
            a0 = {0: 1e-10, 500.: 0.01}
        else:
            # SYNTH dataset configuration - FAST TEST MODE
            beta_values = [1, 10]  # Minimal set for testing  
            num_repetitions = 2  # Very fast testing
            num_epochs = {0: 10, 1: 20, 10: 50}  # Much fewer epochs
            a0 = {0: 0.01, 1: 0.01, 10: 0.1}
        
    else:
        if DATASET_TYPE == 'mnist':
            # beta_values = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]  # Full MNIST experiment
            beta_values = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]  # Extended MNIST experiment
            num_repetitions = 1  # Full experiment
            # num_epochs = {0: 1, 1:10000, 250: 10000, 500: 10000, 1000: 10000, 2000: 40000, 4000: 40000, 8000: 40000, 16000: 40000}
            # a0 = {0: 1e-10, 125:0.2, 250: 0.1, 500: 0.05, 1000: 0.025, 2000: 0.0125, 4000: 0.00625, 8000: 0.003125, 16000: 0.0015625}
            # a0 = {0: 1e-10, 125: 0.01, 250: 0.01, 500: 0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01}
            a0 = {0: 1e-10,500:0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01, 32000: 0.01, 64000: 0.01}

        else:
            beta_values = [0, 1, 10, 30, 50, 70, 100, 200]  # Full SYNTH experiment
            num_repetitions = 30  # Full experiment
            num_epochs = {0: 1, 1: 100, 10: 5000, 30: 10000, 50: 15000, 70: 20000, 100: 30000, 200: 30000}
            a0 = {0: 1e-7, 1: 1e-7, 10: 1e-1, 30: 1e-1, 50: 1e-1, 70: 1e-1, 100: 1e-1, 200: 1e-1}
    
    print(f"\n{'='*70}")
    print(f"Gibbs Generalization EXPERIMENTS")
    print(f"Dataset: {DATASET_TYPE.upper()}")
    print(f"Beta values: {beta_values}")
    print(f"Repetitions per beta: {num_repetitions}")
    print(f"{'='*70}")

    # Create dataloaders once (same dataset for all repetitions and beta values)
    print("\nCreating dataset and dataloaders...")
    if DATASET_TYPE == 'mnist':
        if USE_RANDOM_LABELS:
            train_loader, test_loader = get_mnist_binary_dataloaders_random_labels(
                classes=MNIST_CLASSES,
                n_train_per_group=4000,
                n_test_per_group=5000,
                batch_size=100,
                random_seed=42002,  # Fixed seed for consistent dataset
                normalize=True
            )
        else:
            train_loader, test_loader = get_mnist_binary_dataloaders(
                classes=MNIST_CLASSES,
                n_train_per_group=4000,
                n_test_per_group=5000,
                batch_size=100,
                random_seed=42002,  # Fixed seed for consistent dataset
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
    
    print(f"Dataset created with fixed random seed (42001) for consistency across all experiments")

    
    # Run the experiment with optimizations
    run_beta_experiments(
        beta_values=beta_values,
        a0=a0,  # Now supports dict, callable, or float
        b=0.5,
        sigma_gauss_prior=5,
        device=device,
        n_hidden_layers=2,  # 1 or 2 hidden layers for MNIST
        width=1000,
        dataset_type=DATASET_TYPE,  # 'synth' or 'mnist'
        use_random_labels=USE_RANDOM_LABELS,
        l_max=4.0,
        train_loader=train_loader, 
        test_loader=test_loader,
        min_epochs = 4000,
        alpha_average= 0.01, alpha_stop=0.00025, eta=0.1, eps=1e-7
    )
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED!")

if __name__ == "__main__":
    main()
