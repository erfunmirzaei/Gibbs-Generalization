"""
Main experiment script for the Gibbs generalization bound experiments.

This script orchestrates the complete ULA or SGLD experiment, testing different beta values
and computing PAC-Bayesian generalization bounds for the MNIST or CIFAR-10 datasets.
"""

from dataset import (get_mnist_binary_dataloaders, get_mnist_binary_dataloaders_random_labels,
                    get_cifar10_binary_dataloaders, get_cifar10_binary_dataloaders_random_labels)
from training import run_beta_experiments

# Configuration flags
TEST_MODE = False  # Set to True for quick test, False for full experiment
USE_RANDOM_LABELS = False  # Set to True for random labels, False for correct labels
DATASET_TYPE = 'mnist'  # 'mnist' or 'cifar10'

# MNIST classes for binary classification (only used when DATASET_TYPE='mnist')
# Can be either:
# - Individual classes: [0, 1] 
# - Grouped classes: [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]] for even vs odd
MNIST_CLASSES = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]  # Even vs Odd digits

# CIFAR-10 classes for binary classification (only used when DATASET_TYPE='cifar10')
# Can be either:
# - Individual classes: [0, 1] for airplane vs automobile
# - Grouped classes: [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]] for vehicles vs animals
CIFAR10_CLASSES = [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]]  # Vehicles vs Animals

def main():
    """Main experiment function."""
    device = 'cpu'  # cuda:5
    # Define beta values to test
    if TEST_MODE:
        print("\n" + "="*50)
        print("RUNNING IN TEST MODE")
        print("For full experiment, set TEST_MODE = False")
        print("="*50)

        if DATASET_TYPE == 'mnist':
            beta_values = [2000, 4000]  # Minimal set for testing
            a0 = {0: 0.01, 2000: 0.02, 4000: 0.02}
        
        elif DATASET_TYPE == 'cifar10':
            beta_values = [16000]  # Minimal set for testing
            a0 = {0: 0.01, 16000: 0.01}
        
    else:
        if DATASET_TYPE == 'mnist':
            beta_values = [125, 250, 500, 1000, 2000, 4000, 8000, 16000] # n = 2k
            # beta_values =  [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]  # Extended MNIST experiment, n = 8k
            #beta_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]  # Extended MNIST experiment, n = 2k
            # beta_values = [375, 750, 1250, 1500, 1750]
            # a0 = {375: 0.01, 750:0.01, 1250:0.01, 1500:0.01, 1750:0.01}
            # a0 = {0: 0.0025, 125: 0.0025, 250: 0.0025, 500: 0.0025, 1000: 0.0025, 2000: 0.0025, 4000: 0.0025, 8000: 0.0025, 16000: 0.0025} #TODO: previous best
            # a0 = {0: 0.005,500:0.005, 1000: 0.005, 2000: 0.005, 4000: 0.005, 8000: 0.005, 16000: 0.005, 32000: 0.005, 64000: 0.005}
            a0 = {0: 0.01, 125: 0.01, 250: 0.01, 500: 0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01}
            # a0 = {0: 0.01,500:0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01, 32000: 0.01, 64000: 0.01}

            # beta_values = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]  # Extended MNIST experiment, n = 8k
            #a0 = {0:0.01, 1000:0.01, 2000:0.01, 3000:0.01, 4000:0.01, 5000:0.01, 6000:0.01, 7000:0.01, 8000:0.01, 9000:0.01, 10000:0.01, 11000:0.01, 12000:0.01, 13000:0.01, 14000:0.01, 15000:0.01, 16000:0.01}
            # a0 = {0: 0.01,500:0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01, 32000: 0.01, 64000: 0.01}

        elif DATASET_TYPE == 'cifar10':
            # beta_values = [125, 250, 500, 1000, 2000, 4000, 8000, 16000] # n = 2k
            beta_values =  [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]  # Extended CIFAR-10 experiment, n = 8k
            # a0 = {0: 0.01, 125: 0.01, 250: 0.01, 500: 0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01}
            # a0 = {0: 0.005, 125: 0.005, 250: 0.005, 500: 0.005, 1000: 0.005, 2000: 0.005, 4000: 0.005, 8000: 0.005, 16000: 0.005}
            a0 = {0: 0.005,500:0.005, 1000: 0.005, 2000: 0.005, 4000: 0.005, 8000: 0.005, 16000: 0.005, 32000: 0.005, 64000: 0.005}

    
    print(f"\n{'='*70}")
    print(f"Gibbs Generalization EXPERIMENTS")
    print(f"Dataset: {DATASET_TYPE.upper()}")
    print(f"Beta values: {beta_values}")
    print(f"{'='*70}")

    # Create dataloaders once (same dataset for all repetitions and beta values)
    print("\nCreating dataset and dataloaders...")
    if DATASET_TYPE == 'mnist':
        if USE_RANDOM_LABELS:
            train_loader, test_loader = get_mnist_binary_dataloaders_random_labels(
                classes=MNIST_CLASSES,
                n_train_per_group=1000,
                n_test_per_group=5000,
                batch_size=2000,
                random_seed=42001,  # Fixed seed for consistent dataset
                normalize=True
            )
        else:
            train_loader, test_loader = get_mnist_binary_dataloaders(
                classes=MNIST_CLASSES,
                n_train_per_group=1000,
                n_test_per_group=5000,
                batch_size=2000,
                random_seed=42001,  # Fixed seed for consistent dataset
                normalize=True
            )
    elif DATASET_TYPE == 'cifar10':
        if USE_RANDOM_LABELS:
            train_loader, test_loader = get_cifar10_binary_dataloaders_random_labels(
                classes=CIFAR10_CLASSES,
                n_train_per_group=4000,
                n_test_per_group=5000,
                batch_size=100,
                random_seed=42001,  # Fixed seed for consistent dataset
            )
        else:
            train_loader, test_loader = get_cifar10_binary_dataloaders(
                classes=CIFAR10_CLASSES,
                n_train_per_group=4000,
                n_test_per_group=5000,
                batch_size=100,
                random_seed=42001,  # Fixed seed for consistent dataset
            )
    
    print(f"Dataset created with fixed random seed (42001) for consistency across all experiments")
    
    # Run the experiment with optimizations
    run_beta_experiments(
        loss = 'SAVAGE', #'Savage', #'BBCE', #'BCE', #'Tangent'
        beta_values=beta_values,
        a0=a0,  # Now supports dict, callable, or float
        b=0.5,  # This is used only if you want to schedule the step size (In the current version it is not used)
        sigma_gauss_prior=5,
        device=device,
        n_hidden_layers=1,  # 1 or 2 or 3 hidden layers, if you put 'L' it will be LeNet5 for MNIST and if you put 'V' it will be VGG16 for CIFAR10
        width=500, # Width of each hidden layer, only for fully connected networks
        dataset_type=DATASET_TYPE,  # 'cifar10' or 'mnist'
        use_random_labels=USE_RANDOM_LABELS,
        l_max=4.0,
        train_loader=train_loader, 
        test_loader=test_loader,
        min_steps=400,  # Minimum steps for subsequent betas (or all betas if not annealing)
        alpha_average=0.01,
        alpha_stop=0.00025,
        eta=0.1,  # This is used only if you want to schedule the step size (In the current version it is not used)
        eps=1e-7,
        test_mode=TEST_MODE,
        add_grad_norm=False,
        add_noise=True,  # If False, it becomes (S)GD
        sgld_num=1,  # Choose SGLD variant: 1 or 2
        annealed=True,  # Whether to use annealed SGLD
        min_steps_first_beta=4000,  # For annealing: min steps for first beta>0 (ignored if annealed=False)
    )
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED!")

if __name__ == "__main__":
    main()
