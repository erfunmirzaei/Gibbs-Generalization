"""
Main experim# Test mode flag - set to False for full experiment
TEST_MODE = True

# Random labels flag - set to True to use random labels instead of linear relationship
USE_RANDOM_LABELS = Falsecript for the Gibbs generalization bound experiments.

This script orchestrates the complete SGLD experiment, testing different beta values
and computing PAC-Bayesian generalization bounds for the SYNTH dataset.
"""

from dataset import (get_synth_dataloaders, get_synth_dataloaders_random_labels,
                    get_mnist_binary_dataloaders, get_mnist_binary_dataloaders_random_labels,
                    get_cifar10_binary_dataloaders, get_cifar10_binary_dataloaders_random_labels)
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

# Dataset selection - set to 'mnist' for MNIST binary classification or 'cifar10' for CIFAR-10 binary classification
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
            beta_values = [2000]  # Minimal set for testing
            a0 = {0: 0.01, 2000: 0.01}
        
        elif DATASET_TYPE == 'cifar10':
            # CIFAR-10 needs fewer epochs typically - FAST TEST MODE
            beta_values = [16000]  # Minimal set for testing
            a0 = {0: 1e-10, 16000: 0.01}
        
    else:
        if DATASET_TYPE == 'mnist':
            beta_values = [125, 250, 500, 1000, 2000, 4000, 8000, 16000] # n = 2k 
            # beta_values = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]  # Extended MNIST experiment, n = 8k
            # a0 = {0: 1e-10, 125:0.2, 250: 0.1, 500: 0.05, 1000: 0.025, 2000: 0.0125, 4000: 0.00625, 8000: 0.003125, 16000: 0.0015625}
            # a0 = {0: 1e-10, 125: 0.01, 250: 0.01, 500: 0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01}
            a0 = {0: 0.005, 125: 0.005, 250: 0.005, 500: 0.005, 1000: 0.005, 2000: 0.005, 4000: 0.005, 8000: 0.005, 16000: 0.005}
            # a0 = {0: 1e-10,500:0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01, 32000: 0.01, 64000: 0.01}

        elif DATASET_TYPE == 'cifar10':
            beta_values = [125, 250, 500, 1000, 2000, 4000, 8000, 16000] # n = 2k
            # beta_values = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]  # Extended MNIST experiment, n = 8k
            a0 = {0: 1e-10, 125: 0.01, 250: 0.01, 500: 0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01}
            # a0 = {0: 1e-10,500:0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01, 32000: 0.01, 64000: 0.01}

    
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
                batch_size=50,
                random_seed=42001,  # Fixed seed for consistent dataset
                normalize=True
            )
        else:
            train_loader, test_loader = get_mnist_binary_dataloaders(
                classes=MNIST_CLASSES,
                n_train_per_group=1000,
                n_test_per_group=5000,
                batch_size=50,
                random_seed=42001,  # Fixed seed for consistent dataset
                normalize=True
            )
    elif DATASET_TYPE == 'cifar10':
        if USE_RANDOM_LABELS:
            train_loader, test_loader = get_cifar10_binary_dataloaders_random_labels(
                classes=CIFAR10_CLASSES,
                n_train_per_group=1000,
                n_test_per_group=5000,
                batch_size=2000,
                random_seed=42001,  # Fixed seed for consistent dataset
            )
        else:
            train_loader, test_loader = get_cifar10_binary_dataloaders(
                classes=CIFAR10_CLASSES,
                n_train_per_group=1000,
                n_test_per_group=5000,
                batch_size=2000,
                random_seed=42001,  # Fixed seed for consistent dataset
            )

    
    print(f"Dataset created with fixed random seed (42001) for consistency across all experiments")

    
    # Run the experiment with optimizations
    run_beta_experiments(
        loss = 'BBCE', #'Savage', #'BBCE', #'BCE', #'Tangent'
        beta_values=beta_values,
        a0=a0,  # Now supports dict, callable, or float
        b=0.5,
        sigma_gauss_prior=5,
        device=device,
        n_hidden_layers=2,  # 1 or 2 hidden layers
        width=1000,
        dataset_type=DATASET_TYPE,  # 'cifar10' or 'mnist'
        use_random_labels=USE_RANDOM_LABELS,
        l_max=4.0,
        train_loader=train_loader, 
        test_loader=test_loader,
        min_epochs = 4000,
        alpha_average= 0.01, alpha_stop=0.00025, eta=0.1, eps=1e-7,test_mode = TEST_MODE, add_grad_norm = False, add_noise = False
    )
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED!")

if __name__ == "__main__":
    main()
