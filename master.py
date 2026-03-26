"""
Main experiment script for the Gibbs generalization bound experiments.

This script orchestrates the complete ULA or SGLD experiment, testing different beta values
and computing PAC-Bayesian generalization bounds for MNIST (binary and multi-class classification).

Comparison with PBB Paper:
--------------------------
To compare results with the PAC-Bayes with Backprop (PBB) paper (Perez-Ortiz et al., 2020),
you can use the PBB-compatible architectures and loss functions:

1. Set USE_PBB_CONFIG = True to enable PBB-compatible settings
2. Choose architecture: PBB_ARCHITECTURE = 'fc' (NNet4l) or 'cnn' (CNNet4l)
3. Loss function: PBB_LOSS_TYPE = 'nll' (matches PBB) or 'ce'
4. Prior distribution: PBB_PRIOR_DIST = 'gaussian', 'laplace', or 'truncated_gaussian'

Example for direct PBB comparison:
    USE_PBB_CONFIG = True
    PBB_ARCHITECTURE = 'cnn'  # CNNet4l for fair comparison
    PBB_LOSS_TYPE = 'nll'     # NLL loss used in PBB paper
    PBB_PRIOR_DIST = 'gaussian'  # Standard Gaussian prior

The networks are defined with the exact same architecture and training setup as the
original PBB repository (https://github.com/mperezortiz/PBB).
"""

import torch
import numpy as np
import random
import csv
from datetime import datetime
from dataset import (
    get_mnist_binary_dataloaders_partial_random_labels,
    get_synth_dataloaders
)
from multiclass_dataset_functions import (
    get_mnist_multiclass_dataloaders,
    get_mnist_multiclass_dataloaders_partial_random_labels
)
from training import run_beta_experiments
from pbb_models import NNet4l, CNNet4l
from pbb_prior import initialize_model_with_prior, get_prior_initializer

# Configuration flags
TEST_MODE = False  # Set to True for quick test, False for full experiment
USE_RANDOM_LABELS = 1  # Percentage of randomly labeled data 
DATASET_TYPE = 'mnist'  # 'synth', 'mnist', 'cifar10' or 'cifar100'
SEEDS = [42]  # Random seeds for stability analysis
DATASET_SEED = 42  # Seed for dataset splitting/label randomization (if applicable)
USE_SAME_DATASET_ACROSS_SEEDS = True  # True: same dataset split/labels for all seeds

# MNIST classification mode: 'binary' or 'multiclass'
MNIST_CLASS_MODE = 'binary'  # 'binary' for 2-class, 'multiclass' for multi-class

# MNIST classes for binary classification (only used when DATASET_TYPE='mnist' and MNIST_CLASS_MODE='binary')
# Can be either:
# - Individual classes: [0, 1] 
# - Grouped classes: [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]] for even vs odd
MNIST_CLASSES_BINARY = [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]  # Even vs Odd digits

# MNIST classes for multi-class classification (only used when DATASET_TYPE='mnist' and MNIST_CLASS_MODE='multiclass')
# Can include any subset of digits 0-9
MNIST_CLASSES_MULTICLASS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # All 10 digits

# ============================================================================
# PBB (PAC-Bayes with Backprop) Configuration
# ============================================================================
# Set USE_PBB_CONFIG = True to use identical settings from the PBB paper for comparison
USE_PBB_CONFIG = True  # Set True to match PBB paper exactly

# PBB Architecture selection: 'fc' (NNet4l) or 'cnn' (CNNet4l) for MNIST
# NNet4l: 4-layer fully connected network (600-600-600 hidden units)
# CNNet4l: 4-layer convolutional network (1->32->64 channels)
PBB_ARCHITECTURE = 'fc'  # Options: 'fc', 'cnn'

# PBB Loss function: 'nll' (F.nll_loss) or 'ce' (F.cross_entropy)
# NLL uses log_softmax output (as in original PBB and our models)
# This is automatically handled by the loss selection in training
PBB_LOSS_TYPE = 'nll'  # Options: 'nll', 'ce'

# PBB Prior distribution: 'gaussian', 'laplace', or 'truncated_gaussian'
# 'truncated_gaussian' enables layer-wise truncated initialization + layer-wise SGLD prior scale
PBB_PRIOR_DIST = 'truncated_gaussian'  # Options: 'gaussian', 'laplace', 'truncated_gaussian'
PBB_SIGMA_PRIOR = 0.03  # PBB running_example uses SIGMAPRIOR = 0.03

def set_global_seed(seed):
    """Set random seed across torch/numpy/python for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For fully deterministic behavior on CUDA (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_pbb_model(architecture: str, num_classes: int = 10, device: str = 'cpu'):
    """
    Get a PBB-compatible model (NNet4l or CNNet4l).
    
    Args:
        architecture (str): 'fc' for NNet4l or 'cnn' for CNNet4l
        num_classes (int): Number of output classes. Default: 10 (MNIST)
        device (str): Device to create model on ('cpu', 'cuda', etc.)
        
    Returns:
        nn.Module: The model moved to the specified device
    """
    if architecture.lower() in ['fc', 'nnet4l']:
        model = NNet4l(num_classes=num_classes, dropout_prob=0.0)
    elif architecture.lower() in ['cnn', 'cnnet4l']:
        model = CNNet4l(num_classes=num_classes, dropout_prob=0.0)
    else:
        raise ValueError(f"Unknown PBB architecture: {architecture}. "
                        f"Use 'fc' (NNet4l) or 'cnn' (CNNet4l)")
    
    return model.to(device)


def create_dataloaders(dataset_seed):
    """Create dataset-specific train/test dataloaders."""
    if DATASET_TYPE == 'mnist':
        if MNIST_CLASS_MODE == 'binary':
            return get_mnist_binary_dataloaders_partial_random_labels(
                classes=MNIST_CLASSES_BINARY,
                p=USE_RANDOM_LABELS,
                n_train_per_group=1000,
                n_test_per_group=5000,
                batch_size=2000,
                random_seed=dataset_seed,
                normalize=True
            )
        elif MNIST_CLASS_MODE == 'multiclass':
            return get_mnist_multiclass_dataloaders_partial_random_labels(
                classes=MNIST_CLASSES_MULTICLASS,
                p=USE_RANDOM_LABELS,
                n_train_per_class=6000,
                n_test_per_class=1000,
                batch_size=250,
                random_seed=dataset_seed,
                normalize=True
            )
        else:
            raise ValueError(f"Unsupported MNIST_CLASS_MODE: {MNIST_CLASS_MODE}")

    raise ValueError(f"Unsupported DATASET_TYPE: {DATASET_TYPE}")


def save_seed_stability_summary(seed_results):
    """Save compact stability analysis table across seeds."""
    if not seed_results:
        return None

    rows = []
    for result in seed_results:
        for csv_path in result["csv_paths"]:
            rows.append([
                result["seed"],
                result["dataset_seed"],
                csv_path,
            ])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = f"csv_EMA/SEED_STABILITY_{DATASET_TYPE.upper()}_{timestamp}.csv"
    with open(summary_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["seed", "dataset_seed", "result_csv_path"])
        writer.writerows(rows)

    return summary_path

def main():
    """Main experiment function."""
    if not SEEDS:
        raise ValueError("SEEDS must contain at least one seed value")

    set_global_seed(SEEDS[0])
    print(f"Seed list: {SEEDS}")

    # Automatically select GPU if available, otherwise use CPU
    if torch.cuda.is_available():
        device = f'cuda:{torch.cuda.current_device()}'
        print(f"🚀 GPU detected and will be used: {device}")
        print(f"   GPU Name: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties().total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print(f"⚠️  No GPU detected, using CPU for training")
    # Define beta values to test
    if TEST_MODE:
        print("\n" + "="*50)
        print("RUNNING IN TEST MODE")
        print("For full experiment, set TEST_MODE = False")
        print("="*50)

        if DATASET_TYPE == 'mnist':
            beta_values = [256, 2000]  # Minimal set for testing
            a0 = {0: 0.01, 256:0.01, 2000: 0.01}

        elif DATASET_TYPE in ('cifar10', 'cifar100'):
            beta_values = [128, 250, 500, 1000]  # Minimal set for testing
            a0 = {0: 0.01, 128: 0.01, 250: 0.01, 500: 0.01, 1000: 0.01}

        elif DATASET_TYPE == 'synth':
            beta_values = [50]  # Minimal set for testing
            a0 = {0: 0.01, 50: 0.01}
        
    else:
        if DATASET_TYPE == 'mnist':
            # New beta values for whole MNIST experiment
            beta_values = [500, 1000, 2000, 4000, 8000, 16000, 30000, 60000, 120000, 600000]
            a0 = {0:0.01, 500:0.01, 1000:0.01, 2000:0.01, 4000:0.01, 8000:0.01, 16000:0.01, 30000:0.01, 60000:0.01, 120000:0.01, 600000:0.01}
            
    print(f"\n{'='*70}")
    print(f"Gibbs Generalization EXPERIMENTS")
    print(f"Dataset: {DATASET_TYPE.upper()}")
    print(f"Beta values: {beta_values}")
    print(f"{'='*70}")

    seed_results = []

    for seed in SEEDS:
        dataset_seed = DATASET_SEED
        print("\n" + "-" * 70)
        print(f"Running seed {seed} (dataset_seed={dataset_seed})")
        print("-" * 70)

        set_global_seed(seed)

        print("\nCreating dataset and dataloaders...")
        train_loader, test_loader = create_dataloaders(dataset_seed)

        if DATASET_TYPE == 'mnist':
            selected_classes = MNIST_CLASSES_BINARY if MNIST_CLASS_MODE == 'binary' else MNIST_CLASSES_MULTICLASS

        else:
            selected_classes = None

        csv_paths = run_beta_experiments(
            loss='SAVAGE', #'Savage', #'BBCE', #'BCE', #'Tangent'
            beta_values=beta_values,
            a0=a0,  # Now supports dict, callable, or float
            b=0.5,  # This is used only if you want to schedule the step size (In the current version it is not used)
            sigma_gauss_prior=5.0,
            device=device,
            n_hidden_layers=1,  # 1 or 2 or 3 hidden layers, if you put 'L' it will be LeNet5 for MNIST and if you put 'V' it will be VGG16 for CIFAR10
            width=500, # Width of each hidden layer, only for fully connected networks
            dataset_type=DATASET_TYPE,  # 'cifar10' or 'mnist'
            use_random_labels=USE_RANDOM_LABELS,
            l_max=4.0,
            train_loader=train_loader,
            test_loader=test_loader,
            min_steps=2000,  # Minimum steps for subsequent betas (or all betas if not annealing)
            alpha_average=0.01,
            alpha_stop=0.00025,
            eta=36,  # This is used only if you want to schedule the step size (In the current version it is not used)
            eps=-1e-7,
            test_mode=TEST_MODE,
            add_grad_norm=True,
            add_noise=True,  # If False, it becomes (S)GD
            sgld_num=1,  # Choose SGLD variant: 1 or 2
            annealed=False,  # Whether to use annealed SGLD
            min_steps_first_beta=4000,  # For annealing: min steps for first beta>0 (ignored if annealed=False)
            seed=seed,
            selected_classes=selected_classes,
            # PBB Configuration (optional - only used if USE_PBB_CONFIG=True)
            use_pbb_models=USE_PBB_CONFIG,
            pbb_architecture=PBB_ARCHITECTURE if USE_PBB_CONFIG else None,
            prior_type=PBB_PRIOR_DIST if USE_PBB_CONFIG else None,
            sigma_prior=PBB_SIGMA_PRIOR if USE_PBB_CONFIG else None,
            max_epochs=500,  # Maximum epochs per beta (hard limit), None for convergence-based only
        )

        seed_results.append({
            "seed": seed,
            "dataset_seed": dataset_seed,
            "csv_paths": csv_paths or [],
        })

    summary_path = save_seed_stability_summary(seed_results)
    if summary_path is not None:
        print(f"\nSeed stability summary saved to: {summary_path}")
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED!")

if __name__ == "__main__":
    main()
