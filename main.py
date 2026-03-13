"""
Main experiment script for the Gibbs generalization bound experiments.

This script orchestrates the complete ULA or SGLD experiment, testing different beta values
and computing PAC-Bayesian generalization bounds for the MNIST or CIFAR-10 datasets.
"""

import torch
import numpy as np
import random
import csv
from datetime import datetime
from dataset import (get_mnist_binary_dataloaders_partial_random_labels,
                     get_cifar10_binary_dataloaders_partial_random_labels,
                     get_synth_dataloaders, get_synth_dataloaders_random_labels,)
from training import run_beta_experiments

# TODO: Check the initial values effect for M_t when using BCE
# Configuration flags
TEST_MODE = True  # Set to True for quick test, False for full experiment
USE_RANDOM_LABELS = 1  # Percentage of randomly labeled data 
DATASET_TYPE = 'mnist'  # 'synth', 'mnist' or 'cifar10'
SEEDS = [72, 82, 92]  # Random seeds for stability analysis
DATASET_SEED = 42  # Seed for dataset splitting/label randomization (if applicable)
USE_SAME_DATASET_ACROSS_SEEDS = True  # True: same dataset split/labels for all seeds

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


def create_dataloaders(dataset_seed):
    """Create dataset-specific train/test dataloaders."""
    if DATASET_TYPE == 'mnist':
        return get_mnist_binary_dataloaders_partial_random_labels(
            classes=MNIST_CLASSES,
            p=USE_RANDOM_LABELS,
            n_train_per_group=1000,
            n_test_per_group=5000,
            batch_size=2000,
            random_seed=dataset_seed,
            normalize=True
        )

    if DATASET_TYPE == 'cifar10':
        return get_cifar10_binary_dataloaders_partial_random_labels(
            classes=CIFAR10_CLASSES,
            p=USE_RANDOM_LABELS,
            n_train_per_group=1000,
            n_test_per_group=5000,
            batch_size=2000,
            random_seed=dataset_seed,
        )

    if DATASET_TYPE == 'synth':
        if USE_RANDOM_LABELS:
            return get_synth_dataloaders_random_labels(
                batch_size=50,
                random_seed=dataset_seed,
            )

        return get_synth_dataloaders(
            batch_size=50,
            random_seed=dataset_seed,
        )

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

    device = 'cpu'  # cuda:5
    # Define beta values to test
    if TEST_MODE:
        print("\n" + "="*50)
        print("RUNNING IN TEST MODE")
        print("For full experiment, set TEST_MODE = False")
        print("="*50)

        if DATASET_TYPE == 'mnist':
            beta_values = [ 1000, 8000]  # Minimal set for testing
            a0 = {0: 0.01, 1000: 0.01, 8000: 0.01}

        elif DATASET_TYPE == 'cifar10':
            beta_values = [16000]  # Minimal set for testing
            a0 = {0: 0.01, 16000: 0.01}

        elif DATASET_TYPE == 'synth':
            beta_values = [50]  # Minimal set for testing
            a0 = {0: 0.01, 50: 0.01}
        
    else:
        if DATASET_TYPE == 'mnist':
            # beta_values = [100, 125, 160, 200, 250, 320, 400, 500]
            beta_values = [125, 250, 500, 1000, 2000, 4000, 8000, 16000] # n = 2k
            # beta_values =  [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]  # Extended MNIST experiment, n = 8k
            #beta_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]  # Extended MNIST experiment, n = 2k
            # beta_values = [375, 750, 1250, 1500, 1750]

            # a0 = {375: 0.01, 750:0.01, 1250:0.01, 1500:0.01, 1750:0.01}
            # a0 = {0: 0.0025, 125: 0.0025, 250: 0.0025, 500: 0.0025, 1000: 0.0025, 2000: 0.0025, 4000: 0.0025, 8000: 0.0025, 16000: 0.0025} #TODO: previous best
            # a0 = {0: 0.005,500:0.005, 1000: 0.005, 2000: 0.005, 4000: 0.005, 8000: 0.005, 16000: 0.005, 32000: 0.005, 64000: 0.005}
            a0 = {0: 0.01, 125: 0.01, 250: 0.01, 500: 0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01}
            # a0 = {0: 0.01,500:0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01, 32000: 0.01, 64000: 0.01}
            # a0 = {0: 0.001, 100:0.001, 125:0.001, 160:0.001, 200:0.001, 250:0.001, 320:0.001, 400:0.001, 500:0.001}

            # beta_values = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]  # Extended MNIST experiment, n = 8k
            #a0 = {0:0.01, 1000:0.01, 2000:0.01, 3000:0.01, 4000:0.01, 5000:0.01, 6000:0.01, 7000:0.01, 8000:0.01, 9000:0.01, 10000:0.01, 11000:0.01, 12000:0.01, 13000:0.01, 14000:0.01, 15000:0.01, 16000:0.01}
            # a0 = {0: 0.01,500:0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01, 32000: 0.01, 64000: 0.01}

        elif DATASET_TYPE == 'cifar10':
            beta_values = [125, 250, 500, 1000, 2000, 4000, 8000, 16000] # n = 2k
            # beta_values =  [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]  # Extended CIFAR-10 experiment, n = 8k
            # a0 = {0: 0.01, 125: 0.01, 250: 0.01, 500: 0.01, 1000: 0.01, 2000: 0.01, 4000: 0.01, 8000: 0.01, 16000: 0.01}
            a0 = {0: 0.005, 125: 0.005, 250: 0.005, 500: 0.005, 1000: 0.005, 2000: 0.005, 4000: 0.005, 8000: 0.005, 16000: 0.005}
            # a0 = {0: 0.005,500:0.005, 1000: 0.005, 2000: 0.005, 4000: 0.005, 8000: 0.005, 16000: 0.005, 32000: 0.005, 64000: 0.005}

        elif DATASET_TYPE == 'synth':
            beta_values = [3,6,12,25,50,100,200,400]
            a0 = {0:0.5, 3:36.0, 6:18.0, 12:9.0, 25:4.5, 50:2.25, 100:1.2, 200:0.6, 400:0.3}
            
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
