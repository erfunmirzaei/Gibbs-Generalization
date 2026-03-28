"""
Main experiment script for multiclass Gibbs generalization experiments.

This script mirrors the structure of main.py but runs MNIST multi-class experiments.
"""

import csv
import random
from datetime import datetime

import numpy as np
import torch

from multiclass_dataset_functions import get_mnist_multiclass_dataloaders_partial_random_labels
from training_multiclass import run_beta_experiments

# Configuration flags
TEST_MODE = False  # Set to True for quick test, False for full experiment
USE_RANDOM_LABELS = 1  # Probability of randomizing each label in [0, 1]
DATASET_TYPE = 'mnist'  # Multiclass path currently supports MNIST
SEEDS = [42]  # Random seeds for stability analysis
DATASET_SEED = 42  # Seed for dataset splitting/label randomization
USE_SAME_DATASET_ACROSS_SEEDS = True

# MNIST classes for multi-class classification
MNIST_CLASSES_MULTICLASS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Experiment-level controls
LOSS_FUNCTION = 'bbce'  # 'ce', 'bbce', 'savage', 'nll'
SGLD_SIGMA_GAUSS_PRIOR = 5.0
MAX_ITER = 1000
STOPPING_MODE = 'max_iter_only'  # 'ema' or 'max_iter_only'

# Epoch learning-rate schedule:
# Epochs 1-100: 0.05, 101-200: 0.04, 201-300: 0.03, ...
USE_EPOCH_LR_SCHEDULE = True
EPOCH_LR_START = 0.05
EPOCH_LR_DECREMENT = 0.001
EPOCH_LR_STEP_EPOCHS = 15
EPOCH_LR_MIN = 1e-3


def set_global_seed(seed):
    """Set random seed across torch/numpy/python for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(dataset_seed):
    """Create train/test dataloaders for multiclass MNIST."""
    if DATASET_TYPE != 'mnist':
        raise ValueError(f"Unsupported DATASET_TYPE for multiclass mode: {DATASET_TYPE}")

    return get_mnist_multiclass_dataloaders_partial_random_labels(
        classes=MNIST_CLASSES_MULTICLASS,
        p=USE_RANDOM_LABELS,
        n_train_per_class=6000,
        n_test_per_class=1000,
        batch_size=250,
        random_seed=dataset_seed,
        normalize=True,
    )


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

    if torch.cuda.is_available():
        current_gpu = torch.cuda.current_device()
        device = f'cuda:{current_gpu}'
        print(f"GPU detected and will be used: {device}")
        print(f"GPU Name: {torch.cuda.get_device_name(current_gpu)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(current_gpu).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print("No GPU detected, using CPU for training")

    if TEST_MODE:
        print("\n" + "=" * 50)
        print("RUNNING IN TEST MODE")
        print("For full experiment, set TEST_MODE = False")
        print("=" * 50)
        beta_values = [600000]
        a0 = {0: 0.01, 600000: 0.01}
    else:
        beta_values = [250, 500, 1000, 2000, 4000, 8000, 16000, 30000, 60000, 120000, 600000]
        a0 = {
            0: 0.01,
            250: 0.01,
            500: 0.01,
            1000: 0.01,
            2000: 0.01,
            4000: 0.01,
            8000: 0.01,
            16000: 0.01,
            30000: 0.01,
            60000: 0.01,
            120000: 0.01,
            600000: 0.01,
        }

    print(f"\n{'=' * 70}")
    print("Gibbs Generalization EXPERIMENTS (MULTICLASS)")
    print(f"Dataset: {DATASET_TYPE.upper()}")
    print(f"Classes: {MNIST_CLASSES_MULTICLASS}")
    print(f"Beta values: {beta_values}")
    print(f"Loss: {LOSS_FUNCTION}, MAX_ITER: {MAX_ITER}, stopping: {STOPPING_MODE}")
    if USE_EPOCH_LR_SCHEDULE:
        print(
            "Epoch LR schedule: "
            f"start={EPOCH_LR_START}, decrement={EPOCH_LR_DECREMENT} every {EPOCH_LR_STEP_EPOCHS} epochs, "
            f"min={EPOCH_LR_MIN}"
        )
    print(f"{'=' * 70}")

    seed_results = []

    for seed in SEEDS:
        dataset_seed = DATASET_SEED if USE_SAME_DATASET_ACROSS_SEEDS else seed

        print("\n" + "-" * 70)
        print(f"Running seed {seed} (dataset_seed={dataset_seed})")
        print("-" * 70)

        set_global_seed(seed)

        print("\nCreating dataset and dataloaders...")
        train_loader, test_loader = create_dataloaders(dataset_seed)

        csv_paths = run_beta_experiments(
            loss=LOSS_FUNCTION,
            beta_values=beta_values,
            a0=a0,
            b=0.5,
            sigma_gauss_prior=SGLD_SIGMA_GAUSS_PRIOR,
            device=device,
            n_hidden_layers=3,
            width=600,
            dataset_type=DATASET_TYPE,
            use_random_labels=USE_RANDOM_LABELS,
            l_max=4.0,
            train_loader=train_loader,
            test_loader=test_loader,
            min_steps=2000,
            alpha_average=0.01,
            alpha_stop=0.00025,
            eta=36,
            eps=-1e-7,
            test_mode=TEST_MODE,
            add_grad_norm=True,
            add_noise=True,
            sgld_num=1,
            annealed=False,
            min_steps_first_beta=4000,
            seed=seed,
            selected_classes=MNIST_CLASSES_MULTICLASS,
            max_epochs=MAX_ITER,
            stopping_mode=STOPPING_MODE,
            resume_from_checkpoint=False,
            use_epoch_lr_schedule=USE_EPOCH_LR_SCHEDULE,
            epoch_lr_start=EPOCH_LR_START,
            epoch_lr_decrement=EPOCH_LR_DECREMENT,
            epoch_lr_step_epochs=EPOCH_LR_STEP_EPOCHS,
            epoch_lr_min=EPOCH_LR_MIN,
        )

        seed_results.append({
            "seed": seed,
            "dataset_seed": dataset_seed,
            "csv_paths": csv_paths or [],
        })

    summary_path = save_seed_stability_summary(seed_results)
    if summary_path is not None:
        print(f"\nSeed stability summary saved to: {summary_path}")

    print(f"\n{'=' * 70}")
    print("EXPERIMENT COMPLETED!")


if __name__ == "__main__":
    main()
