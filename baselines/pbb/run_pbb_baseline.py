"""
Run a non-Langevin PBB baseline with original PBB architectures.

This script is standalone by design so core files remain untouched.
It uses:
- exact dataset-construction settings from main.py,
- PBB architectures (NNet4l / CNNet4l),
- PBB-style bounded NLL (pmin),
- PBB prior initialization,
- deterministic SGLD updates (add_noise=False).
"""

from __future__ import annotations

import csv
import os
import random
import sys
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# Allow running this script from any cwd (e.g., baselines/pbb/) by ensuring
# repository-root modules like dataset.py are importable.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import (
    get_cifar10_binary_dataloaders_partial_random_labels,
    get_cifar100_binary_dataloaders_partial_random_labels,
    get_mnist_binary_dataloaders_partial_random_labels,
    get_synth_dataloaders,
    get_synth_dataloaders_random_labels,
)
from losses import PBBBoundedNLLLoss
from pbb_models import CNNet4l, NNet4l
from pbb_prior import initialize_model_with_prior
from pbb_truncated_prior import initialize_prior_truncated_gaussian
from sgld import SGLD


# -----------------------------------------------------------------------------
# Experiment Configuration
# -----------------------------------------------------------------------------
TEST_MODE = False
SEEDS = [42]
DATASET_SEED = 42

DATASET_TYPE = "mnist"  # 'mnist' | 'cifar10' | 'cifar100' | 'synth'
USE_RANDOM_LABELS = 1
MNIST_CLASSES = [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]
CIFAR10_CLASSES = [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]]
CIFAR100_CLASSES = [55, 88]

# PBB architecture choice: 'fc' (NNet4l) or 'cnn' (CNNet4l)
PBB_ARCHITECTURE = "fc"

# PBB objective/prior knobs.
PBB_PMIN = 1e-5
PBB_PRIOR_TYPE = "truncated_gaussian"  # 'gaussian' | 'laplace' | 'truncated_gaussian'
PBB_SIGMA_PRIOR = 0.03

# Deterministic SGLD (non-Langevin) knobs.
SIGMA_GAUSS_PRIOR_FOR_OPT = 5.0
BETA_VALUES = [16000]
A0_BY_BETA = {16000: 0.01}
MAX_EPOCHS = 500 if not TEST_MODE else 5


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(dataset_seed: int):
    if DATASET_TYPE == "mnist":
        return get_mnist_binary_dataloaders_partial_random_labels(
            classes=MNIST_CLASSES,
            p=USE_RANDOM_LABELS,
            n_train_per_group=1000,
            n_test_per_group=5000,
            batch_size=2000,
            random_seed=dataset_seed,
            normalize=True,
        )

    if DATASET_TYPE == "cifar10":
        return get_cifar10_binary_dataloaders_partial_random_labels(
            classes=CIFAR10_CLASSES,
            p=USE_RANDOM_LABELS,
            n_train_per_group=1000,
            n_test_per_group=5000,
            batch_size=2000,
            random_seed=dataset_seed,
        )

    if DATASET_TYPE == "cifar100":
        return get_cifar100_binary_dataloaders_partial_random_labels(
            classes=CIFAR100_CLASSES,
            p=USE_RANDOM_LABELS,
            n_train_per_group=500,
            n_test_per_group=100,
            batch_size=1000,
            random_seed=dataset_seed,
        )

    if DATASET_TYPE == "synth":
        if USE_RANDOM_LABELS:
            return get_synth_dataloaders_random_labels(batch_size=50, random_seed=dataset_seed)
        return get_synth_dataloaders(batch_size=50, random_seed=dataset_seed)

    raise ValueError(f"Unsupported DATASET_TYPE: {DATASET_TYPE}")


def build_model() -> nn.Module:
    if PBB_ARCHITECTURE.lower() in ["fc", "nnet4l"]:
        return NNet4l(num_classes=2, dropout_prob=0.0)
    if PBB_ARCHITECTURE.lower() in ["cnn", "cnnet4l"]:
        if DATASET_TYPE != "mnist":
            raise ValueError("CNNet4l is MNIST-only in this repository.")
        return CNNet4l(num_classes=2, dropout_prob=0.0)
    raise ValueError(f"Unsupported PBB_ARCHITECTURE: {PBB_ARCHITECTURE}")


def apply_prior_init(model: nn.Module, seed: int) -> nn.Module:
    prior_name = str(PBB_PRIOR_TYPE).lower()
    if prior_name in ["truncated_gaussian", "trunc_gaussian", "truncnorm"]:
        return initialize_prior_truncated_gaussian(
            model,
            sigma_scale=PBB_SIGMA_PRIOR,
            truncation=2.0,
            seed=seed,
        )
    return initialize_model_with_prior(
        model,
        prior_type=prior_name,
        sigma_prior=PBB_SIGMA_PRIOR,
        seed=seed,
    )


def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_err = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            log_probs = model(x)
            loss = criterion(log_probs, y)
            pred = torch.argmax(log_probs, dim=1)
            err = (pred != y).float().mean()
            total_loss += float(loss.item())
            total_err += float(err.item())
            n_batches += 1

    return total_loss / max(n_batches, 1), total_err / max(n_batches, 1)


def run_for_beta(beta: int, seed: int, train_loader, test_loader, device: torch.device) -> Tuple[float, float, float, float]:
    model = build_model()
    model = apply_prior_init(model, seed=seed).to(device)

    criterion = PBBBoundedNLLLoss(pmin=PBB_PMIN)
    optimizer = SGLD(
        model.parameters(),
        lr=A0_BY_BETA[beta],
        sigma_gauss_prior=SIGMA_GAUSS_PRIOR_FOR_OPT,
        beta=float(beta),
        add_noise=False,
    )

    for _ in range(MAX_EPOCHS):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            log_probs = model(x)
            loss = criterion(log_probs, y)
            loss.backward()
            optimizer.step()

    train_loss, train_err = evaluate(model, train_loader, criterion, device)
    test_loss, test_err = evaluate(model, test_loader, criterion, device)
    return train_loss, test_loss, train_err, test_err


def save_results(rows):
    os.makedirs("csv_EMA", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = f"csv_EMA/PBB_ORIGINAL_ARCH_{DATASET_TYPE.upper()}_{stamp}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "beta", "train_bounded_nll", "test_bounded_nll", "train_0_1", "test_0_1"])
        writer.writerows(rows)
    return out


def run() -> None:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")

    print("=" * 80)
    print("PBB BASELINE WITH ORIGINAL PBB ARCHITECTURE (NON-LANGEVIN)")
    print(f"Dataset: {DATASET_TYPE}")
    print(f"PBB architecture: {PBB_ARCHITECTURE}")
    print(f"Betas: {BETA_VALUES}")
    print(f"Prior: {PBB_PRIOR_TYPE}, sigma={PBB_SIGMA_PRIOR}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Device: {device}")
    print("=" * 80)

    rows = []
    for seed in SEEDS:
        set_global_seed(seed)
        train_loader, test_loader = create_dataloaders(DATASET_SEED)
        for beta in BETA_VALUES:
            tr_l, te_l, tr_e, te_e = run_for_beta(beta, seed, train_loader, test_loader, device)
            rows.append([seed, beta, tr_l, te_l, tr_e, te_e])
            print(
                f"seed={seed} beta={beta} "
                f"train_loss={tr_l:.6f} test_loss={te_l:.6f} "
                f"train_0_1={tr_e:.6f} test_0_1={te_e:.6f}"
            )

    out_csv = save_results(rows)
    print(f"Saved results to: {out_csv}")


if __name__ == "__main__":
    run()
