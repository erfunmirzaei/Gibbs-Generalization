"""
Run a non-Langevin PBB-style baseline using this repo's architecture family.

This script does not modify core training files. It builds a small standalone loop that:
- uses your dataset construction settings from main.py,
- uses your architecture family (FCN/LeNet/VGG),
- applies PBB-style prior initialization,
- optimizes with deterministic SGLD updates (add_noise=False),
- trains with bounded PBB NLL via pmin.
"""

from __future__ import annotations

import csv
import os
import random
import subprocess
import sys
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from models import FCN1L, FCN2L, FCN3L, LeNet5, CNNet4l, VGG16_CIFAR
from pbb_prior import initialize_model_with_prior
from pbb_truncated_prior import initialize_prior_truncated_gaussian
from sgld import SGLD

# 1. MCL2W1000SGLD8kLR001BBCE
# 2. CCL2W1500SGLD8kLR0005BBCE
# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
TEST_MODE = False
SEEDS = [42]
DATASET_SEED = 42

DATASET_TYPE = "mnist"  # 'mnist' | 'cifar10' | 'cifar100' | 'synth'
# Label randomization probability p used by *_partial_random_labels loaders.
# p=0.0 => clean labels, p=1.0 => fully random labels (near-chance training).
USE_RANDOM_LABELS = 0.0
MNIST_CLASSES = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]] 
CIFAR10_CLASSES = [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]]
CIFAR100_CLASSES = [55, 88]

# Use your architecture family exactly.
N_HIDDEN_LAYERS = 2  # 1 | 2 | 3 | 'L' for MNIST LeNet5 | 'C' for MNIST CNNet4l | 'V' for CIFAR VGG16
WIDTH = 1000

# PBB objective/prior knobs.
PBB_PMIN = 1e-5
PBB_PRIOR_TYPE = "truncated_gaussian"  # 'gaussian' | 'laplace' | 'truncated_gaussian'
PBB_SIGMA_PRIOR = 0.03

# SGLD deterministic update knobs (non-Langevin baseline).
SIGMA_GAUSS_PRIOR_FOR_OPT = 5.0
BETA_VALUES = [16000]
A0_BY_BETA = {16000: 0.01}
MAX_EPOCHS = 500 if not TEST_MODE else 5

# Progress logging during training (set to 0 to disable per-epoch logging).
LOG_EVERY_EPOCHS = 50

# Optional: automatically compute PBB bounds for the generated CSV at the end.
# This keeps the main workflow unchanged and only augments PBB outputs when enabled.
AUTO_COMPUTE_BOUNDS = True
BOUND_DELTA = 0.05
BOUND_MC_SAMPLES = 1000
BOUND_APPEND_TO_TRUE_CSV = True
BOUND_KL_VALUE: Optional[float] = None
BOUND_KL_CSV: Optional[str] = None
BOUND_KL_COLUMN: Optional[str] = None


def _param_l2_norm(model: nn.Module) -> float:
    s = 0.0
    with torch.no_grad():
        for p in model.parameters():
            s += float((p.detach() ** 2).sum().item())
    return float(s ** 0.5)


def _param_l2_delta_from(model: nn.Module, ref_params) -> float:
    s = 0.0
    with torch.no_grad():
        for p, r in zip(model.parameters(), ref_params):
            d = p.detach() - r
            s += float((d ** 2).sum().item())
    return float(s ** 0.5)


class BinaryToTwoClassLogProb(nn.Module):
    """Wrap a binary-logit model and expose 2-class log-probabilities."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x).squeeze(-1)
        log_p1 = F.logsigmoid(logits)
        log_p0 = F.logsigmoid(-logits)
        return torch.stack([log_p0, log_p1], dim=1)


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
            n_train_per_group=4000,
            n_test_per_group=5000,
            batch_size=100,
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


def build_base_model(dataset_type: str, n_hidden_layers, width: int) -> nn.Module:
    if dataset_type == "mnist":
        if n_hidden_layers == 1:
            return FCN1L(input_dim=28 * 28, hidden_dim=width, output_dim=1)
        if n_hidden_layers == 2:
            return FCN2L(input_dim=28 * 28, hidden_dim=width, output_dim=1)
        if n_hidden_layers == 3:
            return FCN3L(input_dim=28 * 28, hidden_dim=width, output_dim=1)
        if n_hidden_layers == "L":
            return LeNet5(num_classes=1)
        if n_hidden_layers == "C":
            return CNNet4l(num_classes=1)

    if dataset_type in ("cifar10", "cifar100"):
        if n_hidden_layers == 1:
            return FCN1L(input_dim=3 * 32 * 32, hidden_dim=width, output_dim=1)
        if n_hidden_layers == 2:
            return FCN2L(input_dim=3 * 32 * 32, hidden_dim=width, output_dim=1)
        if n_hidden_layers == 3:
            return FCN3L(input_dim=3 * 32 * 32, hidden_dim=width, output_dim=1)
        if n_hidden_layers == "V":
            return VGG16_CIFAR(num_classes=1)

    if dataset_type == "synth":
        if n_hidden_layers == 1:
            return FCN1L(input_dim=4, hidden_dim=width, output_dim=1)
        if n_hidden_layers == 2:
            return FCN2L(input_dim=4, hidden_dim=width, output_dim=1)
        if n_hidden_layers == 3:
            return FCN3L(input_dim=4, hidden_dim=width, output_dim=1)

    raise ValueError(f"Unsupported architecture setting: dataset={dataset_type}, n_hidden_layers={n_hidden_layers}")


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
    base_model = build_base_model(DATASET_TYPE, N_HIDDEN_LAYERS, WIDTH)
    wrapped = BinaryToTwoClassLogProb(base_model)
    wrapped = apply_prior_init(wrapped, seed=seed).to(device)

    criterion = PBBBoundedNLLLoss(pmin=PBB_PMIN)
    lr = A0_BY_BETA[beta]
    optimizer = SGLD(
        wrapped.parameters(),
        lr=lr,
        sigma_gauss_prior=SIGMA_GAUSS_PRIOR_FOR_OPT,
        beta=float(beta),
        add_noise=False,
    )
    ref_params = [p.detach().clone() for p in wrapped.parameters()]

    # Show starting metrics so training progress is explicit.
    init_train_loss, init_train_err = evaluate(wrapped, train_loader, criterion, device)
    init_test_loss, init_test_err = evaluate(wrapped, test_loader, criterion, device)
    print(
        f"seed={seed} beta={beta} epoch=0/{MAX_EPOCHS} "
        f"train_loss={init_train_loss:.6f} test_loss={init_test_loss:.6f} "
        f"train_0_1={init_train_err:.6f} test_0_1={init_test_err:.6f} "
        f"param_l2={_param_l2_norm(wrapped):.6f} param_delta_l2=0.000000"
    )

    for epoch in range(MAX_EPOCHS):
        wrapped.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            log_probs = wrapped(x)
            loss = criterion(log_probs, y)
            loss.backward()
            optimizer.step()

        if LOG_EVERY_EPOCHS > 0 and (((epoch + 1) % LOG_EVERY_EPOCHS == 0) or (epoch + 1 == MAX_EPOCHS)):
            tr_l, tr_e = evaluate(wrapped, train_loader, criterion, device)
            te_l, te_e = evaluate(wrapped, test_loader, criterion, device)
            d_tr_l = tr_l - init_train_loss
            d_te_l = te_l - init_test_loss
            d_tr_e = tr_e - init_train_err
            d_te_e = te_e - init_test_err
            print(
                f"seed={seed} beta={beta} epoch={epoch + 1}/{MAX_EPOCHS} "
                f"train_loss={tr_l:.6f} test_loss={te_l:.6f} "
                f"train_0_1={tr_e:.6f} test_0_1={te_e:.6f} "
                f"d_train_loss={d_tr_l:+.6e} d_test_loss={d_te_l:+.6e} "
                f"d_train_0_1={d_tr_e:+.6e} d_test_0_1={d_te_e:+.6e} "
                f"param_l2={_param_l2_norm(wrapped):.6f} "
                f"param_delta_l2={_param_l2_delta_from(wrapped, ref_params):.6f}"
            )

    train_loss, train_err = evaluate(wrapped, train_loader, criterion, device)
    test_loss, test_err = evaluate(wrapped, test_loader, criterion, device)
    return train_loss, test_loss, train_err, test_err


def save_results(rows):
    os.makedirs("csv_EMA", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"csv_EMA/PBB_OURS_ARCH_{DATASET_TYPE.upper()}_{stamp}.csv"
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "beta", "train_bounded_nll", "test_bounded_nll", "train_0_1", "test_0_1"])
        writer.writerows(rows)
    return file_path


def maybe_compute_bounds(out_csv: str, sample_size: int) -> None:
    if not AUTO_COMPUTE_BOUNDS:
        print("Auto bounds: disabled (set AUTO_COMPUTE_BOUNDS=True to enable).")
        return

    if BOUND_KL_CSV is None and BOUND_KL_VALUE is None:
        print(
            "Auto bounds skipped: provide BOUND_KL_CSV or BOUND_KL_VALUE "
            "to compute official PBB bounds."
        )
        return

    bounds_script = os.path.join(REPO_ROOT, "baselines", "pbb", "compute_test_error_bounds_from_csv.py")
    cmd = [
        sys.executable,
        bounds_script,
        "--true-csv",
        out_csv,
        "--sample-size",
        str(int(sample_size)),
        "--delta",
        str(BOUND_DELTA),
        "--mc-samples",
        str(int(BOUND_MC_SAMPLES)),
    ]

    if BOUND_KL_CSV is not None:
        cmd += ["--kl-csv", BOUND_KL_CSV]
    if BOUND_KL_COLUMN is not None:
        cmd += ["--kl-column", BOUND_KL_COLUMN]
    if BOUND_KL_VALUE is not None:
        cmd += ["--kl-value", str(BOUND_KL_VALUE)]
    if BOUND_APPEND_TO_TRUE_CSV:
        cmd += ["--append-to-true-csv"]

    print("Running bounds command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")

    print("=" * 80)
    print("PBB BASELINE WITH OUR ARCHITECTURE (NON-LANGEVIN)")
    print(f"Dataset: {DATASET_TYPE}")
    print(f"Label randomization p: {USE_RANDOM_LABELS}")
    print(f"Betas: {BETA_VALUES}")
    print("Optimizer: SGLD(add_noise=False), Loss: PBBBoundedNLLLoss")
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
    sample_size = len(train_loader.dataset)
    maybe_compute_bounds(out_csv=out_csv, sample_size=sample_size)


if __name__ == "__main__":
    main()
