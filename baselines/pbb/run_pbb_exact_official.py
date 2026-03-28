"""
Run an exact-official PBB training objective on this repo's dataset split.

This script uses the official PBB implementation vendored in baselines/pbb/pbb:
- pbb.models: ProbNNet4l, trainPNNet, testStochastic, computeRiskCertificates
- pbb.bounds: PBBobj

Notes:
- The official MNIST architecture outputs 10 classes. To support this repo's binary
  labels while preserving official probabilistic layers/objective, we apply a small
  2-class projection wrapper that renormalizes classes [0, 1].
- The bound is computed directly by official PBB code (risk_01), so no extra bound
  post-processing script is required for this run.
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Make vendored official package importable as "pbb".
PBB_VENDOR_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PBB_VENDOR_PARENT not in sys.path:
    sys.path.insert(0, PBB_VENDOR_PARENT)

from dataset import get_mnist_binary_dataloaders_partial_random_labels
from models import FCN1L, FCN2L, FCN3L
import pbb.bounds as pbb_bounds_module
import pbb.models as pbb_models_module
from pbb.bounds import PBBobj
from pbb.models import NNet4l, ProbLinear, ProbNNet4l, computeRiskCertificates, testPosteriorMean, testStochastic, trainPNNet


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TEST_MODE = False
SEEDS = [42]
DATASET_SEED = 42

MNIST_CLASSES = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
USE_RANDOM_LABELS = 0.0

N_TRAIN_PER_GROUP = 4000
N_TEST_PER_GROUP = 5000
BATCH_SIZE = 100
N_HIDDEN_LAYERS = 2
WIDTH = 1000

# Official PBB training knobs.
OBJECTIVE = "fquad"
PRIOR_DIST = "gaussian"
SIGMA_PRIOR = 0.03
PMIN = 1e-3
DELTA = 0.025
DELTA_TEST = 0.05
MC_SAMPLES = 1000
KL_PENALTY = 1.0
LEARNING_RATE = 0.005
MOMENTUM = 0.95
TRAIN_EPOCHS = 500 if not TEST_MODE else 3
PRINT_EVERY = 10

# For CSV compatibility with your table format (single-run baseline row).
CSV_BETA_PLACEHOLDER = 0


def _no_tqdm(iterable, *args, **kwargs):
    return iterable


# Disable vendored tqdm progress bars for cleaner logs.
pbb_models_module.tqdm = _no_tqdm
pbb_bounds_module.tqdm = _no_tqdm


class BinaryFromTenClassProb(nn.Module):
    """Wrap official 10-class ProbNNet4l into a normalized 2-class predictor."""

    def __init__(self, base: ProbNNet4l, class_groups):
        super().__init__()
        self.base = base
        if len(class_groups) != 2:
            raise ValueError("class_groups must contain exactly two groups")
        self.group0 = tuple(int(c) for c in class_groups[0])
        self.group1 = tuple(int(c) for c in class_groups[1])
        if set(self.group0) & set(self.group1):
            raise ValueError("class_groups cannot overlap")

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        logp10 = self.base(x, sample=sample, clamping=clamping, pmin=pmin)
        p10 = torch.exp(logp10)
        p0 = p10[:, self.group0].sum(dim=1, keepdim=True)
        p1 = p10[:, self.group1].sum(dim=1, keepdim=True)
        p2 = torch.cat([p0, p1], dim=1)
        p2 = p2 / torch.clamp(p2.sum(dim=1, keepdim=True), min=1e-12)
        if clamping:
            p2 = torch.clamp(p2, min=pmin)
            p2 = p2 / torch.clamp(p2.sum(dim=1, keepdim=True), min=1e-12)
        return torch.log(torch.clamp(p2, min=1e-12))

    def compute_kl(self):
        return self.base.compute_kl()


class ProbFCNBinaryPBB(nn.Module):
    """Probabilistic FCN with a single-logit head exposed as 2-class log-probs."""

    def __init__(self, rho_prior, hidden_layers=1, width=500, prior_dist="gaussian", device="cpu", init_net=None):
        super().__init__()
        self.hidden_layers = int(hidden_layers)
        if self.hidden_layers not in (1, 2, 3):
            raise ValueError("hidden_layers must be one of {1, 2, 3}")

        self.l1 = ProbLinear(
            28 * 28,
            width,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=(init_net.fc1 if init_net is not None else None),
        )
        if self.hidden_layers >= 2:
            self.l2 = ProbLinear(
                width,
                width,
                rho_prior,
                prior_dist=prior_dist,
                device=device,
                init_layer=(init_net.fc2 if init_net is not None else None),
            )
        else:
            self.l2 = None

        if self.hidden_layers >= 3:
            self.l3 = ProbLinear(
                width,
                width,
                rho_prior,
                prior_dist=prior_dist,
                device=device,
                init_layer=(init_net.fc3 if init_net is not None else None),
            )
        else:
            self.l3 = None

        out_init = None
        if init_net is not None:
            if self.hidden_layers == 1:
                out_init = init_net.fc2
            elif self.hidden_layers == 2:
                out_init = init_net.fc3
            else:
                out_init = init_net.fc4

        self.out = ProbLinear(
            width,
            1,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=out_init,
        )

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x, sample))
        if self.l2 is not None:
            x = F.relu(self.l2(x, sample))
        if self.l3 is not None:
            x = F.relu(self.l3(x, sample))
        logits = self.out(x, sample).squeeze(1)

        log_p1 = F.logsigmoid(logits)
        log_p0 = F.logsigmoid(-logits)
        p2 = torch.stack([torch.exp(log_p0), torch.exp(log_p1)], dim=1)
        if clamping:
            p2 = torch.clamp(p2, min=pmin)
            p2 = p2 / torch.clamp(p2.sum(dim=1, keepdim=True), min=1e-12)
        return torch.log(torch.clamp(p2, min=1e-12))

    def compute_kl(self):
        kl = self.l1.kl_div + self.out.kl_div
        if self.l2 is not None:
            kl = kl + self.l2.kl_div
        if self.l3 is not None:
            kl = kl + self.l3.kl_div
        return kl


def build_init_fcn(hidden_layers: int, width: int) -> nn.Module:
    if hidden_layers == 1:
        return FCN1L(input_dim=28 * 28, hidden_dim=width, output_dim=1)
    if hidden_layers == 2:
        return FCN2L(input_dim=28 * 28, hidden_dim=width, output_dim=1)
    if hidden_layers == 3:
        return FCN3L(input_dim=28 * 28, hidden_dim=width, output_dim=1)
    raise ValueError("hidden_layers must be one of {1, 2, 3}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(dataset_seed: int):
    train_loader, test_loader = get_mnist_binary_dataloaders_partial_random_labels(
        classes=MNIST_CLASSES,
        p=USE_RANDOM_LABELS,
        n_train_per_group=N_TRAIN_PER_GROUP,
        n_test_per_group=N_TEST_PER_GROUP,
        batch_size=BATCH_SIZE,
        random_seed=dataset_seed,
        normalize=True,
    )

    # Official PBB code expects class-index targets in LongTensor dtype.
    train_x, train_y = train_loader.dataset.tensors
    test_x, test_y = test_loader.dataset.tensors
    train_ds = TensorDataset(train_x, train_y.long())
    test_ds = TensorDataset(test_x, test_y.long())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # Official testStochastic preallocates using test_loader.batch_size, so keep
    # fixed full batches to avoid last-batch shape mismatch.
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    train_counts = torch.bincount(train_y.long(), minlength=2).tolist()
    test_counts = torch.bincount(test_y.long(), minlength=2).tolist()
    print(
        f"dataset_seed={dataset_seed} train_n={len(train_ds)} test_n={len(test_ds)} "
        f"train_class_counts={train_counts} test_class_counts={test_counts}"
    )
    print(
        f"loader_info train_batches={len(train_loader)} train_batch_size={train_loader.batch_size} "
        f"test_batches={len(test_loader)} test_batch_size={test_loader.batch_size}"
    )
    return train_loader, test_loader


def evaluate_error(net: nn.Module, loader: DataLoader, device: torch.device):
    net.eval()
    total = 0
    correct = 0
    bounded_nll_sum = 0.0
    bounded_scale = 1.0 / math.log(1.0 / PMIN)
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            outputs = net(data, sample=False, clamping=True, pmin=PMIN)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            bounded_nll_sum += bounded_scale * F.nll_loss(outputs, target, reduction="sum").item()

    if total == 0:
        return float("nan"), float("nan")

    return 1.0 - (correct / total), bounded_nll_sum / total


def run_one_seed(seed: int, device: torch.device):
    run_start = time.time()
    set_seed(seed)
    train_loader, test_loader = make_loaders(DATASET_SEED)

    rho_prior = math.log(math.exp(SIGMA_PRIOR) - 1.0)
    init_net = build_init_fcn(N_HIDDEN_LAYERS, WIDTH).to(device)
    net = ProbFCNBinaryPBB(
        rho_prior,
        hidden_layers=N_HIDDEN_LAYERS,
        width=WIDTH,
        prior_dist=PRIOR_DIST,
        device=str(device),
        init_net=init_net,
    ).to(device)

    n_posterior = len(train_loader.dataset)
    n_bound = len(train_loader.dataset)
    pbobj = PBBobj(
        objective=OBJECTIVE,
        pmin=PMIN,
        classes=2,
        delta=DELTA,
        delta_test=DELTA_TEST,
        mc_samples=MC_SAMPLES,
        kl_penalty=KL_PENALTY,
        device=str(device),
        n_posterior=n_posterior,
        n_bound=n_bound,
    )

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    print(
        f"seed={seed} start objective={OBJECTIVE} lr={LEARNING_RATE} momentum={MOMENTUM} "
        f"epochs={TRAIN_EPOCHS} pmin={PMIN} sigma_prior={SIGMA_PRIOR}"
    )
    print(
        f"seed={seed} model={net.__class__.__name__} arch=FCN{N_HIDDEN_LAYERS}L "
        f"width={WIDTH} out_neurons=1 device={device}"
    )
    for epoch in range(TRAIN_EPOCHS):
        trainPNNet(net, optimizer, pbobj, epoch + 1, train_loader, verbose=False)
        if (epoch + 1) % PRINT_EVERY == 0 or (epoch == 0) or (epoch + 1 == TRAIN_EPOCHS):
            elapsed = time.time() - run_start
            current_lr = optimizer.param_groups[0]["lr"]
            train_err, train_nll = evaluate_error(net, train_loader, device)
            test_err, test_nll = evaluate_error(net, test_loader, device)
            print(
                f"seed={seed} epoch={epoch + 1}/{TRAIN_EPOCHS} "
                # f"elapsed_s={elapsed:.1f} lr={current_lr:.6f} "
                f"train_0_1={train_err:.6f} test_0_1={test_err:.6f} "
                f"train_bounded_nll={train_nll:.6f} test_bounded_nll={test_nll:.6f} "
                
            )

    print(
        f"[next] seed={seed} final computeRiskCertificates over {len(train_loader)} "
        f"train batches with mc_samples={MC_SAMPLES}"
    )
    train_obj, risk_ce, risk_01, kl_over_n, loss_ce_train, loss_01_train = computeRiskCertificates(
        net,
        toolarge=True,
        pbobj=pbobj,
        device=str(device),
        train_loader=train_loader,
        whole_train=None,
    )
    print(f"[next] seed={seed} final testStochastic over {len(test_loader)} test batches")
    stch_loss, stch_err = testStochastic(net, test_loader, pbobj, device=str(device))
    post_loss, post_err = testPosteriorMean(net, test_loader, pbobj, device=str(device))

    total_elapsed = time.time() - run_start
    print(
        f"seed={seed} done elapsed_s={total_elapsed:.1f} "
        f"final_train_loss={float(loss_ce_train):.6f} final_test_loss={float(stch_loss):.6f} "
        f"final_bound_01={float(risk_01):.6f} "
        f"post_mean_loss={float(post_loss):.6f} post_mean_0_1={float(post_err):.6f}"
    )

    return {
        "seed": seed,
        "beta": CSV_BETA_PLACEHOLDER,
        "train_bounded_nll": float(loss_ce_train),
        "test_bounded_nll": float(stch_loss),
        "train_0_1": float(loss_01_train),
        "test_0_1": float(stch_err),
        "post_mean_loss": float(post_loss),
        "post_mean_0_1": float(post_err),
        "kl_over_n": float(kl_over_n),
        "kl_raw": float(kl_over_n * n_bound),
        "pbb_risk_ce_bound": float(risk_ce),
        "pbb_risk_01_bound": float(risk_01),
    }


def save_csv(rows):
    os.makedirs("csv_EMA", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = f"csv_EMA/PBB_EXACT_OFFICIAL_MNIST_{stamp}.csv"
    cols = [
        "seed",
        "beta",
        "train_bounded_nll",
        "test_bounded_nll",
        "train_0_1",
        "test_0_1",
        "post_mean_loss",
        "post_mean_0_1",
        "kl_over_n",
        "kl_raw",
        "pbb_risk_ce_bound",
        "pbb_risk_01_bound",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return out


def main() -> None:
    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

    print("=" * 80)
    print("EXACT OFFICIAL PBB RUNNER")
    print(f"Dataset: MNIST binary groups={MNIST_CLASSES}")
    print(f"Label randomization p: {USE_RANDOM_LABELS}")
    print(f"Objective: {OBJECTIVE}")
    print(f"Prior dist: {PRIOR_DIST}, sigma_prior={SIGMA_PRIOR}")
    print(f"Device: {device}")
    print("=" * 80)

    rows = [run_one_seed(seed, device) for seed in SEEDS]
    out = save_csv(rows)
    print(f"Saved exact-official PBB metrics to: {out}")


if __name__ == "__main__":
    main()
