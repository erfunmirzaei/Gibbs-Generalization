"""
Meta-learning experiment script for CIFAR-100 Gibbs generalization bound experiments.

This script samples NEW class pairs from CIFAR-100 such that each class is used at most
once across all rows already present in table_meta_learning.xlsx, runs experiments for the
requested number of additional pairs, and appends one row to the Excel table after each
pair finishes.
"""

import ast
import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from openpyxl import Workbook, load_workbook

from dataset import get_cifar100_binary_dataloaders_partial_random_labels
from training import run_beta_experiments

# =========================
# Configuration flags
# =========================
TEST_MODE = True  # Set to True for quick test, False for full experiment
USE_RANDOM_LABELS = 0  # Percentage of randomly labeled data
SEEDS = [42]  # Random seeds for stability analysis
DATASET_SEED = 42  # Seed for dataset splitting/label randomization
USE_SAME_DATASET_ACROSS_SEEDS = True  # Kept for compatibility

# Core request: choose how many NEW pairs to run (classes cannot repeat across table)
NUM_NEW_PAIRS = 7
PAIR_SAMPLING_SEED = 82

# Each pair runs:
# 1) ULA for beta in [128, 1000]
# 2) GD (noise off) to approximate beta -> infinity columns
ULA_BETA_VALUES = [128, 1000]
GD_BETA_PROXY = [1000000]

# Optional side output
SAVE_SEED_SUMMARY = False

# Excel target
EXCEL_PATH = Path("table_meta_learning.xlsx")

# Fixed CIFAR-100 setup
DATASET_TYPE = "cifar100"
ALL_CIFAR100_CLASSES = list(range(100))

# Keep same learning-rate style while limiting betas to what table needs
if TEST_MODE:
    A0 = {0: 0.01, 128: 0.01, 1000: 0.01, 1000000: 0.01}
else:
    A0 = {0: 0.01, 128: 0.01, 1000: 0.01, 1000000: 0.01}

EXCEL_HEADERS = [
    "CIFAR100 Classes",
    "Mean SAVAGE training loss at beta = 128",
    "Mean SAVAGE training loss at beta = 1000",
    "Mean SAVAGE training  loss at infinity ",
    "Mean SAVAGE test  loss at infinity ",
    "Mean 01 training  loss at infinity ",
    "Mean 01 test  loss at infinity ",
]


def set_global_seed(seed: int) -> None:
    """Set random seed across torch/numpy/python for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(classes, dataset_seed):
    """Create CIFAR-100 train/test dataloaders for a given class pair/grouping."""
    return get_cifar100_binary_dataloaders_partial_random_labels(
        classes=classes,
        p=USE_RANDOM_LABELS,
        n_train_per_group=500,
        n_test_per_group=100,
        batch_size=1000,
        random_seed=dataset_seed,
    )


def _parse_pair_cell(value):
    """Parse class pair from Excel cell like '[2,50]' -> (2, 50)."""
    if value is None:
        return None

    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            pair = tuple(sorted((int(value[0]), int(value[1]))))
            return pair
        except (TypeError, ValueError):
            return None

    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
            try:
                pair = tuple(sorted((int(parsed[0]), int(parsed[1]))))
                return pair
            except (TypeError, ValueError):
                return None

    return None


def load_used_classes_from_excel(excel_path: Path):
    """Read used pairs/classes from first column to enforce class-level uniqueness."""
    used_pairs = set()
    used_classes = set()

    if not excel_path.exists():
        return used_pairs, used_classes

    wb = load_workbook(excel_path)
    ws = wb[wb.sheetnames[0]]

    for row_idx in range(2, ws.max_row + 1):
        pair = _parse_pair_cell(ws.cell(row=row_idx, column=1).value)
        if pair is None:
            continue
        used_pairs.add(pair)
        used_classes.update(pair)

    return used_pairs, used_classes


def sample_new_disjoint_pairs(num_pairs: int, used_classes: set, rng: random.Random):
    """Sample class pairs with no class reused from previous rows or within new batch."""
    if num_pairs <= 0:
        return []

    available_classes = [c for c in ALL_CIFAR100_CLASSES if c not in used_classes]
    needed_classes = 2 * num_pairs

    if needed_classes > len(available_classes):
        raise ValueError(
            f"Not enough unused CIFAR-100 classes left: need {needed_classes}, "
            f"but only {len(available_classes)} available."
        )

    rng.shuffle(available_classes)
    selected = available_classes[:needed_classes]

    pairs = []
    for i in range(0, needed_classes, 2):
        pair = tuple(sorted((selected[i], selected[i + 1])))
        pairs.append(pair)
    return pairs


def _ensure_workbook(excel_path: Path):
    """Create workbook with expected header if file does not exist yet."""
    if excel_path.exists():
        wb = load_workbook(excel_path)
        ws = wb[wb.sheetnames[0]]
        if ws.max_row == 0:
            ws.append(EXCEL_HEADERS)
            wb.save(excel_path)
        elif ws.max_row >= 1 and all(ws.cell(1, col).value is None for col in range(1, len(EXCEL_HEADERS) + 1)):
            for col_idx, header in enumerate(EXCEL_HEADERS, start=1):
                ws.cell(row=1, column=col_idx, value=header)
            wb.save(excel_path)
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(EXCEL_HEADERS)
    wb.save(excel_path)


def _read_metrics_from_csv(csv_path: Path):
    """Read beta-indexed rows from one result CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV output not found: {csv_path}")

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("Beta"):
                continue
            beta_raw = row.get("Beta")
            try:
                beta_val = float(beta_raw)
            except (TypeError, ValueError):
                continue
            rows.append((beta_val, row))

    if not rows:
        raise ValueError(f"No valid data rows found in CSV: {csv_path}")

    rows.sort(key=lambda item: item[0])

    def find_row_for_beta(target_beta):
        for beta, row in rows:
            if abs(beta - target_beta) < 1e-9:
                return row
        return None

    row_128 = find_row_for_beta(128.0)
    row_1000 = find_row_for_beta(1000.0)

    return {
        "row_128": row_128,
        "row_1000": row_1000,
        "row_last": rows[-1][1],
    }


def _combine_metrics(ula_csv_path: Path, gd_csv_path: Path):
    """Build table row metrics from ULA (128/1000) and GD (infinity proxy)."""
    ula_rows = _read_metrics_from_csv(ula_csv_path)
    gd_rows = _read_metrics_from_csv(gd_csv_path)

    row_128 = ula_rows["row_128"]
    row_1000 = ula_rows["row_1000"]
    row_inf = gd_rows["row_last"]

    if row_128 is None:
        raise ValueError(f"Beta 128 not found in ULA CSV: {ula_csv_path}")
    if row_1000 is None:
        raise ValueError(f"Beta 1000 not found in ULA CSV: {ula_csv_path}")

    return {
        "train_128": float(row_128["EMA_BCE_Train"]),
        "train_1000": float(row_1000["EMA_BCE_Train"]),
        "train_inf": float(row_inf["EMA_BCE_Train"]),
        "test_inf": float(row_inf["EMA_BCE_Test"]),
        "train01_inf": float(row_inf["EMA_0-1_Train"]),
        "test01_inf": float(row_inf["EMA_0-1_Test"]),
    }


def append_result_to_excel(excel_path: Path, pair, metrics):
    """Append one completed pair result row to table_meta_learning.xlsx."""
    _ensure_workbook(excel_path)
    wb = load_workbook(excel_path)
    ws = wb[wb.sheetnames[0]]

    row = [
        f"[{pair[0]},{pair[1]}]",
        f"{metrics['train_128']:.4f}",
        f"{metrics['train_1000']:.4f}",
        f"{metrics['train_inf']:.4f}",
        f"{metrics['test_inf']:.4f}",
        f"{metrics['train01_inf']:.4f}",
        f"{metrics['test01_inf']:.4f}",
    ]
    ws.append(row)
    wb.save(excel_path)


def save_seed_stability_summary(seed_results, pair):
    """Optional: save compact stability analysis table across seeds for one pair."""
    if not seed_results:
        return None

    rows = []
    for result in seed_results:
        for csv_path in result["csv_paths"]:
            rows.append([result["seed"], result["dataset_seed"], csv_path])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pair_tag = f"{pair[0]}_{pair[1]}"
    summary_path = f"csv_EMA/SEED_STABILITY_CIFAR100_PAIR_{pair_tag}_{timestamp}.csv"
    with open(summary_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["seed", "dataset_seed", "result_csv_path"])
        writer.writerows(rows)

    return summary_path


def run_pair_experiment(pair):
    """Run ULA(128,1000) and GD for one pair; return both CSV paths."""
    print("\n" + "=" * 70)
    print(f"Running pair: {pair}")
    print(f"Dataset: CIFAR-100")
    print(f"ULA betas: {ULA_BETA_VALUES}, GD proxy beta: {GD_BETA_PROXY[-1]}")
    print("=" * 70)

    ula_seed_results = []
    gd_seed_results = []
    device = "cpu"  # change if needed

    for seed in SEEDS:
        dataset_seed = DATASET_SEED if USE_SAME_DATASET_ACROSS_SEEDS else seed

        print("\n" + "-" * 70)
        print(f"Running seed {seed} (dataset_seed={dataset_seed})")
        print("-" * 70)

        set_global_seed(seed)

        train_loader, test_loader = create_dataloaders(classes=list(pair), dataset_seed=dataset_seed)

        csv_paths_ula = run_beta_experiments(
            loss="SAVAGE",
            beta_values=ULA_BETA_VALUES,
            a0=A0,
            b=0.5,
            sigma_gauss_prior=5.0,
            device=device,
            n_hidden_layers=2,
            width=500,
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
            selected_classes=list(pair),
        )

        ula_seed_results.append(
            {
                "seed": seed,
                "dataset_seed": dataset_seed,
                "csv_paths": csv_paths_ula or [],
            }
        )

        csv_paths_gd = run_beta_experiments(
            loss="SAVAGE",
            beta_values=GD_BETA_PROXY,
            a0=A0,
            b=0.5,
            sigma_gauss_prior=5.0,
            device=device,
            n_hidden_layers=2,
            width=500,
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
            add_noise=False,
            sgld_num=1,
            annealed=False,
            min_steps_first_beta=4000,
            seed=seed,
            selected_classes=list(pair),
        )

        gd_seed_results.append(
            {
                "seed": seed,
                "dataset_seed": dataset_seed,
                "csv_paths": csv_paths_gd or [],
            }
        )

    if SAVE_SEED_SUMMARY:
        summary_path = save_seed_stability_summary(ula_seed_results + gd_seed_results, pair)
        if summary_path is not None:
            print(f"Seed stability summary saved to: {summary_path}")

    ula_csvs = [path for item in ula_seed_results for path in item["csv_paths"]]
    gd_csvs = [path for item in gd_seed_results for path in item["csv_paths"]]
    if not ula_csvs:
        raise RuntimeError(f"No ULA CSV output generated for pair {pair}.")
    if not gd_csvs:
        raise RuntimeError(f"No GD CSV output generated for pair {pair}.")

    return Path(ula_csvs[-1]), Path(gd_csvs[-1])


def main():
    if not SEEDS:
        raise ValueError("SEEDS must contain at least one seed value")

    _ensure_workbook(EXCEL_PATH)
    used_pairs, used_classes = load_used_classes_from_excel(EXCEL_PATH)

    print(f"Existing rows with valid pairs: {len(used_pairs)}")
    print(f"Used CIFAR-100 classes so far: {len(used_classes)}")

    rng = random.Random(PAIR_SAMPLING_SEED)
    new_pairs = sample_new_disjoint_pairs(NUM_NEW_PAIRS, used_classes, rng)

    print(f"Planned new pairs: {new_pairs}")
    print("Mode: ULA at beta 128/1000 + GD for infinity columns")

    for index, pair in enumerate(new_pairs, start=1):
        print(f"\n>>> Pair {index}/{len(new_pairs)}: {pair}")

        if pair in used_pairs:
            print(f"Skipping already existing pair in table: {pair}")
            continue

        try:
            ula_csv_path, gd_csv_path = run_pair_experiment(pair)
            metrics = _combine_metrics(ula_csv_path, gd_csv_path)
            append_result_to_excel(EXCEL_PATH, pair, metrics)

            used_pairs.add(pair)
            used_classes.update(pair)

            print(f"✅ Pair completed and appended to {EXCEL_PATH}: {pair}")

        except Exception as exc:
            print(f"❌ Pair failed ({pair}): {exc}")
            # Continue with next pair while preserving already written rows
    #         continue

    print("\n" + "=" * 70)
    print("META-LEARNING PAIR RUN COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
