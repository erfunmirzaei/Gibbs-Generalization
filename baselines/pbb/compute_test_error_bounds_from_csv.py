"""
Compute official PBB-style test-error certificates from CSV files.

This script uses the same inversion form as the official PBB repository:

    bound_01 = inv_kl(train_01, (KL + ln(2*sqrt(n)/delta)) / n)

where:
- train_01 is the empirical 0-1 training error,
- KL is the posterior-prior KL divergence (raw, not normalized by n),
- n is the sample size used for the bound,
- delta is the confidence parameter.

Accepted metric CSV fields (for train/test errors, beta, sample size):
- New baseline format:
    seed,beta,train_bounded_nll,test_bounded_nll,train_0_1,test_0_1
- Legacy csv_EMA format:
    Sample_size,Beta,...,EMA_0-1_Train,EMA_0-1_Test

KL can be provided via:
1) A KL column in the same CSV (auto-detected or via --kl-column),
2) A separate KL CSV via --kl-csv with at least beta+KL (and optionally seed),
3) A fixed KL value for all rows via --kl-value.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def ln(x: float) -> float:
    return math.log(x)


def kl_bernoulli(p: float, q: float) -> float:
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    q = min(max(q, eps), 1.0 - eps)
    return p * ln(p / q) + (1.0 - p) * ln((1.0 - p) / (1.0 - q))


def invert_kl(p: float, kl_val: float, tol: float = 1e-6) -> float:
    """Find max q in [p,1) such that KL(p||q) <= kl_val."""
    l, u = p, 1.0 - 1e-12
    r = 0.5 * (l + u)
    while (u - l) > tol:
        if kl_bernoulli(p, r) < kl_val:
            l = r
        else:
            u = r
        r = 0.5 * (l + u)
    return r


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _pick_first_float(row: Dict[str, str], keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in row:
            v = _safe_float(row.get(key))
            if v is not None:
                return v
    return None


def _parse_seed(raw_seed: Optional[str]) -> int:
    if raw_seed in (None, ""):
        return -1
    try:
        return int(raw_seed)
    except Exception:
        return -1


def read_rows(
    csv_path: str,
    default_sample_size: Optional[int] = None,
    kl_column: Optional[str] = None,
) -> List[Dict[str, float]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    kl_keys = [
        "kl_divergence",
        "KL_divergence",
        "kl",
        "KL",
        "kl_raw",
        "KL_raw",
    ]
    if kl_column:
        kl_keys = [kl_column]

    rows: List[Dict[str, float]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            beta = _pick_first_float(raw, ["beta", "Beta"])
            if beta is None:
                continue

            train_01 = _pick_first_float(raw, ["train_0_1", "EMA_0-1_Train", "0-1_Train"])
            test_01 = _pick_first_float(raw, ["test_0_1", "EMA_0-1_Test", "0-1_Test"])
            sample_size = _pick_first_float(raw, ["Sample_size"])
            kl_raw = _pick_first_float(raw, kl_keys)

            if sample_size is None and default_sample_size is not None:
                sample_size = float(default_sample_size)
            if sample_size is None:
                raise ValueError(
                    "Sample size is missing in CSV and --sample-size was not provided."
                )
            if train_01 is None:
                raise ValueError(
                    "Could not find required train 0-1 error column."
                )

            seed = _parse_seed(raw.get("seed"))
            rows.append(
                {
                    "seed": float(seed),
                    "beta": float(beta),
                    "train_01": float(train_01),
                    "test_01": float(test_01) if test_01 is not None else float("nan"),
                    "sample_size": float(sample_size),
                    "kl_raw": float(kl_raw) if kl_raw is not None else float("nan"),
                }
            )

    if not rows:
        raise ValueError(f"No usable rows found in {csv_path}")
    return rows


def _key_seed_beta(seed: int, beta: float) -> Tuple[int, float]:
    return (seed, round(float(beta), 12))


def _key_beta(beta: float) -> float:
    return round(float(beta), 12)


def read_kl_map(kl_csv: str, kl_column: Optional[str]) -> Tuple[Dict[Tuple[int, float], float], Dict[float, float]]:
    if not os.path.exists(kl_csv):
        raise FileNotFoundError(f"KL CSV not found: {kl_csv}")

    kl_keys = [
        "kl_divergence",
        "KL_divergence",
        "kl",
        "KL",
        "kl_raw",
        "KL_raw",
    ]
    if kl_column:
        kl_keys = [kl_column]

    by_seed_beta: Dict[Tuple[int, float], float] = {}
    by_beta: Dict[float, float] = {}

    with open(kl_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            beta = _pick_first_float(raw, ["beta", "Beta"])
            kl_raw = _pick_first_float(raw, kl_keys)
            if beta is None or kl_raw is None:
                continue

            seed = _parse_seed(raw.get("seed"))
            by_seed_beta[_key_seed_beta(seed, beta)] = float(kl_raw)
            by_beta[_key_beta(beta)] = float(kl_raw)

    return by_seed_beta, by_beta


def attach_kl(
    rows: List[Dict[str, float]],
    kl_csv: Optional[str],
    kl_column: Optional[str],
    kl_value: Optional[float],
    kl_is_per_sample: bool,
) -> List[Dict[str, float]]:
    if kl_value is not None:
        for r in rows:
            r["kl_raw"] = float(kl_value)

    if kl_csv is not None:
        by_seed_beta, by_beta = read_kl_map(kl_csv, kl_column)
        for r in rows:
            seed = int(r["seed"])
            beta = r["beta"]
            k1 = _key_seed_beta(seed, beta)
            k2 = _key_beta(beta)
            if k1 in by_seed_beta:
                r["kl_raw"] = by_seed_beta[k1]
            elif k2 in by_beta:
                r["kl_raw"] = by_beta[k2]

    missing = [r for r in rows if math.isnan(r["kl_raw"])]
    if missing:
        raise ValueError(
            "Missing KL values. Provide KL via a CSV column (or --kl-column), "
            "or --kl-csv, or --kl-value."
        )

    if kl_is_per_sample:
        for r in rows:
            r["kl_raw"] *= r["sample_size"]

    return rows


def bounds_for_rows(rows: List[Dict[str, float]], delta: float, mc_samples: int) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for r in rows:
        n = r["sample_size"]
        kl_raw = r["kl_raw"]
        budget = (kl_raw + ln(2.0 * math.sqrt(n) / delta)) / n
        # Matches PBB's compute_final_stats_risk() default pipeline:
        # empirical_risk_01 = inv_kl(error_01, ln(2/delta_test)/mc_samples)
        # risk_01 = inv_kl(empirical_risk_01, (KL + ln(2*sqrt(n)/delta_test))/n)
        empirical_01 = invert_kl(r["train_01"], ln(2.0 / delta) / float(mc_samples))
        bound = invert_kl(empirical_01, budget)
        out.append(
            {
                "seed": int(r["seed"]),
                "beta": r["beta"],
                "sample_size": n,
                "train_01": r["train_01"],
                "test_01": r["test_01"],
                "kl_raw": kl_raw,
                "pbb_empirical_01": empirical_01,
                "kl_budget": budget,
                "predicted_test_01_bound": bound,
                "delta": delta,
                "mc_samples": int(mc_samples),
            }
        )

    out.sort(key=lambda x: (x["seed"], x["beta"]))
    return out


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [
        "seed",
        "beta",
        "sample_size",
        "train_01",
        "test_01",
        "kl_raw",
        "pbb_empirical_01",
        "kl_budget",
        "predicted_test_01_bound",
        "delta",
        "mc_samples",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_beta(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    acc: Dict[float, Dict[str, float]] = defaultdict(
        lambda: {
            "beta": 0.0,
            "train_01": 0.0,
            "test_01": 0.0,
            "kl_raw": 0.0,
            "predicted_test_01_bound": 0.0,
            "pbb_empirical_01": 0.0,
            "count": 0.0,
        }
    )

    for r in rows:
        b = r["beta"]
        item = acc[b]
        item["beta"] = b
        item["train_01"] += r["train_01"]
        if not math.isnan(r["test_01"]):
            item["test_01"] += r["test_01"]
        item["kl_raw"] += r["kl_raw"]
        item["pbb_empirical_01"] += r["pbb_empirical_01"]
        item["predicted_test_01_bound"] += r["predicted_test_01_bound"]
        item["count"] += 1.0

    out = []
    for b in sorted(acc.keys()):
        item = acc[b]
        c = item["count"]
        out.append(
            {
                "beta": b,
                "mean_train_01": item["train_01"] / c,
                "mean_test_01": item["test_01"] / c,
                "mean_kl_raw": item["kl_raw"] / c,
                "mean_pbb_empirical_01": item["pbb_empirical_01"] / c,
                "mean_predicted_test_01_bound": item["predicted_test_01_bound"] / c,
                "num_rows": int(c),
            }
        )
    return out


def write_summary(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        raise ValueError("No summary rows to write.")
    cols = [
        "beta",
        "mean_train_01",
        "mean_test_01",
        "mean_kl_raw",
        "mean_pbb_empirical_01",
        "mean_predicted_test_01_bound",
        "num_rows",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute official PBB-style test-error bounds from CSV results."
    )
    parser.add_argument("--true-csv", required=True, help="Path to CSV with train/test metrics.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size if CSV does not include it.",
    )
    parser.add_argument("--delta", type=float, default=0.01, help="Confidence parameter delta.")
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=1000,
        help="Monte Carlo samples used in PBB default empirical risk correction.",
    )
    parser.add_argument(
        "--kl-column",
        default=None,
        help="KL column name if different from default candidates.",
    )
    parser.add_argument(
        "--kl-csv",
        default=None,
        help="Optional CSV providing KL values keyed by beta (and optionally seed).",
    )
    parser.add_argument(
        "--kl-value",
        type=float,
        default=None,
        help="Optional fixed raw KL value used for all rows.",
    )
    parser.add_argument(
        "--kl-is-per-sample",
        action="store_true",
        help="Interpret provided KL values as KL/n and multiply by sample size.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path. Default: csv_EMA/PBB_BOUNDS_<timestamp>.csv",
    )
    parser.add_argument(
        "--append-to-true-csv",
        action="store_true",
        help="Append predicted_test_01_bound (and KL/budget details) into --true-csv itself.",
    )
    return parser.parse_args()


def append_bounds_to_csv(
    csv_path: str,
    bound_rows: List[Dict[str, float]],
    mc_samples: int,
    delta: float,
) -> None:
    bound_map: Dict[Tuple[int, float], Dict[str, float]] = {}
    for row in bound_rows:
        key = (int(row["seed"]), round(float(row["beta"]), 12))
        bound_map[key] = row

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        src_rows = list(reader)
        src_cols = list(reader.fieldnames or [])

    extra_cols = [
        "KL_raw",
        "PBB_empirical_01_default",
        "PBB_kl_budget_default",
        "PBB_predicted_test_01_bound_default",
        "PBB_delta_default",
        "PBB_mc_samples_default",
    ]
    for c in extra_cols:
        if c not in src_cols:
            src_cols.append(c)

    def parse_seed(raw_seed: Optional[str]) -> int:
        if raw_seed in (None, ""):
            return -1
        try:
            return int(raw_seed)
        except Exception:
            return -1

    for raw in src_rows:
        beta = _pick_first_float(raw, ["beta", "Beta"])
        if beta is None:
            continue
        seed = parse_seed(raw.get("seed"))
        key = (seed, round(float(beta), 12))
        if key not in bound_map:
            key = (-1, round(float(beta), 12))
        row = bound_map.get(key)
        if row is None:
            continue

        raw["KL_raw"] = f"{row['kl_raw']:.12g}"
        raw["PBB_empirical_01_default"] = f"{row['pbb_empirical_01']:.12g}"
        raw["PBB_kl_budget_default"] = f"{row['kl_budget']:.12g}"
        raw["PBB_predicted_test_01_bound_default"] = f"{row['predicted_test_01_bound']:.12g}"
        raw["PBB_delta_default"] = f"{delta:.12g}"
        raw["PBB_mc_samples_default"] = str(int(mc_samples))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=src_cols)
        writer.writeheader()
        writer.writerows(src_rows)


def main() -> None:
    args = parse_args()

    rows = read_rows(
        args.true_csv,
        default_sample_size=args.sample_size,
        kl_column=args.kl_column,
    )
    rows = attach_kl(
        rows,
        kl_csv=args.kl_csv,
        kl_column=args.kl_column,
        kl_value=args.kl_value,
        kl_is_per_sample=args.kl_is_per_sample,
    )

    bound_rows = bounds_for_rows(rows, delta=args.delta, mc_samples=args.mc_samples)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = args.out_csv or os.path.join("csv_EMA", f"PBB_BOUNDS_{stamp}.csv")
    out_summary = out_csv.replace(".csv", "_SUMMARY.csv")

    write_csv(out_csv, bound_rows)
    write_summary(out_summary, summarize_by_beta(bound_rows))

    if args.append_to_true_csv:
        append_bounds_to_csv(
            csv_path=args.true_csv,
            bound_rows=bound_rows,
            mc_samples=args.mc_samples,
            delta=args.delta,
        )

    print("Formula: empirical_01=inv_kl(train_01, ln(2/delta)/mc_samples); bound=inv_kl(empirical_01, (KL + ln(2*sqrt(n)/delta))/n)")
    print(f"Saved detailed bounds to: {out_csv}")
    print(f"Saved beta summary to: {out_summary}")
    if args.append_to_true_csv:
        print(f"Appended default PBB bound columns into: {args.true_csv}")


if __name__ == "__main__":
    main()
