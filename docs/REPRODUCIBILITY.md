# Reproducibility Guide

This guide documents the current script-based workflow without changing the experiment implementation.

## Environment

Use one of the repository dependency files:

```bash
conda env create -f environment.yml
conda activate gibbs-generalization
```

or:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you run on CUDA, install the PyTorch wheel or Conda package that matches your CUDA runtime.

## Data

Datasets are loaded through `torchvision` and stored locally under `data/`. The directory is ignored by git because the datasets are externally available and large.

## Main Entry Points

- `main.py` runs binary experiments.
- `master.py` runs multiclass MNIST experiments.
- `baselines/pbb/run_pbb_*.py` run PBB baselines.
- `plot.py`, `plot_q_bounds.py`, `table_MNIST.py`, and `table_CIFAR.py` regenerate analysis artifacts from CSV files.
- `scripts/` contains shell helpers for common runs.

## Suggested Verification Sequence

1. Create the environment.
2. Set `TEST_MODE = True` in the entry-point script you want to verify.
3. Run the script once on CPU or GPU.
4. Confirm that a CSV appears in `csv_EMA/`.
5. Run the relevant plotting/table script against that CSV.
6. Restore the paper configuration before launching a full run.

## Configuration Surface

Most settings are module-level constants in the scripts. Important values include:

- `DATASET_TYPE`
- `USE_RANDOM_LABELS`
- `SEEDS`
- `DATASET_SEED`
- `TEST_MODE`
- `LOSS_FUNCTION` or `loss`
- `beta_values`
- `a0`
- network depth/width settings
- sampler settings such as `add_noise`, `sgld_num`, and `annealed`

Some settings in `main.py` can also be overridden with environment variables:

```bash
DATASET_TYPE=mnist USE_RANDOM_LABELS=0 SEEDS=42,52 python main.py
```

## Artifact Policy

The checked-in `csv_EMA/`, `newplots/`, and `checkpoints/` files are preserved as paper-development artifacts. New generated CSV, plot, log, and checkpoint files are ignored by default so exploratory reruns do not accidentally enter the public history.

For a camera-ready or archival release, consider tagging the exact commit used for the paper and adding a small table that maps each paper figure/table to the CSV files used to generate it.
