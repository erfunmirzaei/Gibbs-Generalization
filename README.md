# Gibbs Generalization Bounds

This repository contains the code used for the ICML paper experiments on Gibbs/PAC-Bayesian generalization bounds with Langevin-style sampling.

The code is intentionally still script-based. The main goal of this public version is to keep the experiments reproducible without doing a large refactor that might change behavior.

## What To Run

- `main.py` runs the binary experiments for MNIST, CIFAR-10, CIFAR-100, SVHN, and synthetic data.
- `master.py` runs the 10-class MNIST experiments.
- `plot.py`, `plot_q_bounds.py`, `table_MNIST.py`, and `table_CIFAR.py` regenerate plots and tables from saved CSV files.
- `baselines/pbb/` contains the PBB baseline code.
- `scripts/` contains a few shell helpers for common runs.

Most experiment settings are near the top of `main.py` and `master.py`.

## Setup

With Conda:

```bash
conda env create -f environment.yml
conda activate gibbs-generalization
```

or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA runs, install the PyTorch build that matches your machine.

## Quick Check

Set `TEST_MODE = True` in the script you want to check, then run:

```bash
python main.py
```

For the multiclass MNIST path:

```bash
python master.py
```

## Common Runs

`main.py` supports a few environment-variable overrides:

```bash
DATASET_TYPE=mnist USE_RANDOM_LABELS=0 SEEDS=42 python main.py
DATASET_TYPE=mnist USE_RANDOM_LABELS=1 SEEDS=42 python main.py
```

For 10-class MNIST:

```bash
USE_RANDOM_LABELS=0 python master.py
USE_RANDOM_LABELS=1 python master.py
```

## Files And Outputs

- `dataset.py`, `multiclass_dataset_functions.py`: dataset loading and label randomization.
- `models.py`: model definitions.
- `losses.py`: bounded losses and evaluation losses.
- `sgld.py`, `mala.py`, `new_MALA.py`: samplers/optimizers.
- `training.py`, `training_multiclass.py`: training loops and CSV export.
- `csv_EMA/`: saved CSV results.
- `newplots/`: saved plots.
- `checkpoints/`: lightweight checkpoints from existing runs.
- `notebooks/`: optional notebooks.

Downloaded datasets live under `data/`, which is ignored by git.

## Reproducibility

Seeds are set in the scripts through `SEEDS` and `DATASET_SEED`. CUDA determinism is enabled where seeds are set, but exact GPU reproducibility can still depend on hardware, drivers, CUDA, and PyTorch versions.

New generated CSVs, plots, logs, and checkpoints are ignored by default. Existing tracked artifacts are kept so the public repository still contains the saved results used during development.

## Citation

Please cite the associated ICML paper if you use this code. The exact proceedings citation should replace this placeholder once available:

```bibtex
@inproceedings{mirzaei2026gibbs,
  title     = {Gibbs Generalization Bounds},
  author    = {Mirzaei, Erfan and collaborators},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2026}
}
```

## License

MIT. See `LICENSE`.
