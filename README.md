# Gibbs Generalization Bound Experiments

Research code for PAC-Bayesian/Gibbs generalization experiments with Langevin-style samplers. The repository contains the experiment drivers, model definitions, losses, plotting scripts, PBB baselines, and selected CSV/plot artifacts used during paper development.

This cleanup keeps the original script-driven workflow intact for reproducibility. Most experiment settings are still module-level constants in the entry-point scripts, so reproducing a specific run means checking the script configuration and the corresponding CSV filename.

## Repository Layout

- `main.py` - binary-classification experiments for MNIST, CIFAR-10, CIFAR-100, SVHN, and synthetic data.
- `master.py` - multiclass MNIST experiment driver.
- `training.py`, `training_multiclass.py` - training loops, SGLD/ULA orchestration, CSV export, and checkpoint hooks.
- `dataset.py`, `multiclass_dataset_functions.py` - dataset loaders and label-randomization utilities.
- `models.py`, `pbb_models.py` - neural network architectures.
- `losses.py`, `sgld.py`, `mala.py`, `new_MALA.py` - losses and samplers/optimizers.
- `plot.py`, `plot_q_bounds.py`, `table_MNIST.py`, `table_CIFAR.py` - analysis, plotting, and table generation.
- `baselines/pbb/` - PBB baseline implementations and helper scripts.
- `csv_EMA/` - tracked experiment CSV artifacts.
- `newplots/` - tracked generated figures.
- `checkpoints/` - tracked lightweight checkpoints used by existing runs.

Local downloaded datasets are expected under `data/`, but that directory is ignored by git.

## Installation

Create a clean environment with either Conda:

```bash
conda env create -f environment.yml
conda activate gibbs-generalization
```

or pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU runs, install the PyTorch build that matches your CUDA version if the default package resolver does not pick it automatically.

## Quick Smoke Test

Before launching a full run, set `TEST_MODE = True` in the relevant entry-point script and run:

```bash
python main.py
```

The scripts create `csv_EMA/` and `newplots/` outputs as needed. Existing tracked artifacts are preserved; new exploratory outputs are ignored by default to avoid accidental large commits.

## Running Experiments

Binary experiments are configured in `main.py`:

```bash
DATASET_TYPE=mnist USE_RANDOM_LABELS=0 SEEDS=42 python main.py
DATASET_TYPE=mnist USE_RANDOM_LABELS=1 SEEDS=42 python main.py
```

Multiclass MNIST experiments are configured in `master.py`:

```bash
USE_RANDOM_LABELS=0 python master.py
USE_RANDOM_LABELS=1 python master.py
```

The shell helpers `run_true_and_random.sh`, `run_master_both_labels.sh`, and `run_cifar10_random_seeds_*.sh` capture common runs. Check the Conda environment name inside each script before using it on a new machine.

## Outputs

CSV files are named with experiment metadata:

```text
{Dataset}{LabelType}{NetworkType}W{Width}{Algorithm}{Steps}LR{LearningRate}{LossType}.csv
```

Examples:

- `MCL2W1000SGLD8kLR001BBCE.csv` - MNIST, correct labels, 2-layer FCN, width 1000, SGLD, 8k steps, learning rate 0.01, BBCE loss.
- `CRL2W1500SGLD8kLR0005BBCE.csv` - CIFAR-style binary setup, random labels, 2-layer FCN, width 1500, SGLD, 8k steps, learning rate 0.0005, BBCE loss.

Plotting and table scripts consume files in `csv_EMA/` and write figures to `newplots/`.

## Reproducibility Notes

- Seeds are set through script-level `SEEDS` and `DATASET_SEED` values. Some scripts also accept environment overrides such as `SEEDS`, `DATASET_TYPE`, and `USE_RANDOM_LABELS`.
- CUDA determinism is enabled where seeds are set, but exact GPU reproducibility can still depend on hardware, driver, CUDA, and PyTorch versions.
- Torchvision datasets are downloaded locally into `data/` and are not committed.
- Full experiments can be expensive. Use `TEST_MODE = True` first to verify installation and paths.
- The repository intentionally avoids a broad package refactor in order to keep paths, filenames, and experiment behavior close to the accepted-paper version.

## Citation

If this repository supports your work, please cite the associated ICML paper. Replace this placeholder with the official proceedings citation once available.

```bibtex
@inproceedings{mirzaei2026gibbs,
  title     = {Gibbs Generalization Bound Experiments},
  author    = {Mirzaei, Erfan and collaborators},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2026}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
