# Gibbs Generalization Bounds

This repository contains the code used for the paper, "Generalization of Gibbs and Langevin Monte Carlo Algorithms in the Interpolation Regime" by Andreas Maurer, Erfan Mirzaei, and Massimiliano Pontil. The paper is available on arXiv at https://arxiv.org/abs/2510.06028. The main focus of the paper is on generalization bounds for Gibbs and Langevin Monte Carlo algorithms in the interpolation regime, where the model can fit the training data perfectly. The experiments in the paper demonstrate the theoretical results and provide insights into the behavior of these algorithms under different conditions. 

The main goal of this public version is to keep the experiments reproducible. For any questions or collaboration suggestions, please reach out to [erfunmirzaei@gmail.com](mailto:erfunmirzaei@gmail.com).

<!-- The core Python files are kept at the repository root on purpose. Moving them into a package would require changing many imports and would add unnecessary risk for the release version. -->

## What To Run

- `main.py` runs the binary experiments for MNIST, CIFAR-10, CIFAR-100, SVHN, and synthetic data.
- `master.py` runs the 10-class MNIST experiments.
- `plot.py`, `plot_q_bounds.py`, `table_MNIST.py`, and `table_CIFAR.py` regenerate plots and tables from saved CSV files.
- `scripts/run_master_both_labels.sh` is kept as an example shell helper.

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

## Main Experiment Configurations

The main CSV names encode the configuration:

```text
{Dataset}{Labels}L{Layers}W{Width}{Sampler}{SampleSize}LR{LearningRate}{Loss}
```

For the main experiments used in the paper:

| CSV pair | Dataset | Labels | Model | Sampler | Train size | Learning rate | Loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `MCL2W1000SGLD8kLR001BBCE.csv` / `MRL2W1000SGLD8kLR001BBCE.csv` | MNIST binary | correct / random | 2 hidden layers, width 1000 | SGLD | 8k | 0.01 | BBCE |
| `CCL2W1500SGLD8kLR0005BBCE.csv` / `CRL2W1500SGLD8kLR0005BBCE.csv` | CIFAR-10 binary | correct / random | 2 hidden layers, width 1500 | SGLD | 8k | 0.005 | BBCE |

Here `M` means MNIST, `C` means CIFAR, `CL` means correct labels, and `RL` means random labels.

## Files And Outputs

- `dataset.py`, `multiclass_dataset_functions.py`: dataset loading and label randomization.
- `models.py`: model definitions.
- `losses.py`: bounded losses and evaluation losses.
- `sgld.py`, `mala.py`, `new_MALA.py`: samplers/optimizers.
- `training.py`, `training_multiclass.py`: training loops and CSV export.

Downloaded datasets live under `data/`, which is ignored by git.
Generated CSVs, plots, and checkpoints are written to `csv_EMA/`, `newplots/`, and `checkpoints/`; these folders are ignored by git.

## Reproducibility

Seeds are set in the scripts through `SEEDS` and `DATASET_SEED`. CUDA determinism is enabled where seeds are set, but exact GPU reproducibility can still depend on hardware, drivers, CUDA, and PyTorch versions. Generated CSVs, plots, logs, and checkpoints are ignored by default. 

## Citation

Please cite the associated ICML paper if you use this code. The exact proceedings citation should replace this placeholder once available:

```bibtex
@article{maurer2025generalization,
  title={Generalization of Gibbs and Langevin Monte Carlo Algorithms in the Interpolation Regime},
  author={Maurer, Andreas and Mirzaei, Erfan and Pontil, Massimiliano},
  journal={arXiv preprint arXiv:2510.06028},
  year={2025}
}
```

## License

MIT. See `LICENSE`.
