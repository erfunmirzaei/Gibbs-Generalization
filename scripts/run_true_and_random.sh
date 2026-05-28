#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Activate conda environment in non-interactive shell.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv

mkdir -p logs

echo "Running with true labels (USE_RANDOM_LABELS=0)"
USE_RANDOM_LABELS=0 python main.py | tee "logs/run_true_labels_$(date +%Y%m%d-%H%M%S).log"

echo "Running with random labels (USE_RANDOM_LABELS=1)"
USE_RANDOM_LABELS=1 python main.py | tee "logs/run_random_labels_$(date +%Y%m%d-%H%M%S).log"

echo "Both runs completed."
