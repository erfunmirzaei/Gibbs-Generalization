#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py39

mkdir -p logs

TS="$(date +%Y%m%d-%H%M%S)"

echo "Running CIFAR-10 with random labels for seeds 7,8"
DATASET_TYPE="cifar10" USE_RANDOM_LABELS=1 SEEDS="7,8" python main.py | tee "logs/cifar10_random_labels_seeds_7_8_${TS}.log"

echo "Running MNIST with random labels for seeds 1,2,3,4,5"
DATASET_TYPE="mnist" USE_RANDOM_LABELS=1 SEEDS="1,2,3,4,5" python main.py | tee "logs/cifar10_random_labels_seeds_1_2_3_4_5_${TS}.log"
echo "Run completed for seeds 7,8."
