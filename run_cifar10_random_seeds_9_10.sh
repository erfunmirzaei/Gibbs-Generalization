#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py39

mkdir -p logs

TS="$(date +%Y%m%d-%H%M%S)"

echo "Running CIFAR-10 with random labels for seeds 6,9,10"
DATASET_TYPE="cifar10" USE_RANDOM_LABELS=1 SEEDS="6,9,10" python main.py | tee "logs/cifar10_random_labels_seeds_6_9_10_${TS}.log"

echo "Running MNIST with random labels for seeds 6,7,8,9,10"
DATASET_TYPE="mnist" USE_RANDOM_LABELS=1 SEEDS="6,7,8,9,10" python main.py | tee "logs/cifar10_random_labels_seeds_6_7_8_9_10_${TS}.log"

echo "Run completed for seeds 9,10."
