#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure conda is available in non-interactive shells, then activate the requested env.
if ! command -v conda >/dev/null 2>&1; then
	if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
		source "$HOME/miniconda3/etc/profile.d/conda.sh"
	elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
		source "$HOME/anaconda3/etc/profile.d/conda.sh"
	else
		echo "Error: conda not found. Install Conda or update this script with your conda.sh path." >&2
		exit 1
	fi
fi

eval "$(conda shell.bash hook)"
conda activate gibbs

echo "Using Python: $(python -V 2>&1)"

echo "Running master.py with USE_RANDOM_LABELS=1"
USE_RANDOM_LABELS=1 python master.py

echo "Running master.py with USE_RANDOM_LABELS=0"
USE_RANDOM_LABELS=0 python master.py

echo "Both runs completed successfully."
