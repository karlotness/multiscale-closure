#!/bin/bash

#SBATCH --job-name=eval-plots
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=15GB
#SBATCH --gpus=1
#SBATCH --partition=gpu

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo 'ERROR: Insufficient parameters for online evaluation'
    echo 'Usage multi-train-cnn.sh OUT_FILE EVAL_FILE NET_FILES...'
    exit 1
fi

readonly OUT_FILE="$1"
readonly EVAL_FILE="$2"
readonly NET_FILES=( "${@:3}" )

# Make Bash more strict
shopt -s failglob
set -euo pipefail

# Constants
source params.sh

# Clone Repository
mkdir -p "$CHECKOUT_DIR"
git clone "$ORIGIN_REPO_DIR" "$CHECKOUT_DIR"

# Prepare output
mkdir -p "$(dirname "$OUT_FILE")"

# Run
export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=32
singularity run --nv "$SINGULARITY_CONTAINER" \
            python "${CHECKOUT_DIR}/src/online_data_eval.py" \
            "$OUT_FILE" \
            "$EVAL_FILE" \
            "${NET_FILES[@]}"
