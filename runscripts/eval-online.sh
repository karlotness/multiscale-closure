#!/bin/bash

#SBATCH --job-name=eval-online
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=15GB
#SBATCH --gpus=1
#SBATCH --partition=gpu

set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo 'ERROR: Insufficient parameters for online evaluation'
    echo 'Usage multi-train-cnn.sh OUT_DIR TRAIN_FACTOR NET_TYPE NET_DIRS...'
    exit 1
fi

readonly OUT_DIR="$1"
readonly TRAIN_FACTOR="$2"
readonly NET_TYPE="$3"
readonly NET_DIRS=( "${@:4}" )
readonly TRAIN_FACTOR_UNDERSCORE=$(echo "$TRAIN_FACTOR" | tr '.' '_')

# Make Bash more strict
shopt -s failglob
set -euo pipefail

# Constants
source params.sh

# Clone Repository
mkdir -p "$CHECKOUT_DIR"
git clone "$ORIGIN_REPO_DIR" "$CHECKOUT_DIR"

# Prepare output
mkdir -p "$OUT_DIR"

# Run
export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=32
singularity run --nv "$SINGULARITY_CONTAINER" \
            python "${CHECKOUT_DIR}/src/online_ensemble_compare.py" \
            --net_type="$NET_TYPE" \
            "$OUT_DIR" \
            "${SCRATCH}/closure/data-rand-1d-eddytojet/factor-${TRAIN_FACTOR_UNDERSCORE}/test/${COARSE_OP}/" \
            "${NET_DIRS[@]}"
