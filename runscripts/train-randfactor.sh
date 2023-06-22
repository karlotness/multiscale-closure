#!/bin/bash

#SBATCH --job-name=train-randfactor
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=25GB
#SBATCH --gpus=1
#SBATCH --partition=gpu

# Reproduce Ross22 paper training loop
# Using data sets with varying system parameters

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo 'ERROR: Insufficient parameters for training'
    echo 'Usage multi-train-cnn.sh OUT_DIR TRAIN_FACTOR'
    exit 1
fi

readonly OUT_DIR="$1"
readonly TRAIN_FACTOR="$2"
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
            python "${CHECKOUT_DIR}/src/train.py" "$OUT_DIR" "${SCRATCH}/closure/data-rand-eddytojet/factor-${TRAIN_FACTOR_UNDERSCORE}/train/${COARSE_OP}/" "${SCRATCH}/closure/data-rand-eddytojet/factor-${TRAIN_FACTOR_UNDERSCORE}/val/${COARSE_OP}/" \
            --optimizer=adam \
            --batch_size=64 \
            --num_epochs=50 \
            --batches_per_epoch=374 \
            --loader_chunk_size=23925 \
            --num_val_samples=100 \
            --val_interval=1 \
            --save_interval=1 \
            --lr=0.001 \
            --end_lr=0.001 \
            --lr_schedule=ross22 \
            --architecture='gz-fcnn-v1' \
            --input_channels "q_64" "rek_64" "delta_64" "beta_64" \
            --output_channels "q_total_forcing_64"
