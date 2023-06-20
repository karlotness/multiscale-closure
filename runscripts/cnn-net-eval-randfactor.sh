#!/bin/bash

#SBATCH --job-name=cnn-net-eval
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --partition=gpu

if [[ $# -lt 3 ]]; then
    echo "ERROR: Insufficient parameters for evaluation"
    echo "Usage cnn-net-eval.sh EVAL_TYPE NET_DIR" TRAIN_FACTOR
    exit 1
fi

readonly EVAL_TYPE="$1"
readonly NET_DIR="$2"
readonly TRAIN_FACTOR="$3"
readonly TRAIN_FACTOR_UNDERSCORE=$(echo "$TRAIN_FACTOR" | tr '.' '_')

# Make Bash more strict
shopt -s failglob
set -euo pipefail

# Constants
source params.sh

# Clone Repository
mkdir -p "$CHECKOUT_DIR"
git clone "$ORIGIN_REPO_DIR" "$CHECKOUT_DIR"

# Run
export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=32
singularity run --nv "$SINGULARITY_CONTAINER" \
            python "${CHECKOUT_DIR}/src/eval.py" "$NET_DIR" "${SCRATCH}/closure/data-rand-eddytojet/factor-${TRAIN_FACTOR_UNDERSCORE}/test/${COARSE_OP}/" "$EVAL_TYPE" \
            --seed=0 \
            --sample_seed=0 \
            --num_samples=1024
