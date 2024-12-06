#!/bin/bash

#SBATCH --job-name=cnn-net-eval
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

if [[ $# -lt 2 ]]; then
    echo "ERROR: Insufficient parameters for evaluation"
    echo "Usage cnn-net-eval.sh EVAL_TYPE NET_DIR"
    exit 1
fi

readonly EVAL_TYPE="$1"
readonly NET_DIR="$2"

# Begin execution
module purge
module load git/2.31.0

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
            python "${CHECKOUT_DIR}/src/eval.py" "$NET_DIR" "$EVAL_DATA_DIR" "$EVAL_TYPE" \
            --seed=0 \
            --sample_seed=0 \
            --num_samples=1024
