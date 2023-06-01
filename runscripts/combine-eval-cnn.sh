#!/bin/bash

#SBATCH --job-name=combine-eval-cnn
#SBATCH --time=2:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=15GB
#SBATCH --gpus=1
#SBATCH --partition=gpu

# Begin execution
module purge
module load git/2.31.0

# Handle arguments
if [[ $# -lt 3 ]]; then
    echo 'ERROR: Insufficient parameters for combined evaluation'
    echo 'Usage multi-train-cnn.sh OUT_DIR DOWNSCALE_NETS(spaces) BUILDUP_NETS(spaces)'
    exit 1
fi

readonly OUT_DIR="$1"
readonly DOWNSCALE_NETS="$2"
readonly BUILDUP_NETS="$3"

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
            python "${CHECKOUT_DIR}/src/combined_eval.py" "$OUT_DIR" "$EVAL_DATA_DIR" \
            --seed=0 \
            --sample_seed=0 \
            --num_samples=1024 \
            --downscale_nets $DOWNSCALE_NETS \
            --buildup_nets $BUILDUP_NETS
