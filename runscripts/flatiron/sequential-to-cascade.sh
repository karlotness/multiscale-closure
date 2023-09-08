#!/bin/bash

#SBATCH --job-name=sequential-to-cascade
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB

if [[ $# -lt 1 ]]; then
    echo "ERROR: Insufficient parameters for evaluation"
    echo "Usage cascade-net-eval.sh NET_DIR"
    exit 1
fi

readonly NET_DIR="$1"

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
singularity run "$SINGULARITY_CONTAINER" \
            python "${CHECKOUT_DIR}/src/sequential_to_cascaded.py" "$NET_DIR"
