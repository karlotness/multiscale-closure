#!/bin/bash

#SBATCH --job-name=net-eval
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

if [[ $# -lt 2 ]]; then
    echo "ERROR: Insufficient parameters for evaluation"
    echo "Usage net-eval.sh EVAL_TYPE NET_DIR"
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
readonly SINGULARITY_CONTAINER="${SCRATCH}/closure/closure.sif"
readonly ORIGIN_REPO_DIR="${HOME}/repos/closure.git"
readonly CHECKOUT_DIR="${SLURM_JOBTMP}/Closure/"
readonly EVAL_DATA_DIR="${SCRATCH}/closure/data/test/op1/"

# Clone Repository
mkdir -p "$CHECKOUT_DIR"
git clone "$ORIGIN_REPO_DIR" "$CHECKOUT_DIR"

# Run
singularity run --nv "$SINGULARITY_CONTAINER" \
            python "${CHECKOUT_DIR}/src/eval.py" "$NET_DIR" "$EVAL_DATA_DIR" "$EVAL_TYPE"
