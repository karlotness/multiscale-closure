#!/bin/bash

#SBATCH --job-name=train-fcnn-snapshot
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1

# Begin execution
module purge
module load git/2.31.0

# Make Bash more strict
shopt -s failglob
set -euo pipefail

# Constants
readonly BASE_NAME="test-fcnn-snapshot"
readonly SINGULARITY_CONTAINER="${SCRATCH}/closure/closure.sif"
readonly ORIGIN_REPO_DIR="${HOME}/repos/closure.git"
readonly OUT_BASE_DIR="${SCRATCH}/closure/run_outputs/"
readonly CHECKOUT_DIR="${SLURM_JOBTMP}/Closure/"
readonly TRAIN_DATA_DIR="${SCRATCH}/closure/data/train/op1/"
readonly VAL_DATA_DIR="${SCRATCH}/closure/data/val/op1/"
readonly OUT_DIR="${OUT_BASE_DIR}/$(date '+%Y%m%d-%H%M%S')-${BASE_NAME}-${SLURM_JOB_ID}"

# Clone Repository
mkdir -p "$CHECKOUT_DIR"
git clone "$ORIGIN_REPO_DIR" "$CHECKOUT_DIR"

# Prepare output
mkdir -p "$OUT_DIR"

# Run
export JAX_ENABLE_X64=True
singularity run --nv "$SINGULARITY_CONTAINER" \
            python "${CHECKOUT_DIR}/src/train.py" "$OUT_DIR" "$TRAIN_DATA_DIR" "$VAL_DATA_DIR" \
            --lr=0.00001 \
            --weight_decay=0 \
            --batch_size=75 \
            --train_epochs=49 \
            --batches_per_epoch=100 \
            --rollout_length='2' \
            --val_steps=250 \
            --val_samples=15 \
            --save_interval=1 \
            --architecture=gz-fcnn-v1 \
