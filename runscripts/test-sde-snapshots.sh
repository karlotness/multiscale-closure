#!/bin/bash

#SBATCH --job-name=train-sde-snap
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=5GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=gv0[13-18]

# Begin execution
module purge
module load git/2.31.0

# Make Bash more strict
shopt -s failglob
set -euo pipefail

# Constants
readonly BASE_NAME="test-sde-snap"
readonly SINGULARITY_CONTAINER="${SCRATCH}/closure/closure.sif"
readonly ORIGIN_REPO_DIR="${HOME}/repos/closure.git"
readonly OUT_BASE_DIR="${SCRATCH}/closure/run_outputs/"
readonly CHECKOUT_DIR="${SLURM_JOBTMP}/Closure/"
readonly TRAIN_DATA_DIR="${SCRATCH}/closure/snap_data/train/op1/"
readonly OUT_DIR="${OUT_BASE_DIR}/${BASE_NAME}-${SLURM_JOB_ID}-$(date '+%Y%m%d-%H%M%S')"

# Clone Repository
mkdir -p "$CHECKOUT_DIR"
git clone "$ORIGIN_REPO_DIR" "$CHECKOUT_DIR"

# Prepare output
mkdir -p "$OUT_DIR"

# Run
singularity run --nv "$SINGULARITY_CONTAINER" \
            python "${CHECKOUT_DIR}/src/train_sdegm.py" "$OUT_DIR" "$TRAIN_DATA_DIR" \
            --batch_size=256 \
            --num_epochs=100 \
            --batches_per_epoch=1000 \
            --save_interval=1 \
            --lr=3e-4 \
            --dt=0.01 \
            --num_epoch_samples=5 \
