#!/bin/bash

#SBATCH --job-name=train-sde-buildup
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=gv0[13-18]

# Begin execution
module purge
module load git/2.31.0

# Make Bash more strict
shopt -s failglob
set -euo pipefail

# Constants
readonly BASE_NAME="train-sde-buildup"
readonly SINGULARITY_CONTAINER="${SCRATCH}/closure/closure.sif"
readonly ORIGIN_REPO_DIR="${HOME}/repos/closure.git"
readonly OUT_BASE_DIR="${SCRATCH}/closure/run_outputs/"
readonly CHECKOUT_DIR="${SLURM_JOBTMP}/Closure/"
readonly TRAIN_DATA_DIR="${SCRATCH}/closure/data/train/op1/"
readonly VAL_DATA_DIR="${SCRATCH}/closure/data/val/op1/"
readonly OUT_DIR="${OUT_BASE_DIR}/${BASE_NAME}-${SLURM_JOB_ID}-$(date '+%Y%m%d-%H%M%S')"

# Clone Repository
mkdir -p "$CHECKOUT_DIR"
git clone "$ORIGIN_REPO_DIR" "$CHECKOUT_DIR"

# Prepare output
mkdir -p "$OUT_DIR"

# Run
export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=32
singularity run --nv "$SINGULARITY_CONTAINER" \
            python "${CHECKOUT_DIR}/src/train_sdegm.py" "$OUT_DIR" "$TRAIN_DATA_DIR" "$VAL_DATA_DIR" \
            --batch_size=256 \
            --num_epochs=100 \
            --batches_per_epoch=1000 \
            --num_val_samples=10 \
            --val_mean_samples=25 \
            --val_interval=2 \
            --save_interval=1 \
            --lr=3e-4 \
            --dt=0.01 \
            --output_size=96 \
            --input_channels q_96 q_total_forcing_64
