#!/bin/bash

#SBATCH --job-name=train-large-cnn-proc128-vanilla
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000,v100
#SBATCH --exclude=gv0[13-18]

# Begin execution
module purge
module load git/2.31.0

# Make Bash more strict
shopt -s failglob
set -euo pipefail

# Constants
readonly BASE_NAME="train-large-cnn-proc128-vanilla"
readonly SINGULARITY_CONTAINER="${SCRATCH}/closure/closure.sif"
readonly ORIGIN_REPO_DIR="${HOME}/repos/closure.git"
readonly OUT_BASE_DIR="${SCRATCH}/closure/run_outputs/"
readonly CHECKOUT_DIR="${SLURM_JOBTMP}/Closure/"
readonly TRAIN_DATA_DIR="${SCRATCH}/closure/data/train/op1/"
readonly VAL_DATA_DIR="${SCRATCH}/closure/data/val/op1/"
readonly OUT_DIR="${OUT_BASE_DIR}/${BASE_NAME}-${SLURM_JOB_ID}"

# Clone Repository
mkdir -p "$CHECKOUT_DIR"
git clone "$ORIGIN_REPO_DIR" "$CHECKOUT_DIR"

# Prepare output
mkdir -p "$OUT_DIR"

# Run
export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=32
singularity run --nv "$SINGULARITY_CONTAINER" \
            python "${CHECKOUT_DIR}/src/train.py" "$OUT_DIR" "$TRAIN_DATA_DIR" "$VAL_DATA_DIR" \
            --optimizer=adam \
            --task_type=basic-cnn \
            --batch_size=256 \
            --num_epochs=125 \
            --batches_per_epoch=1000 \
            --num_val_samples=100 \
            --val_interval=6 \
            --save_interval=1 \
            --lr=3e-4 \
            --architecture=gz-fcnn-v1-large \
            --output_size=64 \
            --processing_size=128 \
            --input_channels q_64
