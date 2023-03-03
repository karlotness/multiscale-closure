#!/bin/bash

#SBATCH --job-name=cascade-train-cnn
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=25GB
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000,v100

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo 'ERROR: Insufficient parameters for training'
    echo 'Usage cascade-train-cnn.sh RUN_KEY ARCH PROCESSING_LEVELS(spaces)'
    exit 1
fi

readonly RUN_KEY="$1"
readonly ARCHITECTURE="$2"
readonly PROCESSING_LEVELS="$3"

# Begin execution
module purge
module load git/2.31.0

# Make Bash more strict
shopt -s failglob
set -euo pipefail

# Constants
readonly BASE_NAME="cascade-train-cnn-${RUN_KEY}"
readonly SINGULARITY_CONTAINER="${SCRATCH}/closure/closure.sif"
readonly ORIGIN_REPO_DIR="${HOME}/repos/closure.git"
readonly OUT_BASE_DIR="${SCRATCH}/closure/run_outputs/"
readonly CHECKOUT_DIR="${SLURM_JOBTMP}/Closure/"
readonly TRAIN_DATA_DIR="${SCRATCH}/closure/data/train/op1/"
readonly VAL_DATA_DIR="${SCRATCH}/closure/data/val/op1/"
readonly OUT_DIR="${OUT_BASE_DIR}/${BASE_NAME}-${SLURM_JOB_ID}"

if [[ "$ARCHITECTURE" == "gz-fcnn-v1" ]]; then
    LR='0.001'
    NUM_EPOCHS='66'
elif [[ "$ARCHITECTURE" == "gz-fcnn-v1-medium" ]]; then
    LR='0.000222'
    NUM_EPOCHS='48'
else
    echo "Unsupported architecture '${ARCHITECTURE}'"
    exit 1
fi

# Clone Repository
mkdir -p "$CHECKOUT_DIR"
git clone "$ORIGIN_REPO_DIR" "$CHECKOUT_DIR"

# Prepare output
mkdir -p "$OUT_DIR"

# Run
export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=32
singularity run --nv "$SINGULARITY_CONTAINER" \
            python "${CHECKOUT_DIR}/src/cascaded_train.py" "$OUT_DIR" "$TRAIN_DATA_DIR" "$VAL_DATA_DIR" \
            $PROCESSING_LEVELS \
            --architecture="$ARCHITECTURE" \
            --optimizer=adabelief \
            --batch_size=256 \
            --num_epochs="$NUM_EPOCHS" \
            --batches_per_epoch=333 \
            --num_val_samples=100 \
            --val_interval=1 \
            --save_interval=1 \
            --lr="$LR" \
            --end_lr=0.0 \
            --lr_schedule=warmup1-cosine
