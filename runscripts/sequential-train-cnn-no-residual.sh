#!/bin/bash

#SBATCH --job-name=sequential-train-cnn-no-residual
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=25GB
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000,v100

set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo 'ERROR: Insufficient parameters for training'
    echo 'Usage sequential-train-cnn-no-residual.sh OUT_DIR ARCH TRAIN_STEP PROCESSING_LEVELS(spaces)'
    exit 1
fi

readonly OUT_DIR="$1"
readonly ARCHITECTURE="$2"
readonly TRAIN_STEP="$3"
readonly PROCESSING_LEVELS="$4"

# Begin execution
module purge
module load git/2.31.0

# Make Bash more strict
shopt -s failglob
set -euo pipefail

# Constants
source params.sh

if [[ "$ARCHITECTURE" == "gz-fcnn-v1" ]]; then
    LR='0.0005'
    NUM_EPOCHS='132'
elif [[ "$ARCHITECTURE" == "gz-fcnn-v1-medium" ]]; then
    LR='0.0002'
    NUM_EPOCHS='96'
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
            python "${CHECKOUT_DIR}/src/sequential_train.py" "$OUT_DIR" "$TRAIN_DATA_DIR" "$VAL_DATA_DIR" \
            "$TRAIN_STEP" \
            $PROCESSING_LEVELS \
            --architecture="$ARCHITECTURE" \
            --optimizer=adam \
            --batch_size=256 \
            --num_epochs="$NUM_EPOCHS" \
            --batches_per_epoch=333 \
            --num_val_samples=100 \
            --val_interval=1 \
            --save_interval=1 \
            --lr="$LR" \
            --end_lr=0.0 \
            --lr_schedule=constant \
            --no_residual
