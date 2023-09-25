#!/bin/bash

#SBATCH --job-name=multi-train-cnn
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=25GB
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ $# -lt 5 ]]; then
    echo 'ERROR: Insufficient parameters for training'
    echo 'Usage multi-train-cnn.sh OUT_DIR ARCH INPUT_CHANNELS(spaces) PROC_SIZE OUTPUT_CHANNELS(spaces)'
    exit 1
fi

readonly OUT_DIR="$1"
readonly ARCHITECTURE="$2"
readonly INPUT_CHANNELS="$3"
readonly PROC_SIZE="$4"
readonly OUTPUT_CHANNELS="$5"

# Begin execution
module purge
module load git/2.31.0

# Make Bash more strict
shopt -s failglob
set -euo pipefail

# Constants
source params.sh

if [[ "$ARCHITECTURE" == "gz-fcnn-v1" || "$ARCHITECTURE" == "stacked-gz-fcnn-v1-d2" || "$ARCHITECTURE" == "stacked-gz-fcnn-v1-d3" ]]; then
    LR='0.0005'
    NUM_EPOCHS='132'
elif [[ "$ARCHITECTURE" == "gz-fcnn-v1-medium" || "$ARCHITECTURE" == "stacked-gz-fcnn-v1-medium-d2" || "$ARCHITECTURE" == "stacked-gz-fcnn-v1-medium-d3" ]]; then
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
            python "${CHECKOUT_DIR}/src/train.py" "$OUT_DIR" "$TRAIN_DATA_DIR" "$VAL_DATA_DIR" \
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
            --architecture="$ARCHITECTURE" \
            --processing_size="$PROC_SIZE" \
            --input_channels $INPUT_CHANNELS \
            --output_channels $OUTPUT_CHANNELS
