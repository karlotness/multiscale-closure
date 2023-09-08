#!/bin/bash

#SBATCH --job-name=unet-train
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=25GB
#SBATCH --gpus=1
#SBATCH --partition=gpu

set -euo pipefail
shopt -s failglob

if [[ $# -lt 3 ]]; then
    echo 'ERROR: Insufficient parameters for training'
    echo 'Usage run-unet-basictrain.sh OUT_DIR TRAIN_DIR VAL_DIR'
    exit 1
fi

readonly OUT_DIR="$1"
readonly TRAIN_DIR="$2"
readonly VAL_DIR="$3"
readonly LR='0.001'
readonly SCALE='64'

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
            python "${CHECKOUT_DIR}/src/train.py" "$OUT_DIR" "$TRAIN_DIR" "$VAL_DIR" \
            --optimizer=adam \
            --batch_size=64 \
            --num_epochs=50 \
            --batches_per_epoch=374 \
            --loader_chunk_size=23925 \
            --num_val_samples=100 \
            --val_interval=1 \
            --save_interval=1 \
            --lr="$LR" \
            --end_lr="$LR" \
            --lr_schedule=ross22 \
            --architecture='unet-v1' \
            --input_channels "q_${SCALE}" "rek_${SCALE}" "delta_${SCALE}" "beta_${SCALE}" \
            --output_channels "q_total_forcing_${SCALE}" \
            --live_gen_start_epoch=1 \
            --live_gen_interval=1 \
            --live_gen_candidates="0" \
            --live_gen_winners="0" \
            --live_gen_mode="schedule-only" \
            --live_gen_net_steps="1"
