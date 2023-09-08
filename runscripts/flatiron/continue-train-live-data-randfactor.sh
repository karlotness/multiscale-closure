#!/bin/bash

#SBATCH --job-name=continue-live
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=25GB
#SBATCH --gpus=1
#SBATCH --partition=gpu

# Reproduce Ross22 paper training loop
# Using data sets with varying system parameters

set -euo pipefail

if [[ $# -lt 9 ]]; then
    echo 'ERROR: Insufficient parameters for training'
    echo 'Usage continue-train-randfactor.sh OUT_DIR TRAIN_DIR VAL_DIR SCALE WEIGHT_FILE LR_MODE NUM_CANDIDATES NUM_WINNERS NOISE_MODE'
    exit 1
fi

readonly OUT_DIR="$1"
readonly TRAIN_DIR="$2"
readonly VAL_DIR="$3"
readonly SCALE="$4"
readonly WEIGHT_FILE="$5"
readonly LR_MODE="$6"
readonly NUM_CANDIDATES="$7"
readonly NUM_WINNERS="$8"
readonly NOISE_MODE="$9"

if [[ "$LR_MODE" == 'continue' ]]; then
    LR='0.000001'
else
    LR='0.001'
fi

if [[ "$NOISE_MODE" == 'add-noise' ]]; then
    noise_args=('--noise_specs' "q_${SCALE}=0.07845675,0.03721482")
else
    noise_args=()
fi

# Make Bash more strict
shopt -s failglob
set -euo pipefail

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
            python "${CHECKOUT_DIR}/src/train.py" "$OUT_DIR" "${TRAIN_DIR}/${COARSE_OP}/" "${VAL_DIR}/${COARSE_OP}/" \
            --optimizer=adam \
            --batch_size=64 \
            --num_epochs=100 \
            --batches_per_epoch=374 \
            --loader_chunk_size=23925 \
            --num_val_samples=100 \
            --val_interval=1 \
            --save_interval=1 \
            --lr="$LR" \
            --end_lr="$LR" \
            --lr_schedule=ross22 \
            --architecture='gz-fcnn-v1' \
            --input_channels "q_${SCALE}" "rek_${SCALE}" "delta_${SCALE}" "beta_${SCALE}" \
            --output_channels "q_total_forcing_${SCALE}" \
            --net_weight_continue="$WEIGHT_FILE" \
            --live_gen_start_epoch=30 \
            --live_gen_interval=25 \
            --live_gen_sampler='rand-eddy-to-jet-1.0' \
            --live_gen_candidates="$NUM_CANDIDATES" \
            --live_gen_winners="$NUM_WINNERS" \
            "${noise_args[@]}"
