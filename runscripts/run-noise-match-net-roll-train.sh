#!/bin/bash

#SBATCH --job-name=noisematch-net
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=25GB
#SBATCH --gpus=1
#SBATCH --partition=gpu

set -euo pipefail
shopt -s failglob

if [[ $# -lt 10 ]]; then
    echo 'ERROR: Insufficient parameters for training'
    echo 'Usage run-noise-match-net-roll-train.sh OUT_DIR TRAIN_DIR VAL_DIR LIVE_DATA_DIR ORIG_WEIGHT GEN_MODE SPEC_STR NUM_STEPS NUM_STEPS ARCHITECTURE'
    exit 1
fi

readonly OUT_DIR="$1"
readonly TRAIN_DIR="$2"
readonly VAL_DIR="$3"
readonly LIVE_DATA_DIR="$4"
readonly WEIGHT_FILE="$5"
readonly GEN_MODE="$6"
readonly SPEC_STR="$7"
readonly NUM_ROLLOUT_STEPS="$8"
readonly NUM_CANDIDATES="$9"
readonly ARCHITECTURE="${10}"

readonly LR='0.001'
readonly SCALE='64'

if [[ -n "$SPEC_STR" ]]; then
    noise_args=('--noise_specs' "$SPEC_STR")
else
    noise_args=()
fi

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
            --num_epochs=100 \
            --batches_per_epoch=374 \
            --loader_chunk_size=23925 \
            --num_val_samples=100 \
            --val_interval=1 \
            --save_interval=1 \
            --lr="$LR" \
            --end_lr="$LR" \
            --lr_schedule=ross22 \
            --architecture="$ARCHITECTURE" \
            --input_channels "q_${SCALE}" "rek_${SCALE}" "delta_${SCALE}" "beta_${SCALE}" \
            --output_channels "q_total_forcing_${SCALE}" \
            --net_weight_continue="$WEIGHT_FILE" \
            --live_gen_start_epoch=1 \
            --live_gen_interval=1 \
            --live_gen_candidates="$NUM_CANDIDATES" \
            --live_gen_winners="$NUM_CANDIDATES" \
            --live_gen_mode="$GEN_MODE" \
            --live_gen_net_steps="$NUM_ROLLOUT_STEPS" \
            --live_gen_base_data="$LIVE_DATA_DIR" \
            "${noise_args[@]}"
