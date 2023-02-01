#!/bin/bash
set -euo pipefail

function get_job_id() {
    local SBATCH_OUTPUT="$1"
    if [[ "$SBATCH_OUTPUT" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    else
        return 1
    fi
}

function launch_job_and_eval() {
    RUN_OUT=$(sbatch multi-train-cnn.sh "$@")
    RUN_JOBID=$(get_job_id "$RUN_OUT")
    echo "Submitted job ${RUN_JOBID}"
    RUN_KEY="$1"
    OUT_DIRECTORY="multi-train-cnn-${RUN_KEY}-${RUN_JOBID}"
    # LAUNCH TRAINING
    sbatch --dependency="afterok:${RUN_JOBID}" --kill-on-invalid-dep=yes cnn-net-eval.sh best_loss "/scratch/kto236/closure/run_outputs/${OUT_DIRECTORY}"
}

readonly LAUNCH_TIME="$(date '+%Y%m%d-%H%M%S')"

# LAUNCH NETWORKS
for arch in 'gz-fcnn-v1' 'gz-fcnn-v1-medium'; do

    if [[ "$arch" == 'gz-fcnn-v1' ]]; then
        arch_size='small'
    else
        arch_size='medium'
    fi

    for i in {1..3}; do
        # RUN_KEY ARCH OUT_SIZE PROC_SIZE INPUT_CHANNELS(spaces)
        # Downscale and across
        launch_job_and_eval "${LAUNCH_TIME}-downscale128-${arch_size}-${i}" "$arch" '64' '128' 'q_128'
        launch_job_and_eval "${LAUNCH_TIME}-across128-${arch_size}-${i}" "$arch" '64' '128' 'q_64'
        launch_job_and_eval "${LAUNCH_TIME}-downscale96-${arch_size}-${i}" "$arch" '64' '96' 'q_96'
        launch_job_and_eval "${LAUNCH_TIME}-across96-${arch_size}-${i}" "$arch" '64' '96' 'q_64'

        # Buildup and direct
        launch_job_and_eval "${LAUNCH_TIME}-buildup64to128-${arch_size}-${i}" "$arch" '128' '128' 'q_128 q_total_forcing_64'
        launch_job_and_eval "${LAUNCH_TIME}-buildup96to128-${arch_size}-${i}" "$arch" '128' '128' 'q_128 q_total_forcing_96'
        launch_job_and_eval "${LAUNCH_TIME}-direct128-${arch_size}-${i}" "$arch" '128' '128' 'q_128'
        launch_job_and_eval "${LAUNCH_TIME}-buildup64to96-${arch_size}-${i}" "$arch" '96' '96' 'q_96 q_total_forcing_64'
        launch_job_and_eval "${LAUNCH_TIME}-direct96-${arch_size}-${i}" "$arch" '96' '96' 'q_96'
    done
done
