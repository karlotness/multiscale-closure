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
    RUN_OUT=$(sbatch cascade-train-cnn.sh "$@")
    RUN_JOBID=$(get_job_id "$RUN_OUT")
    echo "Submitted job ${RUN_JOBID}"
    RUN_KEY="$1"
    OUT_DIRECTORY="cascade-train-cnn-${RUN_KEY}-${RUN_JOBID}"
    # LAUNCH TRAINING
    sbatch --dependency="afterok:${RUN_JOBID}" --kill-on-invalid-dep=yes cascade-net-eval.sh best_loss "/scratch/kto236/closure/run_outputs/${OUT_DIRECTORY}"
    sbatch --dependency="afterok:${RUN_JOBID}" --kill-on-invalid-dep=yes cascade-net-eval.sh interval "/scratch/kto236/closure/run_outputs/${OUT_DIRECTORY}"
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
        # RUN_KEY ARCH PROCESSING_LEVELS(spaces)
        launch_job_and_eval "${LAUNCH_TIME}-scale128_64-${arch_size}-${i}" "$arch" '128 64'
        launch_job_and_eval "${LAUNCH_TIME}-scale128_96-${arch_size}-${i}" "$arch" '128 96'
        launch_job_and_eval "${LAUNCH_TIME}-scale96_64-${arch_size}-${i}" "$arch" '96 64'
        launch_job_and_eval "${LAUNCH_TIME}-scale128_96_64-${arch_size}-${i}" "$arch" '128 96 64'
    done
done
