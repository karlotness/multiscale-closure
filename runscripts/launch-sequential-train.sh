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
    name_prefix="$1"
    scales="$2"
    arch="$3"
    repeat="$4"

    if [[ "$arch" == 'gz-fcnn-v1' ]]; then
        arch_size='small'
    else
        arch_size='medium'
    fi

    joined_scales="$(tr ' ' '_' <<< "$scales")"
    run_name="${name_prefix}-scale${joined_scales}-${arch_size}-${repeat}"
    num_scales="$(wc -w <<< "$scales")"
    net_dir="/scratch/kto236/closure/run_outputs/sequential-train-cnn-${run_name}"

    # Launch first job
    run_out="$(sbatch sequential-train-cnn.sh "$run_name" "$arch" '0' "$scales")"
    run_jobid="$(get_job_id "$run_out")"
    echo "Submitted job ${run_jobid}"
    # Launch remaining training jobs
    for job in $(seq '1' "$(( num_scales - 1))"); do
        run_out="$(sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes sequential-train-cnn.sh "$run_name" "$arch" "$job" "$scales")"
        run_jobid="$(get_job_id "$run_out")"
        echo "Submitted job ${run_jobid}"
    done
    # Join the networks
    run_out="$(sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes sequential-to-cascade.sh "$net_dir")"
    run_jobid="$(get_job_id "$run_out")"
    echo "Submitted job ${run_jobid}"
    # Launch evaluation
    sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes cascade-net-eval.sh best_loss "$net_dir"
    sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes cascade-net-eval.sh interval "$net_dir"
}

readonly LAUNCH_TIME="$(date '+%Y%m%d-%H%M%S')"

# LAUNCH NETWORKS
for net_arch in 'gz-fcnn-v1' 'gz-fcnn-v1-medium'; do

    for i in {1..3}; do
        launch_job_and_eval "${LAUNCH_TIME}" '128 64' "$net_arch" "$i"
        launch_job_and_eval "${LAUNCH_TIME}" '128 96' "$net_arch" "$i"
        launch_job_and_eval "${LAUNCH_TIME}" '96 64' "$net_arch" "$i"
        launch_job_and_eval "${LAUNCH_TIME}" '128 96 64' "$net_arch" "$i"
    done
done
