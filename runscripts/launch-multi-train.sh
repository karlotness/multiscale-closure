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
    sbatch --dependency="afterok:${RUN_JOBID}" --kill-on-invalid-dep=yes cnn-net-eval.sh interval "/scratch/kto236/closure/run_outputs/${OUT_DIRECTORY}"
}

readonly LAUNCH_TIME="$(date '+%Y%m%d-%H%M%S')"

# LAUNCH NETWORKS
for arch in 'gz-fcnn-v1' 'gz-fcnn-v1-medium'; do

    if [[ "$arch" == 'gz-fcnn-v1' ]]; then
        arch_size='small'
    else
        arch_size='medium'
    fi

    if [[ "$arch_size" == 'small' ]]; then
        noise128to96='0.07747045,0.03359182'
        noise128to64='0.00355569,0.00170394'
        noise96to64='0.05255271,0.01690575'
    else
        noise128to96='0.07113694,0.02768604'
        noise128to64='0.00406553,0.00142028'
        noise96to64='0.04949836,0.01457414'
    fi


    for i in {1..3}; do
        # RUN_KEY ARCH INPUT_CHANNELS(spaces) PROC_SIZE OUTPUT_CHANNELS(spaces)
        # Downscale and across
        launch_job_and_eval "${LAUNCH_TIME}-downscale128to96-${arch_size}-${i}" "$arch" 'q_128' '128' 'q_scaled_forcing_128to96' ''
        launch_job_and_eval "${LAUNCH_TIME}-downscale128to64-${arch_size}-${i}" "$arch" 'q_128' '128' 'q_scaled_forcing_128to64' ''
        launch_job_and_eval "${LAUNCH_TIME}-downscale96to64-${arch_size}-${i}" "$arch" 'q_96' '96' 'q_scaled_forcing_96to64' ''
        launch_job_and_eval "${LAUNCH_TIME}-across128to96-${arch_size}-${i}" "$arch" 'q_scaled_128to96' '128' 'q_scaled_forcing_128to96' ''
        launch_job_and_eval "${LAUNCH_TIME}-across128to64-${arch_size}-${i}" "$arch" 'q_scaled_128to64' '128' 'q_scaled_forcing_128to64' ''
        launch_job_and_eval "${LAUNCH_TIME}-across96to64-${arch_size}-${i}" "$arch" 'q_scaled_96to64' '96' 'q_scaled_forcing_96to64' ''

        # Buildup and direct
        launch_job_and_eval "${LAUNCH_TIME}-buildup96to128-${arch_size}-${i}" "$arch" 'q_128 q_scaled_forcing_128to96' '128' 'residual:q_total_forcing_128-q_scaled_forcing_128to96' ''
        launch_job_and_eval "${LAUNCH_TIME}-buildup64to128-${arch_size}-${i}" "$arch" 'q_128 q_scaled_forcing_128to64' '128' 'residual:q_total_forcing_128-q_scaled_forcing_128to64' ''
        launch_job_and_eval "${LAUNCH_TIME}-buildup64to96-${arch_size}-${i}" "$arch" 'q_96 q_scaled_forcing_96to64' '96' 'residual:q_total_forcing_96-q_scaled_forcing_96to64' ''
        launch_job_and_eval "${LAUNCH_TIME}-direct128-${arch_size}-${i}" "$arch" 'q_128' '128' 'q_total_forcing_128' ''
        launch_job_and_eval "${LAUNCH_TIME}-direct96-${arch_size}-${i}" "$arch" 'q_96' '96' 'q_total_forcing_96' ''

        # Buildup with injected noise
        launch_job_and_eval "${LAUNCH_TIME}-buildup96to128-noiseinject-${arch_size}-${i}" "$arch" 'q_128 q_scaled_forcing_128to96' '128' 'residual:q_total_forcing_128-q_scaled_forcing_128to96' "q_scaled_forcing_128to96=${noise128to96}"
        launch_job_and_eval "${LAUNCH_TIME}-buildup64to128-noiseinject-${arch_size}-${i}" "$arch" 'q_128 q_scaled_forcing_128to64' '128' 'residual:q_total_forcing_128-q_scaled_forcing_128to64' "q_scaled_forcing_128to64=${noise128to64}"
        launch_job_and_eval "${LAUNCH_TIME}-buildup64to96-noiseinject-${arch_size}-${i}" "$arch" 'q_96 q_scaled_forcing_96to64' '96' 'residual:q_total_forcing_96-q_scaled_forcing_96to64' "q_scaled_forcing_96to64=${noise96to64}"
    done
done
