#!/bin/bash
set -euo pipefail

# Set to 'true' or 'false'
readonly DRY_RUN='true'
readonly SCALES=('64')
readonly NUM_REPEATS='3'
readonly LAUNCH_TIME="$(date '+%Y%m%d-%H%M%S')"
readonly TRAIN_SETS=(
    "${SCRATCH}/closure/data-rand-eddytojet/factor-1_0"
    "${SCRATCH}/closure/data-eddyonly"
)
readonly TRAIN_SET_NAMES=(
    "rand-eddytojet"
    "eddyonly"
)
readonly SUBSET_SIZES=('2' '10' '25' '50' '100')
readonly OUT_DIR="${SCRATCH}/closure/run_outputs/run-varied-data-size-${LAUNCH_TIME}"

if [[ "$DRY_RUN" != 'true' ]]; then
    mkdir -p "$OUT_DIR"
else
    DRY_RUN_COUNTER=10000
fi

function echoing_sbatch() {
    local var
    >&2 echo -n "sbatch "
    for var in "$@"; do
        >&2 echo -n "\"$var\" "
    done
    >&2 echo ""
    if [[ "$DRY_RUN" != 'true' ]]; then
        sbatch "$@"
        sleep 0.5
    else
        echo "Submitted batch job $((DRY_RUN_COUNTER++))"
    fi
}


function get_job_id() {
    local SBATCH_OUTPUT="$1"
    if [[ "$SBATCH_OUTPUT" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    else
        return 1
    fi
}

seti=0
for train_set_base in "${TRAIN_SETS[@]}"; do
    train_set_name="${TRAIN_SET_NAMES[$seti]}"
    for subset_size in "${SUBSET_SIZES[@]}"; do
        train_dir="${train_set_base}/train${subset_size}"
        val_dir="${train_set_base}/val"
        test_dir="${train_set_base}/test"

        for scale in "${SCALES[@]}"; do
            scale_out_dir="${OUT_DIR}/${train_set_name}/size${subset_size}-scale${scale}"
            if [[ "$DRY_RUN" != 'true' ]]; then
                mkdir -p "$scale_out_dir"
            else
                echo "mkdir -p $scale_out_dir"
            fi
            declare -a net_out_dirs=()
            online_eval_deps=""
            for i in $(seq 0 "$(( NUM_REPEATS - 1 ))" ); do
                run_out_dir="$scale_out_dir/net${i}"
                net_out_dirs+=("$run_out_dir")

                # Launch network training
                run_out=$(echoing_sbatch train-randfactor.sh "$run_out_dir" "$train_dir" "$val_dir" "$scale")
                run_jobid="$(get_job_id "$run_out")"
                echo "Submitted job ${run_jobid}"
                online_eval_deps="${online_eval_deps}:${run_jobid}"

                # Launch single network evaluation
                echoing_sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes cnn-net-eval-randfactor.sh best_loss "${run_out_dir}/" "$test_dir"
                echoing_sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes cnn-net-eval-randfactor.sh interval "${run_out_dir}/" "$test_dir"
            done

            # Launch online evaluation
            echoing_sbatch --dependency="afterok${online_eval_deps}" --kill-on-invalid-dep=yes eval-online.sh "${scale_out_dir}/ensemble-eval" "$test_dir" 'best_loss' "${net_out_dirs[@]}"
        done
    done
    seti=$(( seti + 1 ))
done
