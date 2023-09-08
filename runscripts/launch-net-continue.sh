#!/bin/bash
set -euo pipefail

readonly LAUNCH_TIME="$(date '+%Y%m%d-%H%M%S')"
readonly OUT_DIR="${SCRATCH}/closure/run_outputs/continue-runs-${LAUNCH_TIME}"

# Set to 'true' or 'false'
readonly DRY_RUN='true'
readonly NET_FILES=(
    "${SCRATCH}/closure/run_outputs/run-varied-data-size-20230626-174855/rand-eddytojet/size100-scale64/net2/weights/epoch0025.eqx"
    "${SCRATCH}/closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net0/weights/best_loss.eqx"
    "${SCRATCH}/closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net2/weights/best_loss.eqx"
    "${SCRATCH}/closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net3/weights/best_loss.eqx"
    "${SCRATCH}/closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net3/weights/epoch0045.eqx"
    "${SCRATCH}/closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net4/weights/best_loss.eqx"
    "${SCRATCH}/closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net4/weights/epoch0035.eqx"
    "${SCRATCH}/closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net4/weights/epoch0045.eqx"
)
readonly TRAIN_DIR="${SCRATCH}/closure/data-rand-eddytojet/factor-1_0/train100"
readonly VAL_DIR="${SCRATCH}/closure/data-rand-eddytojet/factor-1_0/val"
readonly TEST_DIR="${SCRATCH}/closure/data-rand-eddytojet/factor-1_0/test-trainset"
readonly LR_MODES=('continue' 'restart')
readonly NUM_REPEATS='3'


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
        echo "Submitted batch job $(( DRY_RUN_COUNTER++ ))"
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

for net_file in "${NET_FILES[@]}"; do
    if [[ $net_file =~ /closure/run_outputs/run-varied-data-size-([[:digit:]]+-[[:digit:]]+)/([^/]+)/size([[:digit:]]+)-scale([[:digit:]]+)/([^/]+)/weights/([^/]+).eqx ]]; then
        continue_date="${BASH_REMATCH[1]}"
        continue_type="${BASH_REMATCH[2]}"
        continue_size="${BASH_REMATCH[3]}"
        continue_scale="${BASH_REMATCH[4]}"
        continue_name="${BASH_REMATCH[5]}"
        continue_weight="${BASH_REMATCH[6]}"
        for lr_mode in "${LR_MODES[@]}"; do
            net_out_dir="${OUT_DIR}/run-varied-data-size-${continue_date}-${continue_type}-size${continue_size}-scale${continue_scale}-${continue_name}-${continue_weight}-${lr_mode}"
            online_eval_deps=""
            if [[ "$DRY_RUN" != 'true' ]]; then
                mkdir -p "$net_out_dir"
            else
                echo "mkdir -p $net_out_dir"
            fi
            declare -a net_out_dirs=()
            for repeat in $(seq "$NUM_REPEATS" ); do
                launch_dir="${net_out_dir}/trial${repeat}"
                net_out_dirs+=("$launch_dir")
                run_out=$(echoing_sbatch continue-train-randfactor.sh "$launch_dir" "$TRAIN_DIR" "$VAL_DIR" "$continue_scale" "$net_file" "$lr_mode")
                run_jobid="$(get_job_id "$run_out")"
                echo "Submitted job ${run_jobid}"
                online_eval_deps="${online_eval_deps}:${run_jobid}"
           done

            if [[ "$DRY_RUN" != 'true' ]]; then
                mkdir -p "${net_out_dir}/online-ke-trainset/"
            else
                echo "mkdir -p ${net_out_dir}/online-ke-trainset/"
            fi

            # Launch online evaluation
            for eval_epoch in 'best_loss' 'epoch0050' 'epoch0075' 'epoch0100'; do
                echoing_sbatch --dependency="afterok${online_eval_deps}" --kill-on-invalid-dep=yes run-online-ke-data.sh "${net_out_dir}/online-ke-trainset/ke-${eval_epoch}.hdf5" "$TEST_DIR" ${net_out_dirs[*]/%//weights/${eval_epoch}.eqx}
            done
        done
    fi
done
