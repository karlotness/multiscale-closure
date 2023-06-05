#!/bin/bash
set -euo pipefail

# Set to 'true' or 'false'
readonly DRY_RUN='true'
readonly SCALES='128 96 64 48'
readonly NUM_REPEATS='3'
readonly LAUNCH_TIME="$(date '+%Y%m%d-%H%M%S')"
readonly OUT_DIR="${SCRATCH}/closure/run_outputs/run-all-noresidual-nowarmup-${LAUNCH_TIME}/"

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

function check_descending() {
    local candidate="$1"
    local min_val=''
    local v
    for v in $candidate; do
        if [[ -n $min_val ]] && (( v >= min_val )); then
             return 1
        fi
        min_val="$v"
    done
    return 0
}

function generate_multi_scale() {
    local num_scales
    local concat_cmd
    local scales_braces
    local candidate
    local scale_depth
    local group
    local i
    num_scales="$(echo "$SCALES" | wc -w)"
    scales_braces="{$(echo "$SCALES" | tr -s ' ' ',')}"
    for scale_depth in $(seq 2 "$num_scales"); do
        # Construct command
        concat_cmd='echo '
        for i in $(seq "$scale_depth"); do
            concat_cmd="${concat_cmd}${scales_braces}-"
        done
        # Loop over the groups
        for group in $(eval "$concat_cmd" | tr -s ' ' "\n"); do
            candidate="$(echo "$group" | tr -s '-' ' ' | sed 's/^ *//;s/ *$//')"
            if check_descending "$candidate"; then
                echo "$candidate"
            fi
        done
    done
}

function generate_paired_scale() {
    local small_scale
    local large_scale
    for large_scale in $SCALES; do
        for small_scale in $SCALES; do
            if (( large_scale > small_scale )); then
                echo "$large_scale $small_scale"
            fi
        done
    done
}

function generate_direct_scale() {
    echo "$SCALES" | tr ' ' "\n" | sort -gr | head -n -1
}

mapfile -t < <(generate_multi_scale)
readonly multi_scales=("${MAPFILE[@]}")
mapfile -t < <(generate_paired_scale)
readonly paired_scales=("${MAPFILE[@]}")
mapfile -t < <(generate_direct_scale)
readonly direct_scales=("${MAPFILE[@]}")

# LAUNCH SEQUENTIAL RUNS
for net_arch in 'gz-fcnn-v1' 'gz-fcnn-v1-medium'; do

    if [[ "$net_arch" == 'gz-fcnn-v1' ]]; then
        arch_size='small'
    else
        arch_size='medium'
    fi

    for i in $(seq "$NUM_REPEATS"); do
        for scales in "${multi_scales[@]}"; do
            joined_scales="$(tr ' ' '_' <<< "$scales")"
            run_name="${joined_scales}-${arch_size}-${i}"
            num_scales="$(wc -w <<< "$scales")"
            net_dir="${OUT_DIR}/sequential-train-cnn-no-residual-${run_name}"

            # Launch first job
            run_out="$(echoing_sbatch sequential-train-cnn-no-residual.sh "$net_dir" "$net_arch" '0' "$scales")"
            run_jobid="$(get_job_id "$run_out")"
            echo "Submitted job ${run_jobid}"

            # Launch remaining training jobs
            for job in $(seq '1' "$(( num_scales - 1))"); do
                run_out="$(echoing_sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes sequential-train-cnn-no-residual.sh "$net_dir" "$net_arch" "$job" "$scales")"
                run_jobid="$(get_job_id "$run_out")"
                echo "Submitted job ${run_jobid}"
            done

            # Join the networks
            run_out="$(echoing_sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes sequential-to-cascade.sh "$net_dir")"
            run_jobid="$(get_job_id "$run_out")"
            echo "Submitted job ${run_jobid}"

            # Launch evaluation
            echoing_sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes cascade-net-eval.sh best_loss "$net_dir"
            echoing_sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes cascade-net-eval.sh interval "$net_dir"
        done
    done
done
