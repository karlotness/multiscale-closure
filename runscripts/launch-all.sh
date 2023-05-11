#!/bin/bash
set -euo pipefail

# Set to 'true' or 'false'
readonly DRY_RUN='true'
readonly LAUNCH_NON_CASCADED='false'
readonly SCALES='128 96 64 48'
readonly NUM_REPEATS='3'
readonly LAUNCH_TIME="$(date '+%Y%m%d-%H%M%S')"
readonly OUT_DIR="${SCRATCH}/closure/run_outputs/run-all-nowarmup-${LAUNCH_TIME}/"

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
            net_dir="${OUT_DIR}/sequential-train-cnn-${run_name}"

            # Launch first job
            run_out="$(echoing_sbatch sequential-train-cnn.sh "$net_dir" "$net_arch" '0' "$scales")"
            run_jobid="$(get_job_id "$run_out")"
            echo "Submitted job ${run_jobid}"

            # Launch remaining training jobs
            for job in $(seq '1' "$(( num_scales - 1))"); do
                run_out="$(echoing_sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes sequential-train-cnn.sh "$net_dir" "$net_arch" "$job" "$scales")"
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


if [[ "$LAUNCH_NON_CASCADED" != 'true' ]]; then
    exit 0
fi


declare -A sep_train_ids
declare -A sep_train_paths
for scale_pair in "${paired_scales[@]}"; do
    mapfile -t -d ' ' < <(echo -n "$scale_pair")
    large_scale="${MAPFILE[0]}"
    small_scale="${MAPFILE[1]}"
    for arch_size in 'small' 'medium'; do
        sep_train_ids["downscale${large_scale}to${small_scale}-${arch_size}"]=''
        sep_train_ids["across${large_scale}to${small_scale}-${arch_size}"]=''
        sep_train_ids["buildup${small_scale}to${large_scale}-${arch_size}"]=''
        sep_train_ids["direct${large_scale}-${arch_size}"]=''
        sep_train_paths["downscale${large_scale}to${small_scale}-${arch_size}"]=''
        sep_train_paths["across${large_scale}to${small_scale}-${arch_size}"]=''
        sep_train_paths["buildup${small_scale}to${large_scale}-${arch_size}"]=''
        sep_train_paths["direct${large_scale}-${arch_size}"]=''
    done
done


function launch_separate_job() {
    local run_type="$1"
    local run_arch="$2"
    local run_in_chans="$3"
    local run_proc_size="$4"
    local run_out_chans="$5"
    local run_trial="$6"

    if [[ "$run_arch" == 'gz-fcnn-v1' ]]; then
        local run_arch_size='small'
    else
        local run_arch_size='medium'
    fi

    local run_sep_out="${OUT_DIR}/multi-train-cnn-${run_type}-${run_arch_size}-${run_trial}"

    local run_out=$(echoing_sbatch multi-train-cnn.sh "$run_sep_out" "$run_arch" "$run_in_chans" "$run_proc_size" "$run_out_chans")
    local run_jobid="$(get_job_id "$run_out")"
    echo "Submitted job ${run_jobid}"

    # Store job path and training ID
    local run_arr_key="${run_type}-${run_arch_size}"
    sep_train_paths[$run_arr_key]="${sep_train_paths[$run_arr_key]} ${run_sep_out}/weights/best_loss.eqx"
    sep_train_ids[$run_arr_key]="${sep_train_ids[$run_arr_key]} ${run_jobid}"

    # Launch single net evaluation
    echoing_sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes cnn-net-eval.sh best_loss "${run_sep_out}"
    echoing_sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes cnn-net-eval.sh interval "${run_sep_out}"
}

# LAUNCH SEPARATE RUNS
for net_arch in 'gz-fcnn-v1' 'gz-fcnn-v1-medium'; do
    for i in $(seq "$NUM_REPEATS"); do

        # Paired scale nets
        for scale_pair in "${paired_scales[@]}"; do
            mapfile -t -d ' ' < <(echo -n "$scale_pair")
            large_scale="${MAPFILE[0]}"
            small_scale="${MAPFILE[1]}"

            # Downscale and across
            launch_separate_job "downscale${large_scale}to${small_scale}" "$net_arch" "q_${large_scale}" "$large_scale" "q_scaled_forcing_${large_scale}to${small_scale}" "$i"
            launch_separate_job "across${large_scale}to${small_scale}" "$net_arch" "q_scaled_${large_scale}to${small_scale}" "$large_scale" "q_scaled_forcing_${large_scale}to${small_scale}" "$i"

            # Buildup
            launch_separate_job "buildup${small_scale}to${large_scale}" "$net_arch" "q_${large_scale} q_scaled_forcing_${large_scale}to${small_scale}" "$large_scale" "residual:q_total_forcing_${large_scale}-q_scaled_forcing_${large_scale}to${small_scale}" "$i"
        done

        # Buildup and direct
        for scale in "${direct_scales[@]}"; do
            launch_separate_job "direct${scale}" "$net_arch" "q_${scale}" "${scale}" "q_total_forcing_${scale}" "$i"
        done
    done
done

# LAUNCH COMBINED EVALUATIONS (on separate nets)

function abbrev_arch() {
    local ARCH_NAME="$1"
    if [[ "$ARCH_NAME" == "small" ]]; then
        echo "sm"
        return 0
    elif [[ "$ARCH_NAME" == "medium" ]]; then
        echo "md"
        return 0
    elif [[ "$ARCH_NAME" == "large" ]]; then
        echo "lg"
        return 0
    fi
}

for small_size in $SCALES; do
    for big_size in $SCALES; do
        if (( small_size >= big_size )); then
            continue
        fi
        # No training noise version
        for buildup_arch in small medium; do
            buildup_abbrev=$(abbrev_arch "$buildup_arch")
            buildup_var="buildup${small_size}to${big_size}-${buildup_arch}"
            buildup_waitids=$(echo "${sep_train_ids[$buildup_var]}" | sed 's/^ *//;s/ *$//' | tr ' ' ':')
            buildup_dirs=$(echo "${sep_train_paths[$buildup_var]}" | sed 's/^ *//;s/ *$//')
            for downscale_arch in small medium; do
                downscale_abbrev=$(abbrev_arch "$downscale_arch")
                downscale_var="downscale${big_size}to${small_size}-${downscale_arch}"
                downscale_waitids=$(echo "${sep_train_ids[$downscale_var]}" | sed 's/^ *//;s/ *$//' | tr ' ' ':')
                downscale_dirs=$(echo "${sep_train_paths[$downscale_var]}" | sed 's/^ *//;s/ *$//')
                # Network output directory
                combine_out="${OUT_DIR}/combine-eval-cnn-${small_size}to${big_size}-${downscale_abbrev}${buildup_abbrev}"
                combine_waitids="${buildup_waitids}:${downscale_waitids}"
                echoing_sbatch --dependency="afterok:${combine_waitids}" --kill-on-invalid-dep=yes combine-eval-cnn.sh "$combine_out" "$downscale_dirs" "$buildup_dirs"
            done
        done
    done
done
