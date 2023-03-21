#!/bin/bash
set -euo pipefail

readonly LAUNCH_TIME="$(date '+%Y%m%d-%H%M%S')"
readonly OUT_DIR="${SCRATCH}/closure/run_outputs/run-all-${LAUNCH_TIME}/"
mkdir -p "$OUT_DIR"

function echoing_sbatch() {
    >&2 echo "sbatch" "$@"
    sbatch "$@"
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

# LAUNCH SEQUENTIAL RUNS
for net_arch in 'gz-fcnn-v1' 'gz-fcnn-v1-medium'; do

    if [[ "$net_arch" == 'gz-fcnn-v1' ]]; then
        arch_size='small'
    else
        arch_size='medium'
    fi

    for i in {1..3}; do
        for scales in '128 64' '128 96' '96 64' '128 96 64'; do
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


declare -A sep_train_ids
sep_train_ids[downscale128to96-medium]=''
sep_train_ids[downscale128to64-medium]=''
sep_train_ids[downscale96to64-medium]=''
sep_train_ids[across128to96-medium]=''
sep_train_ids[across128to64-medium]=''
sep_train_ids[across96to64-medium]=''
sep_train_ids[downscale128to96-small]=''
sep_train_ids[downscale128to64-small]=''
sep_train_ids[downscale96to64-small]=''
sep_train_ids[across128to96-small]=''
sep_train_ids[across128to64-small]=''
sep_train_ids[across96to64-small]=''
sep_train_ids[buildup96to128-medium]=''
sep_train_ids[buildup64to128-medium]=''
sep_train_ids[buildup64to96-medium]=''
sep_train_ids[direct128-medium]=''
sep_train_ids[direct96-medium]=''
sep_train_ids[buildup96to128-small]=''
sep_train_ids[buildup64to128-small]=''
sep_train_ids[buildup64to96-small]=''
sep_train_ids[direct128-small]=''
sep_train_ids[direct96-small]=''

declare -A sep_train_paths
sep_train_paths[downscale128to96-medium]=''
sep_train_paths[downscale128to64-medium]=''
sep_train_paths[downscale96to64-medium]=''
sep_train_paths[across128to96-medium]=''
sep_train_paths[across128to64-medium]=''
sep_train_paths[across96to64-medium]=''
sep_train_paths[downscale128to96-small]=''
sep_train_paths[downscale128to64-small]=''
sep_train_paths[downscale96to64-small]=''
sep_train_paths[across128to96-small]=''
sep_train_paths[across128to64-small]=''
sep_train_paths[across96to64-small]=''
sep_train_paths[buildup96to128-medium]=''
sep_train_paths[buildup64to128-medium]=''
sep_train_paths[buildup64to96-medium]=''
sep_train_paths[direct128-medium]=''
sep_train_paths[direct96-medium]=''
sep_train_paths[buildup96to128-small]=''
sep_train_paths[buildup64to128-small]=''
sep_train_paths[buildup64to96-small]=''
sep_train_paths[direct128-small]=''
sep_train_paths[direct96-small]=''


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
    sep_train_paths[$run_arr_key]="${sep_train_paths[$run_arr_key]} ${run_sep_out}"
    sep_train_ids[$run_arr_key]="${sep_train_ids[$run_arr_key]} ${run_jobid}"

    # Launch single net evaluation
    echoing_sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes cnn-net-eval.sh best_loss "${run_sep_out}"
    echoing_sbatch --dependency="afterok:${run_jobid}" --kill-on-invalid-dep=yes cnn-net-eval.sh interval "${run_sep_out}"
}

# LAUNCH SEPARATE RUNS
for net_arch in 'gz-fcnn-v1' 'gz-fcnn-v1-medium'; do
    for i in {1..3}; do

        # Downscale and across
        launch_separate_job "downscale128to96" "$net_arch" 'q_128' '128' 'q_scaled_forcing_128to96' "$i"
        launch_separate_job "downscale128to64" "$net_arch" 'q_128' '128' 'q_scaled_forcing_128to64' "$i"
        launch_separate_job "downscale96to64" "$net_arch" 'q_96' '96' 'q_scaled_forcing_96to64' "$i"
        launch_separate_job "across128to96" "$net_arch" 'q_scaled_128to96' '128' 'q_scaled_forcing_128to96' "$i"
        launch_separate_job "across128to64" "$net_arch" 'q_scaled_128to64' '128' 'q_scaled_forcing_128to64' "$i"
        launch_separate_job "across96to64" "$net_arch" 'q_scaled_96to64' '96' 'q_scaled_forcing_96to64' "$i"

        # Buildup and direct
        launch_separate_job "buildup96to128" "$net_arch" 'q_128 q_scaled_forcing_128to96' '128' 'residual:q_total_forcing_128-q_scaled_forcing_128to96' "$i"
        launch_separate_job "buildup64to128" "$net_arch" 'q_128 q_scaled_forcing_128to64' '128' 'residual:q_total_forcing_128-q_scaled_forcing_128to64' "$i"
        launch_separate_job "buildup64to96" "$net_arch" 'q_96 q_scaled_forcing_96to64' '96' 'residual:q_total_forcing_96-q_scaled_forcing_96to64' "$i"
        launch_separate_job "direct128" "$net_arch" 'q_128' '128' 'q_total_forcing_128' "$i"
        launch_separate_job "direct96" "$net_arch" 'q_96' '96' 'q_total_forcing_96' "$i"

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

for small_size in 64 96 128; do
    for big_size in 64 96 128; do
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
