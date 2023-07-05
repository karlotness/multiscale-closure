#!/bin/bash
set -euo pipefail
shopt -s nullglob

export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=32

# Set to 'true' or 'false'
readonly DRY_RUN='true'
readonly SCALES=('64' '48')
readonly RUN_DIRS=(
    "${SCRATCH}/closure/run_outputs/run-varied-data-size-20230626-174855"
    "${SCRATCH}/closure/run_outputs/run-varied-data-size-20230629-173901"
)
readonly RUN_TYPES=(
    'jetonly'
    'eddyonly'
    'rand-eddytojet'
)
readonly SUBSET_SIZES=('100' '50' '25')
readonly WEIGHT_TYPES=('best_loss' 'epoch0045' 'epoch0035' 'epoch0025')

if [[ "$DRY_RUN" != 'true' ]]; then
    true
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

for subset_size in "${SUBSET_SIZES[@]}"; do
    for weight_type in "${WEIGHT_TYPES[@]}"; do
        for run_dir in "${RUN_DIRS[@]}"; do
            for run_type in "${RUN_TYPES[@]}"; do
                for scale in "${SCALES[@]}"; do
                    base_dir="${run_dir}/${run_type}/size${subset_size}-scale${scale}"
                    nets=("$base_dir"/net*)
                    if (( ${#nets[@]} == 0 )); then
                        # Skip any runs that don't have a network collection
                        continue
                    fi
                    if [[ "$run_type" == "rand-eddytojet" ]]; then
                        eval_path="${SCRATCH}/closure/data-${run_type}/factor-1_0/test/op1/"
                    else
                        eval_path="${SCRATCH}/closure/data-${run_type}/test/op1/"
                    fi
                    out_dir="${base_dir}/online-ke"
                    if [[ "$DRY_RUN" != 'true' ]]; then
                        mkdir -p "$out_dir"
                    fi
                    out_file="${out_dir}/ke-${weight_type}.hdf5"

                    echoing_sbatch  --job-name="ke-eval" --time="4:00:00" --cpus-per-task=1 --ntasks=1 --mem="10G" --gpus=1 --partition=gpu \
                                    --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python online_ke_data.py '${out_file}' '${eval_path}' ${nets[*]/%//weights/${weight_type}.eqx}"

                done
            done
        done
    done
done
