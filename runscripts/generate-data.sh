#! /bin/bash
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


export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=32
readonly COARSEN_OP='op1'

for expansion_level in '0.05' '0.10' '0.20' '0.50' '0.75'; do
    level_underscore=$(echo "$expansion_level" | tr '.' '_')
    DATA_DIR="data-rand-eddytojet/factor-${level_underscore}"
    data_config="rand-eddy-to-jet-${expansion_level}"

    mkdir -p "${SCRATCH}/closure/${DATA_DIR}"
    # Train
    TRAIN_OUT=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/train/' qg 0 --config ${data_config} --num_trajs=275 --coarse_op ${COARSEN_OP} --subsample 1000 --small_size 64 48 --dt 3600.0 --tmax 470160000.0 --twarmup 155520000.0" --job-name="qg-gen-train" --time="8:00:00" --cpus-per-task=1 --mem="20G" --gpus=1 --partition=gpu)
    sleep 0.5
    # Val
    VAL_OUT=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/val/' qg 1 --config ${data_config} --num_trajs=4 --coarse_op ${COARSEN_OP} --subsample 1000 --small_size 64 48 --dt 3600.0 --tmax 470160000.0 --twarmup 155520000.0" --job-name="qg-gen-val" --time="1:00:00" --cpus-per-task=1 --mem="15G" --gpus=1 --partition=gpu)
    sleep 0.5
    # Test
    TEST_OUT=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/test/' qg 2 --config ${data_config} --num_trajs=4 --coarse_op ${COARSEN_OP} --subsample 8 --small_size 64 48 --dt 3600.0 --tmax 470160000.0 --twarmup 155520000.0" --job-name="qg-gen-test" --time="1:00:00" --cpus-per-task=1 --mem="15G" --gpus=1 --partition=gpu)
    sleep 0.5

    TRAIN_JOBID=$(get_job_id "$TRAIN_OUT")
    VAL_JOBID=$(get_job_id "$VAL_OUT")
    TEST_JOBID=$(get_job_id "$TEST_OUT")

    # Shuffle train data
    SHUF_OUT=$(sbatch --wrap="singularity exec '${SCRATCH}/closure/closure.sif' python shuffle_data.py '${SCRATCH}/closure/${DATA_DIR}/train/${COARSEN_OP}/'" --job-name="qg-shuf-train" --time="8:00:00" --cpus-per-task=4 --mem="4G" --dependency="afterok:$TRAIN_JOBID" --kill-on-invalid-dep=yes)
    SHUF_JOBID=$(get_job_id "$SHUF_OUT")
    sleep 0.5

    echo "EddyToJet data expansion: ${expansion_level}"
    echo "Generate train: $TRAIN_JOBID"
    echo "Generate val:   $VAL_JOBID"
    echo "Generate test:  $TEST_JOBID"
    echo "Shuffle train:  $SHUF_JOBID"

done
