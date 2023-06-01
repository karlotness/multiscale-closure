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

readonly DATA_DIR='data-nowarmup'
readonly COARSEN_OP='op1'

export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=32

TRAIN_OUT=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/train/' qg 0 --num_trajs=200 --coarse_op ${COARSEN_OP} --subsample 8 --small_size 128 96 64 48 --tmax 311040000.0 --twarmup 155520000.0" --job-name="qg-gen-train" --time="8:00:00" --cpus-per-task=1 --mem="20G" --gpus=1 --partition=gpu)
VAL_OUT=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/val/' qg 1 --num_trajs=3 --coarse_op ${COARSEN_OP} --subsample 8 --small_size 128 96 64 48 --tmax 311040000.0 --twarmup 155520000.0" --job-name="qg-gen-val" --time="1:00:00" --cpus-per-task=1 --mem="15G" --gpus=1 --partition=gpu)
TEST_OUT=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/test/' qg 2 --num_trajs=10 --coarse_op ${COARSEN_OP} --subsample 8 --small_size 128 96 64 48 --tmax 311040000.0 --twarmup 155520000.0" --job-name="qg-gen-test" --time="1:00:00" --cpus-per-task=1 --mem="15G" --gpus=1 --partition=gpu)

TRAIN_JOBID=$(get_job_id "$TRAIN_OUT")
VAL_JOBID=$(get_job_id "$VAL_OUT")
TEST_JOBID=$(get_job_id "$TEST_OUT")

SHUF_OUT=$(sbatch --wrap="singularity exec '${SCRATCH}/closure/closure.sif' python shuffle_data.py '${SCRATCH}/closure/${DATA_DIR}/train/${COARSEN_OP}/'" --job-name="qg-shuf-train" --time="8:00:00" --cpus-per-task=4 --mem="4G" --dependency="afterok:$TRAIN_JOBID" --kill-on-invalid-dep=yes)
SHUF_JOBID=$(get_job_id "$SHUF_OUT")

echo "Generate train: $TRAIN_JOBID"
echo "Generate val:   $VAL_JOBID"
echo "Generate test:  $TEST_JOBID"
echo "Shuffle train:  $SHUF_JOBID"
