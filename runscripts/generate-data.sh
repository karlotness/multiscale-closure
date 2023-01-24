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

TRAIN_OUT=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/data/train/' qg 0 --num_trajs=100 --coarse_op op2 --subsample 8" --job-name="qg-gen-train" --time="8:00:00" --cpus-per-task=1 --mem="20G" --gres=gpu:1)
VAL_OUT=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/data/val/' qg 1 --num_trajs=3 --coarse_op op2 --subsample 8" --job-name="qg-gen-val" --time="1:00:00" --cpus-per-task=1 --mem="15G" --gres=gpu:1)
TEST_OUT=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/data/test/' qg 2 --num_trajs=10 --coarse_op op2 --subsample 8" --job-name="qg-gen-test" --time="1:00:00" --cpus-per-task=1 --mem="15G" --gres=gpu:1)

TRAIN_JOBID=$(get_job_id "$TRAIN_OUT")
VAL_JOBID=$(get_job_id "$VAL_OUT")
TEST_JOBID=$(get_job_id "$TEST_OUT")

SHUF_OUT=$(sbatch --wrap="singularity exec '${SCRATCH}/closure/closure.sif' python shuffle_data.py '${SCRATCH}/closure/data/train/op2/'" --job-name="qg-shuf-train" --time="8:00:00" --cpus-per-task=4 --mem="4G" --dependency="afterok:$TRAIN_JOBID" --kill-on-invalid-dep=yes)
SHUF_JOBID=$(get_job_id "$SHUF_OUT")

echo "Generate train: $TRAIN_JOBID"
echo "Generate val:   $VAL_JOBID"
echo "Generate test:  $TEST_JOBID"
echo "Shuffle train:  $SHUF_JOBID"
