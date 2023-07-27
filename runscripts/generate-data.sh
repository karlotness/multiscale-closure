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

for expansion_level in '1.0'; do
    level_underscore=$(echo "$expansion_level" | tr '.' '_')
    DATA_DIR='data-smallstep-rand-eddytojet/factor-${level_underscore}'
    data_config="rand-eddy-to-jet-${expansion_level}"

    mkdir -p "${SCRATCH}/closure/${DATA_DIR}"
    # Train
    TRAIN_OUT1=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/train/' qg 0 --config ${data_config} --precision double --num_trajs=275 --traj_slice '0:60' --coarse_op ${COARSEN_OP} --subsample 10 --small_size 64 --dt 3600.0 --tmax 470160000.0 --twarmup 155520000.0" --job-name="qg-gen-train" --time="7:00:00" --cpus-per-task=1 --mem="20G" --gpus=1 --partition=gpu)
    TRAIN_JOBID1=$(get_job_id "$TRAIN_OUT1")
    sleep 0.5
    TRAIN_OUT2=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/train/' qg 0 --config ${data_config} --precision double --num_trajs=275 --traj_slice '60:120' --coarse_op ${COARSEN_OP} --subsample 10 --small_size 64 --dt 3600.0 --tmax 470160000.0 --twarmup 155520000.0" --job-name="qg-gen-train" --time="7:00:00" --cpus-per-task=1 --mem="20G" --gpus=1 --partition=gpu)
    TRAIN_JOBID2=$(get_job_id "$TRAIN_OUT2")
    sleep 0.5
    TRAIN_OUT3=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/train/' qg 0 --config ${data_config} --precision double --num_trajs=275 --traj_slice '120:180' --coarse_op ${COARSEN_OP} --subsample 10 --small_size 64 --dt 3600.0 --tmax 470160000.0 --twarmup 155520000.0" --job-name="qg-gen-train" --time="7:00:00" --cpus-per-task=1 --mem="20G" --gpus=1 --partition=gpu)
    TRAIN_JOBID3=$(get_job_id "$TRAIN_OUT3")
    sleep 0.5
    TRAIN_OUT4=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/train/' qg 0 --config ${data_config} --precision double --num_trajs=275 --traj_slice '180:240' --coarse_op ${COARSEN_OP} --subsample 10 --small_size 64 --dt 3600.0 --tmax 470160000.0 --twarmup 155520000.0" --job-name="qg-gen-train" --time="7:00:00" --cpus-per-task=1 --mem="20G" --gpus=1 --partition=gpu)
    TRAIN_JOBID4=$(get_job_id "$TRAIN_OUT4")
    sleep 0.5
    TRAIN_OUT5=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/train/' qg 0 --config ${data_config} --precision double --num_trajs=275 --traj_slice '240:' --coarse_op ${COARSEN_OP} --subsample 10 --small_size 64 --dt 3600.0 --tmax 470160000.0 --twarmup 155520000.0" --job-name="qg-gen-train" --time="7:00:00" --cpus-per-task=1 --mem="20G" --gpus=1 --partition=gpu)
    TRAIN_JOBID5=$(get_job_id "$TRAIN_OUT5")
    sleep 0.5
    TRAIN_OUT=$(sbatch --wrap="singularity exec '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/train/${COARSEN_OP}/' combine_qg_slice && rm ${SCRATCH}/closure/${DATA_DIR}/train/${COARSEN_OP}/data-slice*.hdf5" --job-name="qg-combine" --time="8:00:00" --cpus-per-task=2 --mem="20G" --dependency="afterok:$TRAIN_JOBID1:$TRAIN_JOBID2:$TRAIN_JOBID3:$TRAIN_JOBID4:$TRAIN_JOBID5" --kill-on-invalid-dep=yes)
    sleep 0.5
    # Val
    VAL_OUT=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/val/' qg 1 --config ${data_config} --precision double --num_trajs=4 --coarse_op ${COARSEN_OP} --subsample 1000 --small_size 64 --dt 3600.0 --tmax 470160000.0 --twarmup 155520000.0" --job-name="qg-gen-val" --time="4:00:00" --cpus-per-task=1 --mem="15G" --gpus=1 --partition=gpu)
    sleep 0.5
    # Test
    TEST_OUT=$(sbatch --wrap="singularity exec --nv '${SCRATCH}/closure/closure.sif' python generate_data.py '${SCRATCH}/closure/${DATA_DIR}/test/' qg 2 --config ${data_config} --precision double --num_trajs=16 --coarse_op ${COARSEN_OP} --subsample 8 --small_size 64 --dt 3600.0 --tmax 470160000.0 --twarmup 155520000.0" --job-name="qg-gen-test" --time="4:00:00" --cpus-per-task=1 --mem="15G" --gpus=1 --partition=gpu)
    sleep 0.5

    TRAIN_JOBID=$(get_job_id "$TRAIN_OUT")
    VAL_JOBID=$(get_job_id "$VAL_OUT")
    TEST_JOBID=$(get_job_id "$TEST_OUT")

    # Shuffle train data
    SHUF_OUT=$(sbatch --wrap="singularity exec '${SCRATCH}/closure/closure.sif' python shuffle_data.py '${SCRATCH}/closure/${DATA_DIR}/train/${COARSEN_OP}/'" --job-name="qg-shuf-train" --time="8:00:00" --cpus-per-task=4 --mem="4G" --dependency="afterok:$TRAIN_JOBID" --kill-on-invalid-dep=yes)
    SHUF_JOBID=$(get_job_id "$SHUF_OUT")
    sleep 0.5

    echo "EddyToJet data expansion: ${expansion_level}"
    echo "Generate train slice 1: $TRAIN_JOBID1"
    echo "Generate train slice 2: $TRAIN_JOBID2"
    echo "Generate train slice 3: $TRAIN_JOBID3"
    echo "Generate train slice 4: $TRAIN_JOBID4"
    echo "Generate train slice 5: $TRAIN_JOBID5"
    echo "Generate train:         $TRAIN_JOBID"
    echo "Generate val:           $VAL_JOBID"
    echo "Generate test:          $TEST_JOBID"
    echo "Shuffle train:          $SHUF_JOBID"

done
