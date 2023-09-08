# Constants
readonly DATA_SUBDIR='data-nowarmup'
readonly COARSE_OP='op1'

if [[ ! -v SCRATCH ]]; then
   export SCRATCH="/mnt/ceph/users/$USER"
fi

readonly JOBTMP_DIR=$(mktemp --tmpdir -d "${USER}_job${SLURM_JOBID}_XXXXXXXXXX")

function cleanup_jobtmp_dir() {
    rm -rf "$JOBTMP_DIR"
}
trap cleanup_jobtmp_dir EXIT

export TMPDIR="$JOBTMP_DIR"

# Run parameters
readonly SINGULARITY_CONTAINER="${SCRATCH}/closure/closure.sif"
readonly ORIGIN_REPO_DIR="${HOME}/repos/closure.git"
readonly CHECKOUT_DIR="${JOBTMP_DIR}/Closure/"
readonly TRAIN_DATA_DIR="${SCRATCH}/closure/${DATA_SUBDIR}/train/${COARSE_OP}/"
readonly VAL_DATA_DIR="${SCRATCH}/closure/${DATA_SUBDIR}/val/${COARSE_OP}/"
readonly EVAL_DATA_DIR="${SCRATCH}/closure/${DATA_SUBDIR}/test/${COARSE_OP}/"
