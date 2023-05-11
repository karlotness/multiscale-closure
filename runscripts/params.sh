# Constants
readonly DATA_SUBDIR='data-nowarmup'
readonly COARSE_OP='op1'

# Run parameters
readonly SINGULARITY_CONTAINER="${SCRATCH}/closure/closure.sif"
readonly ORIGIN_REPO_DIR="${HOME}/repos/closure.git"
readonly CHECKOUT_DIR="${SLURM_JOBTMP}/Closure/"
readonly TRAIN_DATA_DIR="${SCRATCH}/closure/${DATA_SUBDIR}/train/${COARSE_OP}/"
readonly VAL_DATA_DIR="${SCRATCH}/closure/${DATA_SUBDIR}/val/${COARSE_OP}/"
readonly EVAL_DATA_DIR="${SCRATCH}/closure/${DATA_SUBDIR}/test/${COARSE_OP}/"
