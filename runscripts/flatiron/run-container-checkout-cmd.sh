#!/bin/bash

set -euo pipefail
shopt -s failglob

if [[ ! -v SCRATCH ]]; then
   export SCRATCH="/mnt/ceph/users/$USER"
fi

readonly JOBTMP_DIR="$(mktemp --tmpdir -d "${USER}_job${SLURM_JOBID}_XXXXXXXXXX")"

function cleanup_jobtmp_dir() {
    rm -rf "$JOBTMP_DIR"
}
trap cleanup_jobtmp_dir EXIT

export TMPDIR="$JOBTMP_DIR"

# Run parameters
readonly use_gpu="$1"
readonly SINGULARITY_CONTAINER="${SCRATCH}/closure/closure.sif"
readonly ORIGIN_REPO_DIR="${HOME}/repos/closure.git"
readonly CHECKOUT_DIR="${JOBTMP_DIR}/Closure/"

# Clone Repository
mkdir -p "$CHECKOUT_DIR"
git clone "$ORIGIN_REPO_DIR" "$CHECKOUT_DIR"

# Run
if [[ "$use_gpu" == "cuda" ]]; then
    gpu_args=('--nv')
else
    export JAX_PLATFORM_NAME=cpu
    gpu_args=()
fi
export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=32
cd "${CHECKOUT_DIR}/src"
singularity run "${gpu_args[@]}" "$SINGULARITY_CONTAINER" "${@:2}"
