#!/bin/bash

set -euo pipefail
shopt -s failglob

readonly JOBTMP_DIR="$(mktemp --tmpdir="$SLURM_JOBTMP" -d "${USER}_job${SLURM_JOBID}_XXXXXXXXXX")"

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
    export JAX_PLATFORMS=cpu
    gpu_args=()
fi
export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=32
export CUDA_CACHE_DISABLE=1
cd "${CHECKOUT_DIR}/src"
singularity run "${gpu_args[@]}" "$SINGULARITY_CONTAINER" "${@:2}"
