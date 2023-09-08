#!/bin/bash

set -euxo pipefail

export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=32
readonly CONTAINER='/mnt/ceph/users/kotness/closure/closure.sif'
readonly VAR_DATA='/mnt/ceph/users/kotness/closure/data-nowarmup/val/op1/'
readonly NET_BASE='/mnt/ceph/users/kotness/closure/run_outputs/'

# Ross22 training
sbatch --time='03:00:00' --mem='15G' --cpus-per-task=1 --gpus=1 -p gpu --wrap="singularity exec --nv '${CONTAINER}' python online_ensemble_compare.py '${NET_BASE}/repro-ross22-training/ensemble-eval/' '$VAR_DATA' '${NET_BASE}'/repro-ross22-training/net{0,1,2}/"

# Direct48 training
sbatch --time='03:00:00' --mem='15G' --cpus-per-task=1 --gpus=1 -p gpu --wrap="singularity exec --nv '${CONTAINER}' python online_ensemble_compare.py '${NET_BASE}/train-direct48/ensemble-eval/' '$VAR_DATA' '${NET_BASE}'/train-direct48/net{0,1,2}/"

# Direct64 training
sbatch --time='03:00:00' --mem='15G' --cpus-per-task=1 --gpus=1 -p gpu --wrap="singularity exec --nv '${CONTAINER}' python online_ensemble_compare.py '${NET_BASE}/run-all-nowarmup-20230531-100641/ensemble-eval-direct64/' '$VAR_DATA' '${NET_BASE}'/run-all-nowarmup-20230531-100641/multi-train-cnn-direct64-small-{1,2,3}/"

# Ross22 zero mean training
sbatch --time='03:00:00' --mem='15G' --cpus-per-task=1 --gpus=1 -p gpu --wrap="singularity exec --nv '${CONTAINER}' python online_ensemble_compare.py '${NET_BASE}/repro-ross22-training-zeromean/ensemble-eval/' '$VAR_DATA' '${NET_BASE}'/repro-ross22-training-zeromean/net{0,1,2}/"
