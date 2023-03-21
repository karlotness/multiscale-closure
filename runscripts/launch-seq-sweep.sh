#!/bin/bash
set -euo pipefail

readonly LAUNCH_TIME="$(date '+%Y%m%d-%H%M%S')"

# LAUNCH NETWORKS
for net_arch in 'gz-fcnn-v1' 'gz-fcnn-v1-medium'; do
    for lr in '0.005' '0.001' '0.0005' '0.0002' '0.0001' '0.00005'; do
        sbatch sequential-lr-sweep.sh "${LAUNCH_TIME}-${net_arch}-lr${lr}" "$net_arch" "$lr"
    done
done
