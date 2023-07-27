#! /bin/bash
set -euxo pipefail

for base_data_path in "${SCRATCH}/closure/data-smallstep-rand-eddytojet/factor-1_0"; do
    for set_size in 100 200; do
        out_name="train${set_size}"
        mkdir -p "${base_data_path}/${out_name}/op1"
        python subset_shuffle_data.py "${base_data_path}/train/op1/data.hdf5" "${base_data_path}/${out_name}/op1/data.hdf5" "$set_size"
        python shuffle_data.py "${base_data_path}/${out_name}/op1/"
        rm "${base_data_path}/${out_name}/op1/data.hdf5"
    done
done
