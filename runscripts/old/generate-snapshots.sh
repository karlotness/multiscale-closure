export JAX_ENABLE_X64=True
sbatch --wrap='singularity exec --nv ${SCRATCH}/closure/closure.sif python generate_data.py ${SCRATCH}/closure/snap_data/train/ qg_snap 0 --num_trajs=5000' --job-name="qg-snap-gen" --time="12:00:00" --cpus-per-task=1 --mem="4G" --gres=gpu:1
