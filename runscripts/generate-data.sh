export JAX_ENABLE_X64=True
sbatch --wrap='singularity exec --nv ${SCRATCH}/closure/closure.sif python generate_data.py ${SCRATCH}/closure/data/train/ qg 0 --num_trajs=100 --coarse_op op1' --job-name="qg-gen" --time="12:00:00" --cpus-per-task=2 --mem="15G" --gres=gpu:1
sbatch --wrap='singularity exec --nv ${SCRATCH}/closure/closure.sif python generate_data.py ${SCRATCH}/closure/data/val/ qg 1 --num_trajs=3 --coarse_op op1' --job-name="qg-gen" --time="8:00:00" --cpus-per-task=2 --mem="15G" --gres=gpu:1
sbatch --wrap='singularity exec --nv ${SCRATCH}/closure/closure.sif python generate_data.py ${SCRATCH}/closure/data/test/ qg 2 --num_trajs=10 --coarse_op op1' --job-name="qg-gen" --time="8:00:00" --cpus-per-task=2 --mem="15G" --gres=gpu:1
