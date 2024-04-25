import subprocess
import codecs
import re
import time
import pathlib
import math
import os
import itertools
import operator
import dataclasses
import typing
import launch_utils as lu

SCRATCH = pathlib.Path(os.environ["SCRATCH"]).resolve()
VAST = pathlib.Path(os.environ["VAST"]).resolve()

# lu.enable_real_launch()

BATCH_TRAJS = 2
SIMULTANEOUS_TRAJS = 2
GEN_TIME = "12:00:00"
BIG_SIZE = 2048
SMALL_SIZES = {64, 128}

NUM_TRAIN_TRAJS = 128
NUM_VAL_TRAJS = 3
NUM_TEST_TRAJS = 16

launch_time = time.strftime("%Y%m%d-%H%M%S")
base_out_dir = SCRATCH / "closure" / f"data-ns-{launch_time}"


lu.dry_run_mkdir(base_out_dir)
for phase, num_trajs, seed in [
    ("train", NUM_TRAIN_TRAJS, 0),
    ("val", NUM_VAL_TRAJS, 1),
    ("test", NUM_TEST_TRAJS, 2),
]:
    phase_dir = base_out_dir / phase
    lu.dry_run_mkdir(phase_dir)
    launches = []
    # Compute each slice
    for slice_start in range(0, num_trajs, BATCH_TRAJS):
        slice_end = min(num_trajs, slice_start + BATCH_TRAJS)
        launch_args = [
            "python",
            "generate_data.py",
            phase_dir,
            "ns",
            seed,
            "--num_trajs", f"{num_trajs}",
            "--traj_slice", f"{slice_start}:{slice_end}",
            "--simultaneous_trajs", SIMULTANEOUS_TRAJS,
            "--big_size", BIG_SIZE,
            "--store_as_single",
        ]
        launch_args.append("--small_size")
        launch_args.extend(sorted(SMALL_SIZES))
        launch_id = lu.container_cmd_launch(
            launch_args,
            time_limit=GEN_TIME,
            job_name=f"ns-{phase}",
            cpus=2,
            gpus=1,
            mem_gb=20,
        )
        launches.append(launch_id)
