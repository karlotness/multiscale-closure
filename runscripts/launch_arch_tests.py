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

TrainFileSet = dataclasses.make_dataclass("TrainFileSet", ["train", "val", "test"])

SCRATCH = pathlib.Path(os.environ["SCRATCH"]).resolve()

# lu.enable_real_launch()

# General parameters
NUM_REPEATS = 4
DATA_TYPES = ["eddy"]
SCALE_SETS = [(128, 96)]
NET_DEPTHS = [8, 4, 2]
ARCH_BASES = ["small", "medium", "puresmall", "puremedium"]
LR_SCHEDULE = "warmup1-cosine"

@dataclasses.dataclass
class SweepParams:
    peak_lr: float
    num_epochs: int


LR_ASSIGNMENTS = {
    "small": SweepParams(peak_lr=0.001, num_epochs=100),
    "puresmall": SweepParams(peak_lr=0.001, num_epochs=100),
    "medium": SweepParams(peak_lr=0.0004, num_epochs=50),
    "puremedium": SweepParams(peak_lr=0.0004, num_epochs=50),
}


# Training and eval sets
COARSE_OP = "op1"
DATA_FILES = {
    "eddy": TrainFileSet(
        train=SCRATCH / "closure" / "data-nowarmup" / f"train/{COARSE_OP}/shuffled.hdf5",
        val=SCRATCH / "closure" / "data-nowarmup" / f"val/{COARSE_OP}/data.hdf5",
        test=SCRATCH / "closure" / "data-eddyonly" / f"test/{COARSE_OP}/data.hdf5",
    ),
    "jet": TrainFileSet(
        train=SCRATCH / "closure" / "data-jet-nowarmup" / f"train/{COARSE_OP}/shuffled.hdf5",
        val=SCRATCH / "closure" / "data-jet-nowarmup" / f"val/{COARSE_OP}/data.hdf5",
        test=SCRATCH / "closure" / "data-jet-nowarmup" / f"test/{COARSE_OP}/data.hdf5",
    ),
}
assert all(p.is_file() for p in itertools.chain.from_iterable([df.train, df.val, df.test] for df in DATA_FILES.values()))

launch_time = time.strftime("%Y%m%d-%H%M%S")
base_out_dir = SCRATCH / "closure" / "run_outputs"/ f"arch-test-runs-{launch_time}"


def launch_training(
    *,
    out_dir,
    train_dir,
    val_dir,
    scale,
    optimizer="adam",
    batch_size=64,
    num_epochs=50,
    batches_per_epoch=374,
    num_val_samples=300,
    val_interval=1,
    save_interval=10,
    lr=0.001,
    end_lr=0.0,
    lr_schedule="ross22",
    architecture="gz-fcnn-v1",
    processing_size=None,
    input_channels=None,
    output_channels=None,
    dependency_ids=None,
):
    args = [
        "python",
        "train.py",
        out_dir,
        train_dir,
        val_dir,
        f"--optimizer={optimizer}",
        f"--batch_size={batch_size:d}",
        f"--num_epochs={num_epochs:d}",
        f"--batches_per_epoch={batches_per_epoch:d}",
        "--loader_chunk_size=23925",
        f"--num_val_samples={num_val_samples:d}",
        f"--val_interval={val_interval:d}",
        f"--save_interval={save_interval:d}",
        f"--lr={lr}",
        f"--end_lr={end_lr}",
        f"--lr_schedule={lr_schedule}",
        f"--architecture={architecture}",
    ]
    if input_channels:
        args.append("--input_channels")
        args.extend(input_channels)
    else:
        args.extend(["--input_channels", f"q_{scale:d}"])
    if output_channels:
        args.append("--output_channels")
        args.extend(output_channels)
    else:
        args.extend(["--output_channels", f"q_total_forcing_{scale:d}"])
    if processing_size is not None:
        args.extend([f"--processing_size={processing_size:d}"])
    return lu.container_cmd_launch(args, time_limit="15:00:00", job_name="train", cpus=1, gpus=1, mem_gb=32, dependency_ids=dependency_ids)



for scale_set in SCALE_SETS:
    scale_set = tuple(sorted(scale_set, reverse=True))
    assert len(scale_set) == 2
    scale_set_underscore = "_".join(map(str, scale_set))
    for data_type, task_type in itertools.product(DATA_TYPES, ["downscale", "upscale"]):
        data_type_dir = base_out_dir / f"{data_type}-{task_type}-{scale_set_underscore}"
        lu.dry_run_mkdir(data_type_dir)
        data_files = DATA_FILES[data_type]

        if task_type == "downscale":
            input_channels = [f"q_scaled_{max(scale_set)}to{max(scale_set)}"]
            output_channels = [f"q_scaled_forcing_{max(scale_set)}to{min(scale_set)}"]
        elif task_type == "upscale":
            input_channels=[f"q_scaled_{max(scale_set)}to{max(scale_set)}", f"q_scaled_forcing_{max(scale_set)}to{min(scale_set)}"]
            output_channels=[f"residual:q_total_forcing_{max(scale_set)}-q_scaled_forcing_{max(scale_set)}to{min(scale_set)}"]
        else:
            raise ValueError(f"invalid task {task_type}")

        for arch_base, depth, repeat in itertools.product(ARCH_BASES, NET_DEPTHS, range(NUM_REPEATS)):
            train_params = LR_ASSIGNMENTS[arch_base]
            num_epochs = train_params.num_epochs
            peak_lr = train_params.peak_lr
            arch = f"shallow-gz-fcnn-v1-{arch_base}-l{depth:d}"
            out_dir = data_type_dir / f"train-shallow-{arch_base}-l{depth:d}-{repeat + 1:d}"
            lu.dry_run_mkdir(out_dir)
            launch_training(
                out_dir=out_dir,
                train_dir=data_files.train.parent,
                val_dir=data_files.val.parent,
                scale=max(scale_set),
                num_epochs=num_epochs,
                lr=peak_lr,
                lr_schedule=LR_SCHEDULE,
                architecture=arch,
                input_channels=input_channels,
                output_channels=output_channels,
            )
