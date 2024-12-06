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
SCHEDULE_TYPES = ["warmup1-cosine"]
DATA_TYPES = ["eddy"]


@dataclasses.dataclass
class SweepParams:
    peak_lrs: typing.List[float]
    num_epochs: typing.List[int]


params = {
    "small": SweepParams(
        peak_lrs=[0.002, 0.005, 0.01],
        num_epochs=[15, 25, 50],
    ),
    "medium": SweepParams(
        peak_lrs=sorted(set([0.0004, 0.0008])),
        num_epochs=[15, 25, 50],
    ),
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
base_out_dir = SCRATCH / "closure" / "run_outputs"/ f"lr-test-runs-{launch_time}"


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


scale_set = (128, 64)
scale_set_underscore = "_".join(map(str, scale_set))
for data_type in DATA_TYPES:
    data_type_dir = base_out_dir / data_type
    lu.dry_run_mkdir(data_type_dir)
    data_files = DATA_FILES[data_type]
    for arch_size in ["small", "medium"]:
        if arch_size == "small":
            base_arch = "gz-fcnn-v1"
        elif arch_size == "medium":
            base_arch = "gz-fcnn-v1-medium"
        arch_params = params[arch_size]
        for peak_lr, schedule_type, num_epochs in itertools.product(arch_params.peak_lrs, SCHEDULE_TYPES, arch_params.num_epochs):
            run_type_dir = data_type_dir / f"s-{schedule_type}_lr-{str(peak_lr).replace('.', '-')}_ne-{num_epochs}"
            lu.dry_run_mkdir(run_type_dir)
            for repeat in range(NUM_REPEATS):
                out_dir = run_type_dir / f"sequential-train-cnn-{scale_set_underscore}-{arch_size}-{repeat + 1:d}"
                lu.dry_run_mkdir(out_dir)
                launch_training(
                    out_dir=out_dir,
                    train_dir=data_files.train.parent,
                    val_dir=data_files.val.parent,
                    scale=min(scale_set),
                    num_epochs=num_epochs,
                    lr=peak_lr,
                    lr_schedule=schedule_type,
                    architecture=base_arch,
                    input_channels=[f"q_scaled_{max(scale_set)}to{max(scale_set)}", f"q_scaled_forcing_{max(scale_set)}to{min(scale_set)}"],
                    output_channels=[f"residual:q_total_forcing_{max(scale_set)}-q_scaled_forcing_{max(scale_set)}to{min(scale_set)}"],
                )
