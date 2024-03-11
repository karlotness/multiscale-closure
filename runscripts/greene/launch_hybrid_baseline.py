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
VAST = pathlib.Path(os.environ["VAST"]).resolve()

# lu.enable_real_launch()

# General parameters
NUM_REPEATS = 4
DATA_TYPES = ["eddy"]
SCALE_SETS = [(128, 96), (96, 64)]
LR_SCHEDULE = "warmup1-cosine"
ONLINE_ALPHA_PARAMS = [1.0, 5.0, 10.0, 15.0]


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


# TOP 5 arches 128->96
# psm4pmd8  seq  96,128
# pmd8pmd8  seq  96,128
# psm8md8   seq  96,128
# pmd8md8   seq  96,128
# sm8sm8    seq  96,128

# TOP 5 arches 96->64
# psm4pmd8  seq  64,96
# md4md8    seq  64,96
# sm4pmd8   seq  64,96
# sm4md8    seq  64,96
# pmd8md8   seq  64,96


def gen_arch_desc(desc):
    size_map = {
        "psm": "puresmall",
        "pmd": "puremedium",
        "md": "medium",
        "sm": "small",
        "lg": "large",
    }
    d1, d2 = itertools.tee(filter(bool, re.split(r"(\d+)", desc)), 2)
    arches = []
    for sz, lrs in zip(itertools.islice(d1, None, None, 2), itertools.islice(d2, 1, None, 2)):
        arches.append(f"shallow-gz-fcnn-v1-{size_map[sz]}-l{lrs}")
    return tuple(arches)


def largest_arch(desc):
    order = ["psm", "sm", "pmd", "md", "lg"]
    size_map = {
        "psm": "puresmall",
        "pmd": "puremedium",
        "md": "medium",
        "sm": "small",
        "lg": "large",
    }
    return size_map[max(filter(bool, re.split(r"\d+", desc)), key=order.index)]


def to_stacked_arch_string(arches):
    arch_str = ":".join(arches)
    return f"stacked-noscale-net-v1-{arch_str}"


@dataclasses.dataclass
class ArchStr:
    short_name: str
    arch: str
    largest_arch: str


WINNING_ARCHES = {
    (128, 96): [
        ArchStr(
            short_name=name,
            arch=to_stacked_arch_string(gen_arch_desc(name)),
            largest_arch=largest_arch(name),
        )
        for name in ["psm4pmd8", "pmd8pmd8", "psm8md8", "pmd8md8", "sm8sm8"]
    ],
    (96, 64): [
        ArchStr(
            short_name=name,
            arch=to_stacked_arch_string(gen_arch_desc(name)),
            largest_arch=largest_arch(name),
        )
        for name in ["psm4pmd8", "md4md8", "sm4pmd8", "sm4md8", "pmd8md8"]
    ],
}


# Training and eval sets
COARSE_OP = "op1"
DATA_FILES = {
    "eddy": TrainFileSet(
        train=VAST / "closure" / "data-nowarmup" / f"train/{COARSE_OP}/shuffled.hdf5",
        val=VAST / "closure" / "data-nowarmup" / f"val/{COARSE_OP}/data.hdf5",
        test=VAST / "closure" / "data-eddyonly" / f"test/{COARSE_OP}/data.hdf5",
    ),
    "jet": TrainFileSet(
        train=SCRATCH / "closure" / "data-jet-nowarmup" / f"train/{COARSE_OP}/shuffled.hdf5",
        val=SCRATCH / "closure" / "data-jet-nowarmup" / f"val/{COARSE_OP}/data.hdf5",
        test=SCRATCH / "closure" / "data-jet-nowarmup" / f"test/{COARSE_OP}/data.hdf5",
    ),
}
assert all(p.is_file() for p in itertools.chain.from_iterable([df.train, df.val, df.test] for df in DATA_FILES.values()))

launch_time = time.strftime("%Y%m%d-%H%M%S")
base_out_dir = SCRATCH / "closure" / "run_outputs"/ f"hybrid-baseline-runs-{launch_time}"


@dataclasses.dataclass
class BaseNetRun:
    out_dir: pathlib.Path
    run_id: str


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
    return lu.container_cmd_launch(args, time_limit="15:00:00", job_name="train", cpus=2, gpus=1, mem_gb=32, dependency_ids=dependency_ids)


def launch_online_eval(*, out_file, eval_file, weight_files, dependency_ids=None, param_alpha=None):
    args = [
        "python",
        "online_data_eval.py",
        "--corr_num_samples=0",
    ]
    if param_alpha is not None:
        args.extend(["--param_alpha", str(param_alpha)])
    args.extend([out_file, eval_file])
    args.extend(weight_files)
    return lu.container_cmd_launch(args, time_limit="15:00:00", job_name="eval-plots", cpus=1, gpus=1, mem_gb=25, dependency_ids=dependency_ids)


for data_type, scale_set in itertools.product(DATA_TYPES, SCALE_SETS):
    data_files = DATA_FILES[data_type]
    scale_set = tuple(sorted(scale_set, reverse=True))
    assert len(scale_set) == 2
    scale_set_underscore = "_".join(map(str, scale_set))
    scale_set_dir = base_out_dir / f"{data_type}-{scale_set_underscore}"
    lu.dry_run_mkdir(scale_set_dir)
    online_eval_dir = scale_set_dir / "online-eval-results"
    lu.dry_run_mkdir(online_eval_dir)
    # Launch baseline runs
    for arch_record in WINNING_ARCHES[scale_set]:
        train_params = LR_ASSIGNMENTS[arch_record.largest_arch]
        num_epochs = train_params.num_epochs
        peak_lr = train_params.peak_lr
        arch = arch_record.arch
        base_inputs = [f"q_{max(scale_set)}"]
        base_outputs = [f"q_total_forcing_{max(scale_set)}"]
        eval_recs = []
        for repeat in range(NUM_REPEATS):
            out_dir = scale_set_dir / f"train-base-{arch_record.short_name}-{repeat + 1:d}"
            lu.dry_run_mkdir(out_dir)
            train_id = launch_training(
                out_dir=out_dir,
                train_dir=data_files.train.parent,
                val_dir=data_files.val.parent,
                scale=max(scale_set),
                num_epochs=num_epochs,
                lr=peak_lr,
                lr_schedule=LR_SCHEDULE,
                architecture=arch,
                input_channels=base_inputs,
                output_channels=base_outputs,
            )
            eval_recs.append(
                BaseNetRun(
                    out_dir=out_dir,
                    run_id=train_id,
                )
            )
        # Launch online evaluation runs
        for alpha_level in ONLINE_ALPHA_PARAMS:
            alpha_underscore = str(alpha_level).replace(".", "_")
            eval_weight_files = [er.out_dir / "weights" / "best_val_loss.eqx" for er in eval_recs]
            train_runs = [er.run_id for er in eval_recs]
            launch_online_eval(
                out_file=online_eval_dir / f"hybrid-base-{scale_set_underscore}-{arch_record.short_name}-alpha{alpha_underscore}-best_val_loss.hdf5",
                eval_file=data_files.test,
                weight_files=eval_weight_files,
                dependency_ids=train_runs,
                param_alpha=alpha_level,
            )
