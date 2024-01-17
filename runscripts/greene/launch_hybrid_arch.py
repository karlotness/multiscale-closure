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
SCALE_SETS = [(128, 96), (96, 64)]
NET_DEPTHS = [8, 4]
ARCH_BASES = ["puresmall", "small", "puremedium", "medium"]
LR_SCHEDULE = "warmup1-cosine"
ONLINE_ALPHA_PARAMS = [1.0, 5.0, 10.0, 15.0]


def iter_increasing_nets(prev_arch, prev_depth):
    def key(a, d):
        return (ARCH_BASES.index(a), d)
    base_key = key(prev_arch, prev_depth)
    yield from filter(lambda v: key(*v) >= base_key, itertools.product(ARCH_BASES, NET_DEPTHS))


def rec_increasing_nets(prev_nets, depth):
    if depth <= 1:
        yield prev_nets
    else:
        for net in iter_increasing_nets(*prev_nets[-1]):
            yield from rec_increasing_nets(prev_nets + (net,), depth - 1)


def iter_net_sequences(base_arch, base_depth, seq_len):
    yield from rec_increasing_nets(((base_arch, base_depth),), seq_len)


@dataclasses.dataclass
class SweepParams:
    peak_lr: float
    num_epochs: int


@dataclasses.dataclass
class BaseNetRun:
    out_dir: pathlib.Path
    run_id: str


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
base_out_dir = SCRATCH / "closure" / "run_outputs"/ f"hybrid-arch-runs-{launch_time}"


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


def launch_sequential_training(
    *,
    out_dir,
    train_dir,
    val_dir,
    scales,
    optimizer="adam",
    batch_size=64,
    num_epochs=[50],
    batches_per_epoch=374,
    num_val_samples=300,
    val_interval=1,
    save_interval=10,
    lrs=[0.001],
    end_lr=0.0,
    lr_schedule="ross22",
    architectures=["gz-fcnn-v1"],
    dependency_ids=None,
    net_load_type="best_loss",
    skip_nets=0,
):
    scales = [str(s) for s in sorted(set(scales))]
    if len(scales) < 2:
        raise ValueError(f"need at least 2 scales, got {len(scales)}")
    for train_step in range(len(scales)):
        if train_step < skip_nets:
            continue
        args = [
            "python",
            "sequential_train.py",
            out_dir,
            train_dir,
            val_dir,
            str(train_step),
        ]
        args.extend(scales)
        args.extend(
            [
                f"--optimizer={optimizer}",
                f"--batch_size={batch_size:d}",
                "--loader_chunk_size=23925",
                f"--num_epochs={num_epochs[train_step]:d}",
                f"--batches_per_epoch={batches_per_epoch:d}",
                f"--num_val_samples={num_val_samples:d}",
                f"--val_interval={val_interval:d}",
                f"--save_interval={save_interval:d}",
                f"--lr={lrs[train_step]}",
                f"--end_lr={end_lr}",
                f"--lr_schedule={lr_schedule}",
                f"--net_load_type={net_load_type}",
            ]
        )
        assert len(architectures) == len(scales)
        args.append("--architecture")
        args.extend(architectures)
        final_launch_id = lu.container_cmd_launch(
            args,
            time_limit="15:00:00",
            job_name="seq-train",
            cpus=2,
            mem_gb=32,
            gpus=1,
            dependency_ids=dependency_ids,
        )
        dependency_ids = [final_launch_id]
    # Combine with sequential_to_cascade
    args = [
        "python",
        "sequential_to_cascaded.py",
        out_dir
    ]
    return lu.container_cmd_launch(args, time_limit="0:30:00", job_name="seq-to-cascade", cpus=1, gpus=0, mem_gb=4, dependency_ids=dependency_ids)


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
    # Launch base runs
    base_runs = {}
    base_inputs = [f"q_{max(scale_set)}"]
    base_outputs = [f"q_scaled_forcing_{max(scale_set)}to{min(scale_set)}"]
    for arch_base, base_depth in itertools.product(ARCH_BASES, NET_DEPTHS):
        base_key = (arch_base, base_depth)
        base_runs[base_key] = []
        train_params = LR_ASSIGNMENTS[arch_base]
        num_epochs = train_params.num_epochs
        peak_lr = train_params.peak_lr
        arch = f"shallow-gz-fcnn-v1-{arch_base}-l{base_depth:d}"
        for repeat in range(NUM_REPEATS):
            out_dir = scale_set_dir / f"train-base-{arch}-{repeat + 1:d}"
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
            base_runs[base_key].append(
                BaseNetRun(
                    out_dir=out_dir,
                    run_id=train_id,
                )
            )
    # Launch follow-up sequential runs
    train_eval_groups = {}
    for base_key in base_runs:
        for net_seq in iter_net_sequences(base_key[0], base_key[1], len(scale_set)):
            if net_seq not in train_eval_groups:
                train_eval_groups[net_seq] = []
            for repeat, base_record in enumerate(base_runs[base_key]):
                net_seq_str = "_".join(f"{a}{b}" for a, b in net_seq)
                out_dir = scale_set_dir / f"train-seq-{net_seq_str}-{repeat + 1:d}"
                lu.dry_run_mkdir(out_dir)
                # Copy the base network into position
                copy_job = lu.copy_dir_launch(
                    src=base_record.out_dir,
                    dst=out_dir / "net0",
                    dependency_ids=[base_record.run_id],
                    job_name="copybase"
                )
                # Launch sequential training
                architecture_list = []
                num_epochs_list = []
                lrs_list = []
                for (arch_base, depth) in net_seq:
                    train_params = LR_ASSIGNMENTS[arch_base]
                    architecture_list.append(f"shallow-gz-fcnn-v1-{arch_base}-l{depth:d}")
                    num_epochs_list.append(train_params.num_epochs)
                    lrs_list.append(train_params.peak_lr)
                train_job = launch_sequential_training(
                    out_dir=out_dir,
                    train_dir=data_files.train.parent,
                    val_dir=data_files.val.parent,
                    scales=scale_set,
                    num_epochs=num_epochs_list,
                    lrs=lrs_list,
                    lr_schedule=LR_SCHEDULE,
                    architectures=architecture_list,
                    dependency_ids=[copy_job],
                    net_load_type="best_val_loss",
                    skip_nets=1,
                )
                train_eval_groups[net_seq].append(
                    BaseNetRun(
                        out_dir=out_dir,
                        run_id=train_job,
                    )
                )
    # Launch online evaluation
    online_eval_dir = scale_set_dir / "online-eval-results"
    lu.dry_run_mkdir(online_eval_dir)
    for net_seq, eval_recs in train_eval_groups.items():
        net_seq_str = "_".join(f"{a}{b}" for a, b in net_seq)
        # Online evaluation
        for alpha_level in ONLINE_ALPHA_PARAMS:
            alpha_underscore = str(alpha_level).replace(".", "_")
            eval_weight_files = [er.out_dir / "weights" / "best_val_loss.eqx" for er in eval_recs]
            train_runs = [er.run_id for er in eval_recs]
            launch_online_eval(
                out_file=online_eval_dir / f"sequential-train-cnn-{scale_set_underscore}-{net_seq_str}-alpha{alpha_underscore}-best_val_loss.hdf5",
                eval_file=data_files.test,
                weight_files=eval_weight_files,
                dependency_ids=train_runs,
                param_alpha=alpha_level,
            )
