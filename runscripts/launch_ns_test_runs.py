import re
import time
import pathlib
import os
import itertools
import dataclasses
import launch_utils as lu

TrainFileSet = dataclasses.make_dataclass("TrainFileSet", ["train", "val", "test"])

SCRATCH = pathlib.Path(os.environ["SCRATCH"]).resolve()
VAST = pathlib.Path(os.environ["VAST"]).resolve()

# lu.enable_real_launch()

# General parameters
NUM_REPEATS = 5
SCALE_SETS = [(64, 32), (64, 48), (128, 96), (128, 64)]
NET_DEPTHS = [8]
ARCH_BASES = ["medium"]
LR_SCHEDULE = "warmup1-cosine"
ONLINE_ALPHA_PARAMS = [1.0, 5.0, 10.0, 15.0]

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
    "small": SweepParams(peak_lr=0.001, num_epochs=150),
    "puresmall": SweepParams(peak_lr=0.001, num_epochs=150),
    "medium": SweepParams(peak_lr=0.0004, num_epochs=75),
    "puremedium": SweepParams(peak_lr=0.0004, num_epochs=75),
}


# Training and eval sets
DATA_FILES = TrainFileSet(
    train=VAST / "closure" / "data-ns-20240425-051048" / "train" / "shuffled.hdf5",
    val=VAST / "closure" / "data-ns-20240425-051048" / "val" / "data.hdf5",
    test=VAST / "closure" / "data-ns-20240425-051048" / "test" / "data.hdf5",
)
assert all(p.is_file() for p in itertools.chain.from_iterable([df.train, df.val, df.test] for df in [DATA_FILES]))


launch_time = time.strftime("%Y%m%d-%H%M%S")
base_out_dir = SCRATCH / "closure" / "run_outputs"/ f"ns-test-runs-{launch_time}"


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
        "--optimizer", str(optimizer),
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
        args.extend(["--input_channels", f"ns_uv_{scale:d}", f"ns_vort_{scale:d}"])
    if output_channels:
        args.append("--output_channels")
        args.extend(output_channels)
    else:
        args.extend(["--output_channels", f"ns_uv_corr_{scale:d}"])
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
                "--optimizer", str(optimizer),
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


def launch_online_eval(*, out_file, eval_file, weight_files, dependency_ids=None):
    args = [
        "python",
        "online_ns_data_eval.py",
        out_file,
        eval_file,
    ]
    args.extend(weight_files)
    return lu.container_cmd_launch(args, time_limit="15:00:00", job_name="eval", cpus=1, gpus=1, mem_gb=20, dependency_ids=dependency_ids)


scale_set_lengths = set(len(scs) for scs in SCALE_SETS)
if len(scale_set_lengths) != 1:
    raise ValueError("Scale sets should all have the same length")
scale_set_length = scale_set_lengths.pop()
for peak_scale in set(max(scs) for scs in SCALE_SETS):
    scale_set_dir = base_out_dir / f"ns-{peak_scale:d}"
    lu.dry_run_mkdir(scale_set_dir)
    # Launch base runs
    base_runs = {}
    base_inputs = [f"ns_uv_{peak_scale:d}", f"ns_vort_{peak_scale:d}"]
    base_outputs = [f"ns_uv_corr_{peak_scale:d}"]
    for arch_base, base_depth in itertools.product(ARCH_BASES, NET_DEPTHS):
        # BASELINE RUNS
        arch_out_dir = scale_set_dir / f"{arch_base}-{base_depth}"
        base_key = (arch_base, base_depth)
        base_runs[base_key] = []
        train_params = LR_ASSIGNMENTS[arch_base]
        num_epochs = train_params.num_epochs
        peak_lr = train_params.peak_lr
        base_arch = f"shallow-gz-fcnn-v1-{arch_base}-l{base_depth:d}"
        arch_str = ":".join(itertools.repeat(base_arch, scale_set_length))
        arch = f"stacked-noscale-net-v1-{arch_str}"
        for repeat in range(NUM_REPEATS):
            out_dir = arch_out_dir / f"base-stacked-{repeat + 1:d}"
            lu.dry_run_mkdir(out_dir)
            train_id = launch_training(
                out_dir=out_dir,
                train_dir=DATA_FILES.train.parent,
                val_dir=DATA_FILES.val.parent,
                scale=peak_scale,
                num_epochs=num_epochs,
                batch_size=32,
                optimizer="adam:eps=0.0001",
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

    # Launch online evaluation
    online_eval_dir = scale_set_dir / "online-eval-results"
    lu.dry_run_mkdir(online_eval_dir)
    for base_key, eval_records in base_runs.items():
        net_seq_str = "_".join(map(str, base_key))
        eval_weight_files = [er.out_dir / "weights" / "best_val_loss.eqx" for er in eval_records]
        train_runs = [er.run_id for er in eval_records]
        launch_online_eval(
            out_file=online_eval_dir / f"base-{peak_scale}-{net_seq_str}-best_val_loss.hdf5",
            eval_file=DATA_FILES.test,
            weight_files=eval_weight_files,
            dependency_ids=train_runs,
        )

for scale_set in SCALE_SETS:
    scale_set = tuple(sorted(scale_set, reverse=True))
    assert len(scale_set) == 2
    scale_set_underscore = "_".join(map(str, scale_set))
    scale_set_dir = base_out_dir / f"ns-{scale_set_underscore}"
    lu.dry_run_mkdir(scale_set_dir)
    # Launch base runs
    seq_runs = {}
    base_inputs = [f"ns_uv_{max(scale_set):d}", f"ns_vort_{max(scale_set):d}"]
    base_outputs = [f"ns_uv_corr_{max(scale_set):d}"]
    for arch_base, base_depth in itertools.product(ARCH_BASES, NET_DEPTHS):
        # BASELINE RUNS
        arch_out_dir = scale_set_dir / f"{arch_base}-{base_depth}"
        base_key = (arch_base, base_depth)
        train_params = LR_ASSIGNMENTS[arch_base]
        num_epochs = train_params.num_epochs
        peak_lr = train_params.peak_lr
        base_arch = f"shallow-gz-fcnn-v1-{arch_base}-l{base_depth:d}"
        # SEQUENTIAL RUNS
        seq_runs[base_key] = []
        for repeat in range(NUM_REPEATS):
            out_dir = arch_out_dir / f"seq-multiscale-{repeat + 1:d}"
            lu.dry_run_mkdir(out_dir)
            architecture_list = [base_arch] * len(scale_set)
            train_id = launch_sequential_training(
                out_dir=out_dir,
                train_dir=DATA_FILES.train.parent,
                val_dir=DATA_FILES.val.parent,
                scales=scale_set,
                batch_size=32,
                optimizer="adam:eps=0.0001",
                num_epochs=[num_epochs] * len(scale_set),
                lrs=[peak_lr] * len(scale_set),
                lr_schedule=LR_SCHEDULE,
                architectures=architecture_list,
                net_load_type="best_val_loss",
            )
            seq_runs[base_key].append(
                BaseNetRun(
                    out_dir=out_dir,
                    run_id=train_id,
                )
            )
    # Launch online evaluation
    online_eval_dir = scale_set_dir / "online-eval-results"
    lu.dry_run_mkdir(online_eval_dir)
    for base_key, eval_records in seq_runs.items():
        net_seq_str = "_".join(map(str, base_key))
        eval_weight_files = [er.out_dir / "weights" / "best_val_loss.eqx" for er in eval_records]
        train_runs = [er.run_id for er in eval_records]
        launch_online_eval(
            out_file=online_eval_dir / f"seq-{scale_set_underscore}-{net_seq_str}-best_val_loss.hdf5",
            eval_file=DATA_FILES.test,
            weight_files=eval_weight_files,
            dependency_ids=train_runs,
        )
