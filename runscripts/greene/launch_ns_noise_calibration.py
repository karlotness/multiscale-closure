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
NUM_REPEATS = 10
SCALE_SETS = [(64, 32), (64, 48)]
NET_ARCH_SETS = [
    ("md8md8", ("shallow-gz-fcnn-v1-medium-l8", "shallow-gz-fcnn-v1-medium-l8")),
    ("psm4pmd8", ("shallow-gz-fcnn-v1-puresmall-l4", "shallow-gz-fcnn-v1-puremedium-l8")),
]
COARSEN_OPS = ["spectral"]
LR_SCHEDULE = "warmup1-cosine"
EPOCHS = 150
PEAK_LR = 0.001 * 0.75
END_LR = 0.0001
OPTIMIZER = "adam:eps=0.001"
OPTIM_WRAP = "none"
EVAL_TYPE = "best_val_loss"
BATCH_SIZE = 32
NOISE_LEVELS = [0.02, 0.035, 0.05, 0.075, 0.1]
NOISE_MODE = "simple-prob-clean"
NOISE_START_EPOCH = 6
NOISE_PROB_CLEAN = 0.25
NOISELESS_ALT_SOURCE = [False]
ROLLOUT_LENGTH = 15

VAR_BASE_STDDEV = {
    "ns_vort_64": (4.2444882,),
    "ns_vort_128": (4.4990916,),
    "ns_uv_64": (1.0, 1.0),
    "ns_uv_128": (1.0, 1.0),
}


@dataclasses.dataclass
class BaseNetRun:
    out_dir: pathlib.Path
    run_id: str


# Training and eval sets
DATA_FILES = TrainFileSet(
    train=VAST / "closure" / "data-ns-highre-20240519-185845" / "train" / "shuffled.hdf5",
    val=VAST / "closure" / "data-ns-highre-20240519-185845" / "val" / "data.hdf5",
    test=VAST / "closure" / "data-ns-highre-20240519-185845" / "test" / "data.hdf5",
)
assert all(p.is_file() for p in itertools.chain.from_iterable([df.train, df.val, df.test] for df in [DATA_FILES]))


launch_time = time.strftime("%Y%m%d-%H%M%S")
top_base_out_dir = SCRATCH / "closure" / "run_outputs"/ f"ns-noisecalibrate-{launch_time}"


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
    channel_coarsen_type="spectral",
    noise_vars={},
    wrap_optim="legacy",
    noise_mode="off",
    noise_start_epoch=1,
    noise_prob_clean=0.5,
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
        f"--channel_coarsen_type={channel_coarsen_type}",
        f"--wrap_optim={wrap_optim}",
        f"--noisy_batch_mode={noise_mode}",
        f"--simple_prob_clean={noise_prob_clean}",
        f"--simple_prob_clean_start_epoch={noise_start_epoch}",
    ]
    if noise_vars:
        args.append("--noise_specs")
        for k, v in noise_vars.items():
            if not k.endswith(f"_{scale:d}"):
                continue
            var_joined = ",".join(map(str, v))
            args.append(f"{k}={var_joined}")
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
    channel_coarsen_type="spectral",
    wrap_optim="legacy",
    noise_vars={},
    noise_mode="off",
    noise_start_epoch=1,
    noise_prob_clean=0.5,
    noiseless_alt_source=False,
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
                f"--channel_coarsen_type={channel_coarsen_type}",
                f"--wrap_optim={wrap_optim}",
                f"--noisy_batch_mode={noise_mode}",
                f"--simple_prob_clean={noise_prob_clean}",
                f"--simple_prob_clean_start_epoch={noise_start_epoch}",
            ]
        )
        if noiseless_alt_source:
            args.append("--noise_free_alt_source")
        if noise_vars:
            args.append("--noise_specs")
            for k, v in noise_vars.items():
                if not k.endswith(f"_{max(map(int, scales)):d}"):
                    continue
                var_joined = ",".join(map(str, v))
                args.append(f"{k}={var_joined}")
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


def launch_online_eval(*, out_file, eval_file, weight_files, rollout_length=7.5, dependency_ids=None):
    args = [
        "python",
        "online_ns_data_eval.py",
        f"--rollout_length_limit={rollout_length}",
        out_file,
        eval_file,
    ]
    args.extend(weight_files)
    return lu.container_cmd_launch(args, time_limit="20:00:00", job_name="eval", cpus=1, gpus=1, mem_gb=20, dependency_ids=dependency_ids)


# Launch baselines
for noise_level in NOISE_LEVELS:
    level_underscore = str(noise_level).replace(".", "_")
    base_out_dir = top_base_out_dir / f"noise-{level_underscore}"
    lu.dry_run_mkdir(base_out_dir)

    noise_vars = {
        k: tuple((s * noise_level)**2 for s in v)
        for k, v in VAR_BASE_STDDEV.items()
    }

    peak_sizes = sorted(set(max(scs) for scs in SCALE_SETS))
    for peak_scale, (arch_key, arch_parts), chan_coarse_op in itertools.product(
            peak_sizes, NET_ARCH_SETS, COARSEN_OPS,
    ):
        scale_set_dir = base_out_dir / f"ns-{peak_scale:d}"
        lu.dry_run_mkdir(scale_set_dir)
        arch_core_str = ":".join(arch_parts)
        arch = f"stacked-noscale-net-v1-{arch_core_str}"
        base_inputs = [f"ns_uv_{peak_scale:d}", f"ns_vort_{peak_scale:d}"]
        base_outputs = [f"ns_uv_corr_{peak_scale:d}"]
        arch_out_dir = scale_set_dir / arch_key
        lu.dry_run_mkdir(arch_out_dir)
        runs = []
        for repeat in range(NUM_REPEATS):
            out_dir = arch_out_dir / f"base-stacked-{chan_coarse_op}-{repeat + 1:d}"
            lu.dry_run_mkdir(out_dir)
            train_id = launch_training(
                out_dir=out_dir,
                train_dir=DATA_FILES.train.parent,
                val_dir=DATA_FILES.val.parent,
                scale=peak_scale,
                num_epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                optimizer=OPTIMIZER,
                lr=PEAK_LR,
                end_lr=END_LR,
                lr_schedule=LR_SCHEDULE,
                architecture=arch,
                input_channels=base_inputs,
                output_channels=base_outputs,
                channel_coarsen_type=chan_coarse_op,
                wrap_optim=OPTIM_WRAP,
                noise_vars=noise_vars,
                noise_mode=NOISE_MODE,
                noise_start_epoch=NOISE_START_EPOCH,
                noise_prob_clean=NOISE_PROB_CLEAN,
            )
            runs.append(
                BaseNetRun(
                    out_dir=out_dir,
                    run_id=train_id,
                )
            )
        online_eval_dir = scale_set_dir / "online-eval-results"
        lu.dry_run_mkdir(online_eval_dir)
        launch_online_eval(
            out_file=online_eval_dir / f"base-{peak_scale}-{arch_key}-{chan_coarse_op}-{EVAL_TYPE}.hdf5",
            eval_file=DATA_FILES.test,
            weight_files=[er.out_dir / "weights" / f"{EVAL_TYPE}.eqx" for er in runs],
            dependency_ids=[er.run_id for er in runs],
            rollout_length=ROLLOUT_LENGTH,
        )

    for scale_set, (arch_key, arch_parts), chan_coarse_op, noiseless_alt_source in itertools.product(
        SCALE_SETS, NET_ARCH_SETS, COARSEN_OPS, NOISELESS_ALT_SOURCE
    ):
        scale_set = tuple(sorted(scale_set, reverse=True))
        assert len(scale_set) == 2
        scale_set_underscore = "_".join(map(str, scale_set))
        noiseless_str = "-noiselessalt" if noiseless_alt_source else ""
        scale_set_dir = base_out_dir / f"ns-{scale_set_underscore}{noiseless_str}"
        lu.dry_run_mkdir(scale_set_dir)
        arch_out_dir = scale_set_dir / arch_key
        lu.dry_run_mkdir(arch_out_dir)
        runs = []
        for repeat in range(NUM_REPEATS):
            out_dir = arch_out_dir / f"seq-multiscale-{chan_coarse_op}-{repeat + 1:d}"
            lu.dry_run_mkdir(out_dir)
            train_id = launch_sequential_training(
                out_dir=out_dir,
                train_dir=DATA_FILES.train.parent,
                val_dir=DATA_FILES.val.parent,
                scales=scale_set,
                batch_size=BATCH_SIZE,
                optimizer=OPTIMIZER,
                num_epochs=[EPOCHS] * len(scale_set),
                lrs=[PEAK_LR] * len(scale_set),
                end_lr=END_LR,
                lr_schedule=LR_SCHEDULE,
                architectures=list(arch_parts),
                net_load_type=EVAL_TYPE,
                channel_coarsen_type=chan_coarse_op,
                wrap_optim=OPTIM_WRAP,
                noise_vars=noise_vars,
                noise_mode=NOISE_MODE,
                noise_start_epoch=NOISE_START_EPOCH,
                noise_prob_clean=NOISE_PROB_CLEAN,
                noiseless_alt_source=noiseless_alt_source,
            )
            runs.append(
                BaseNetRun(
                    out_dir=out_dir,
                    run_id=train_id,
                )
            )
        online_eval_dir = scale_set_dir / "online-eval-results"
        lu.dry_run_mkdir(online_eval_dir)
        launch_online_eval(
            out_file=online_eval_dir / f"seq-multiscale-{scale_set_underscore}-{arch_key}-{chan_coarse_op}-{EVAL_TYPE}.hdf5",
            eval_file=DATA_FILES.test,
            weight_files=[er.out_dir / "weights" / f"{EVAL_TYPE}.eqx" for er in runs],
            dependency_ids=[er.run_id for er in runs],
            rollout_length=ROLLOUT_LENGTH,
        )
