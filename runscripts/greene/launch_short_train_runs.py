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

TrainFileSet = dataclasses.make_dataclass("TrainFileSet", ["train", "val", "test"])

DRY_RUN = True
SCRATCH = pathlib.Path(os.environ["SCRATCH"]).resolve()

# General parameters
SCALES = {128, 96, 64}
NUM_REPEATS = 5
ONLINE_ALPHA_PARAMS = [1.0, 5.0, 10.0, 15.0]

# Training and eval sets
COARSE_OP = "op1"
DATA_FILES = {
    "eddy": TrainFileSet(
        train=SCRATCH / "closure" / "data-nowarmup" / f"train/{COARSE_OP}/shuffled.hdf5",
        val=SCRATCH / "closure" / "data-nowarmup" / f"val/{COARSE_OP}/data.hdf5",
        test=SCRATCH / "closure" / "data-nowarmup" / f"test/{COARSE_OP}/data.hdf5",
    ),
    "jet": TrainFileSet(
        train=SCRATCH / "closure" / "data-jet-nowarmup" / f"train/{COARSE_OP}/shuffled.hdf5",
        val=SCRATCH / "closure" / "data-jet-nowarmup" / f"val/{COARSE_OP}/data.hdf5",
        test=SCRATCH / "closure" / "data-jet-nowarmup" / f"test/{COARSE_OP}/data.hdf5",
    ),
}
assert all(p.is_file() for p in itertools.chain.from_iterable([df.train, df.val, df.test] for df in DATA_FILES.values()))

launch_time = time.strftime("%Y%m%d-%H%M%S")
base_out_dir = SCRATCH / "closure" / "run_outputs"/ f"short-train-runs-{launch_time}"
dry_run_counter = 123000

def sbatch_launch(args, *, dependency_ids=None, time_limit=None, job_name=None, cpus=1, gpus=0, mem_gb=25):
    global dry_run_counter
    extra_sbatch_args = []
    if dependency_ids:
        deps = ":".join(f"{did}" for did in dependency_ids)
        extra_sbatch_args.extend(["--dependency", f"afterok:{deps}", "--kill-on-invalid-dep", "yes"])
    if time_limit:
        extra_sbatch_args.append(f"--time={time_limit}")
    if job_name:
        extra_sbatch_args.extend(["--job-name", str(job_name)])
    if cpus < 1:
        raise ValueError(f"must request at least one cpu (got {cpus})")
    extra_sbatch_args.append(f"--cpus-per-task={cpus:d}")
    if gpus < 0:
        raise ValueError(f"invalid number of gpus {gpus}")
    elif gpus > 0:
        extra_sbatch_args.extend([f"--gres=gpu:{gpus:d}"])
    extra_sbatch_args.append(f"--mem={mem_gb:d}GB")
    args = ["--parsable"] + extra_sbatch_args + [str(a) for a in args]
    print("sbatch", " ".join(f"'{a}'" for a in args))
    if not DRY_RUN:
        proc = subprocess.run(["sbatch"] + args, check=True, capture_output=True)
        output = codecs.decode(proc.stdout, encoding="utf8").strip()
        m = re.match(r"^\s*(?P<jobid>[^;]+)(?:;|$)", output)
        if m:
            time.sleep(0.5)
            return m.group("jobid").strip()
        else:
            raise ValueError(f"could not parse {output}")
    else:
        dry_run_counter += 1
        return str(dry_run_counter)


def container_cmd_launch(args, *, dependency_ids=None, time_limit=None, job_name=None, cpus=1, gpus=0, mem_gb=25):
    return sbatch_launch(
        ["run-container-checkout-cmd.sh", ("cuda" if gpus > 0 else "cpu")] + args,
        dependency_ids=dependency_ids,
        time_limit=time_limit,
        job_name=job_name,
        cpus=cpus,
        gpus=gpus,
        mem_gb=mem_gb,
    )


def dry_run_mkdir(dir_path):
    print("mkdir -p", f"'{pathlib.Path(dir_path).resolve()}'")
    if not DRY_RUN:
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


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
    return container_cmd_launch(args, time_limit="15:00:00", job_name="noise-train", cpus=1, gpus=1, mem_gb=25, dependency_ids=dependency_ids)


def launch_sequential_training(
    *,
    out_dir,
    train_dir,
    val_dir,
    scales,
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
    dependency_ids=None,
    net_load_type="best_loss",
):
    scales = [str(s) for s in sorted(set(scales))]
    if len(scales) < 2:
        raise ValueError(f"need at least 2 scales, got {len(scales)}")
    for train_step in range(len(scales)):
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
                f"--architecture={architecture}",
                f"--optimizer={optimizer}",
                f"--batch_size={batch_size:d}",
                "--loader_chunk_size=23925",
                f"--num_epochs={num_epochs:d}",
                f"--batches_per_epoch={batches_per_epoch:d}",
                f"--num_val_samples={num_val_samples:d}",
                f"--val_interval={val_interval:d}",
                f"--save_interval={save_interval:d}",
                f"--lr={lr}",
                f"--end_lr={end_lr}",
                f"--lr_schedule={lr_schedule}",
                f"--net_load_type={net_load_type}",
            ]
        )
        final_launch_id = container_cmd_launch(
            args,
            time_limit="15:00:00",
            job_name="seq-train",
            cpus=1,
            mem_gb=25,
            gpus=1,
            dependency_ids=dependency_ids
        )
        dependency_ids = [final_launch_id]
    # Combine with sequential_to_cascade
    args = [
        "python",
        "sequential_to_cascaded.py",
        out_dir
    ]
    final_launch_id = container_cmd_launch(args, time_limit="0:30:00", job_name="seq-to-cascade", cpus=4, gpus=0, mem_gb=4, dependency_ids=dependency_ids)
    return final_launch_id


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
    return container_cmd_launch(args, time_limit="15:00:00", job_name="eval-plots", cpus=1, gpus=1, mem_gb=20, dependency_ids=dependency_ids)


for run_type in ["eddy", "jet"]:
    run_type_dir = base_out_dir / run_type
    online_eval_dir = run_type_dir / "online-eval-results"
    dry_run_mkdir(online_eval_dir)
    data_files = DATA_FILES[run_type]
    for arch_size in ["small", "medium"]:
        if arch_size == "small":
            base_arch = "gz-fcnn-v1"
            stacked_base = "stacked-gz-fcnn-v2"
        elif arch_size == "medium":
            base_arch = "gz-fcnn-v1-medium"
            stacked_base = "stacked-gz-fcnn-v2-medium"
        # Launch sequential runs
        for scale_set in itertools.chain.from_iterable(itertools.combinations(SCALES, sz) for sz in range(2, 1 + min(3, len(SCALES)))):
            arch = base_arch
            scale_set = tuple(sorted(scale_set, reverse=True))
            scale_set_underscore = "_".join(map(str, scale_set))
            train_runs = []
            eval_weight_files = []
            for repeat in range(NUM_REPEATS):
                out_dir = run_type_dir / f"sequential-train-cnn-{scale_set_underscore}-{arch_size}-{repeat + 1:d}"
                dry_run_mkdir(out_dir)
                job_id = launch_sequential_training(
                    out_dir=out_dir,
                    train_dir=data_files.train.parent,
                    val_dir=data_files.val.parent,
                    scales=scale_set,
                    lr=0.001,
                    architecture=arch,
                    net_load_type="checkpoint",
                )
                train_runs.append(job_id)
                eval_weight_files.append(out_dir / "weights" / "checkpoint.eqx")
            # Online evaluation
            for alpha_level in ONLINE_ALPHA_PARAMS:
                alpha_underscore = str(alpha_level).replace(".", "_")
                launch_online_eval(
                    out_file=online_eval_dir / f"sequential-train-cnn-{scale_set_underscore}-{arch_size}-alpha{alpha_underscore}-checkpoint.hdf5",
                    eval_file=data_files.test,
                    weight_files=eval_weight_files,
                    dependency_ids=train_runs,
                    param_alpha=alpha_level,
                )
        # Launch stacked baselines
        for depth in [2, 3]:
            for scale in SCALES:
                if scale == min(SCALES):
                    continue
                arch = f"{stacked_base}-d{depth:d}"
                train_runs = []
                eval_weight_files = []
                for repeat in range(NUM_REPEATS):
                    out_dir = run_type_dir / f"{arch}-{scale:d}-{repeat + 1:d}"
                    dry_run_mkdir(out_dir)
                    job_id = launch_training(
                        out_dir=out_dir,
                        train_dir=data_files.train.parent,
                        val_dir=data_files.val.parent,
                        scale=scale,
                        lr=0.001,
                        architecture=arch,
                    )
                    train_runs.append(job_id)
                    eval_weight_files.append(out_dir / "weights" / "checkpoint.eqx")
                # Online evaluation
                for alpha_level in ONLINE_ALPHA_PARAMS:
                    alpha_underscore = str(alpha_level).replace(".", "_")
                    launch_online_eval(
                        out_file=online_eval_dir / f"{arch}-{scale:d}-alpha{alpha_underscore}-checkpoint.hdf5",
                        eval_file=data_files.test,
                        weight_files=eval_weight_files,
                        dependency_ids=train_runs,
                        param_alpha=alpha_level,
                    )
