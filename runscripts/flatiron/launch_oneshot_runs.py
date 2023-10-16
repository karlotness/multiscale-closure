import subprocess
import codecs
import re
import time
import pathlib
import math
import os
import itertools
import operator

DRY_RUN = True
SCRATCH = pathlib.Path(os.environ["SCRATCH"]).resolve()

# General parameters
NUM_REPEATS = 7
NOISE_START_EPOCH = 10
SCALE = 96
CANDIDATES_PER_EPOCH = operator.index(round((2 * 87 * 100) / (50 - NOISE_START_EPOCH)))
NET_STEPS = [5, 10, 30, 75]

# Mapping of steps to layer variances
STEP_LAYER_VARS = dict([
    (1, [0.00797639, 0.0038702]),
    (5, [0.01694924, 0.00856806]),
    (10, [0.03143954, 0.01700634]),
    (20, [0.07309306, 0.04431301]),
    (30, [0.12395001, 0.07433827]),
    (50, [0.22673633, 0.15860094]),
    (75, [0.36116423, 0.28159175]),
])


launch_time = time.strftime("%Y%m%d-%H%M%S")
base_out_dir = SCRATCH / "closure" / "run_outputs"/ f"oneshot-noise-nets-scale{SCALE:d}-{launch_time}"

if SCALE == 64:
    train_file = SCRATCH / "closure/data-rand-eddytojet/factor-1_0/train100/op1/shuffled.hdf5"
    live_data_file = SCRATCH / "closure/data-smallstep-rand-eddytojet/factor-1_0/train100/op1/shuffled.hdf5"
    val_file = SCRATCH / "closure/data-rand-eddytojet/factor-1_0/val/op1/data.hdf5"
    test_train_file = SCRATCH / "closure/data-rand-eddytojet/factor-1_0/test-trainset/op1/data.hdf5"
    test_heldout_file = SCRATCH / "closure/data-rand-eddytojet/factor-1_0/test/op1/data.hdf5"
    assert test_train_file.is_file()
elif SCALE == 96:
    train_file = SCRATCH / "closure/data-size96-rand-eddytojet/largestep/factor-1_0/train100/op1/shuffled.hdf5"
    live_data_file = SCRATCH / "closure/data-size96-rand-eddytojet/smallstep/factor-1_0/train100/op1/shuffled.hdf5"
    val_file = SCRATCH / "closure/data-size96-rand-eddytojet/largestep/factor-1_0/val/op1/data.hdf5"
    test_heldout_file = SCRATCH / "closure/data-size96-rand-eddytojet/largestep/factor-1_0/test/op1/data.hdf5"

assert all(p.is_file() for p in [train_file, live_data_file, val_file, test_heldout_file])

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
        extra_sbatch_args.extend([f"--gpus={gpus:d}", "--partition=gpu"])
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


def launch_training(*, out_dir, train_dir, val_dir, scale=SCALE, live_gen_net_steps=5, live_gen_mode="schedule-only", live_gen_candidates=0, live_gen_data_dir=None, architecture="gz-fcnn-v1", noise_vars=None, switch_live_set_interval=None, live_gen_start_epoch=1, live_gen_interval=1, live_gen_winners=None, dependency_ids=None):
    if live_gen_winners is None:
        live_gen_winners = live_gen_candidates
    args = [
        "python",
        "train.py",
        out_dir,
        train_dir,
        val_dir,
        "--optimizer=adam",
        "--batch_size=64",
        "--num_epochs=50",
        "--batches_per_epoch=374",
        "--loader_chunk_size=23925",
        "--num_val_samples=100",
        "--val_interval=1",
        "--save_interval=10",
        "--lr=0.001",
        "--end_lr=0.001",
        "--lr_schedule=ross22",
        f"--architecture={architecture}",
        "--input_channels", f"q_{scale:d}", f"rek_{scale:d}", f"delta_{scale:d}", f"beta_{scale:d}",
        "--output_channels", f"q_total_forcing_{scale:d}",
        # Fixed live generation arguments
        f"--live_gen_start_epoch={live_gen_start_epoch:d}",
        f"--live_gen_interval={live_gen_interval:d}",
        f"--live_gen_candidates={live_gen_candidates}",
        f"--live_gen_winners={live_gen_winners:d}",
        "--live_gen_mode", live_gen_mode,
        f"--live_gen_net_steps={live_gen_net_steps:d}",
    ]
    if noise_vars:
        spec_arg_v = ",".join(map(str, noise_vars))
        args.extend(["--noise_specs", f"q_{scale:d}={spec_arg_v}"])
    if switch_live_set_interval is not None:
        args.extend(["--live_gen_switch_set_interval", f"{switch_live_set_interval:d}"])
    if live_gen_data_dir:
        args.extend(["--live_gen_base_data", live_gen_data_dir])
    return container_cmd_launch(args, time_limit="15:00:00", job_name="noise-train", cpus=1, gpus=1, mem_gb=25, dependency_ids=dependency_ids)


def launch_online_eval(*, out_file, eval_file, weight_files, dependency_ids=None):
    args = [
        "python",
        "online_data_eval.py",
        "--corr_num_samples=0",
        out_file,
        eval_file,
    ]
    args.extend(weight_files)
    return container_cmd_launch(args, time_limit="15:00:00", job_name="eval-plots", cpus=1, gpus=1, mem_gb=20, dependency_ids=dependency_ids)


dry_run_mkdir(base_out_dir)
for noise_type, steps in itertools.chain(
    [("noiseless", 5)],
    itertools.product(
        ["netroll", "gaussian"],
        NET_STEPS,
    )
):
    if noise_type == "noiseless":
        type_dir = base_out_dir / noise_type
    else:
        type_dir = base_out_dir / noise_type / f"steps{steps:d}"
    online_eval_dir = type_dir / "poster-plot-data"
    # Configure run parameters
    if noise_type == "noiseless":
        gen_mode = "schedule-only"
        num_candidates = 0
        spec_vars = None
        gen_start_epoch = 1
        gen_interval = 1
    elif noise_type == "netroll":
        gen_mode = "network-noise-onestep"
        num_candidates = CANDIDATES_PER_EPOCH
        spec_vars = None
        gen_start_epoch = 10
        gen_interval = 1
    elif noise_type == "gaussian":
        gen_mode = "schedule-only"
        num_candidates = CANDIDATES_PER_EPOCH
        spec_vars = STEP_LAYER_VARS[steps]
        gen_start_epoch = 10
        gen_interval = 1
    else:
        raise ValueError(f"invalid type {noise_type}")
    batch_train_ids = []
    batch_eval_weights = []
    dry_run_mkdir(type_dir)
    for repeat in range(NUM_REPEATS):
        out_dir = type_dir / f"net{repeat}"
        net_run_id = launch_training(
            out_dir=out_dir,
            train_dir=train_file.parent,
            val_dir=val_file.parent,
            live_gen_data_dir=live_data_file.parent,
            live_gen_net_steps=steps,
            live_gen_mode=gen_mode,
            live_gen_candidates=num_candidates,
            noise_vars=spec_vars,
            live_gen_start_epoch=gen_start_epoch,
            live_gen_interval=gen_interval,
        )
        batch_train_ids.append(net_run_id)
        batch_eval_weights.append(out_dir / "weights" / "epoch0050.eqx")
    # Launch evaluation on this group of networks
    dry_run_mkdir(online_eval_dir)
    launch_online_eval(
        out_file=online_eval_dir / "test.hdf5",
        eval_file=test_heldout_file,
        weight_files=batch_eval_weights,
        dependency_ids=batch_train_ids,
    )
