import os
import pathlib
import time
import subprocess
import itertools
import re

start_time = time.strftime("%Y%m%d-%H%M%S")

DRY_RUN = True
SCRATCH = pathlib.Path(os.environ["SCRATCH"])
ONLINE_EVAL_FILE = SCRATCH / "closure" / "data-eddyonly" / "test" / "op1" / "data.hdf5"
experiment_dir = SCRATCH / "closure" / "run_outputs" / f"stacked-baselines-{start_time}"
EPOCH_WEIGHTS = ["best_loss", "interval"]
ARCHITECTURES = ["stacked-gz-fcnn-v1-d2", "stacked-gz-fcnn-v1-d3"]
SCALES = [128, 96, 64]
NUM_REPEATS = 3
dry_run_counter = 0

def sbatch_launch(args, dependency_ids=None):
    global dry_run_counter
    if dependency_ids is None:
        dependency_ids = []
    deps = ":".join(f"{did}" for did in dependency_ids)
    if deps:
        dep_args = ["--dependency", f"afterok:{deps}", "--kill-on-invalid-dep", "yes"]
    else:
        dep_args = []
    args = ["--parsable"] + dep_args + [str(a) for a in args]
    print("sbatch", " ".join(f"'{a}'" for a in args))
    if not DRY_RUN:
        # Actually launch
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


def dry_run_mkdir(dir_path):
    print("mkdir -p", f"'{pathlib.Path(dir_path).resolve()}'")
    if not DRY_RUN:
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def launch_net_training(out_dir, arch, scale):
    input_channels = [f"q_{scale}"]
    output_channels = [f"q_total_forcing_{scale}"]
    job_id = sbatch_launch(["multi-train-cnn.sh", out_dir, arch, " ".join(input_channels), scale, " ".join(output_channels)])
    return job_id


def launch_single_offline_eval(net_dir, weight_type, train_job):
    job_id = sbatch_launch(["cnn-net-eval.sh", weight_type, net_dir], dependency_ids=[train_job])
    return job_id


def launch_net_evals(out_file, eval_file, net_weights, train_job_ids):
    assert not pathlib.Path(out_file).exists()
    assert pathlib.Path(eval_file).is_file()
    return sbatch_launch(["online-eval.sh", out_file, eval_file] + list(net_weights), dependency_ids=train_job_ids)


online_eval_dir = experiment_dir / "online-eval-results"
dry_run_mkdir(online_eval_dir)

for arch, scale in itertools.product(ARCHITECTURES, SCALES):
    type_name = f"{arch}-{scale}"
    train_job_ids = []
    out_dirs = []
    for rep in range(NUM_REPEATS):
        out_dir = experiment_dir / f"{type_name}-{rep}"
        out_dirs.append(out_dir)
        job_id = launch_net_training(out_dir=out_dir, arch=arch, scale=scale)
        train_job_ids.append(job_id)
        # Launch single network offline evaluations
        for weight_type in EPOCH_WEIGHTS:
            launch_single_offline_eval(net_dir=out_dir, weight_type=weight_type, train_job=job_id)
    # Launch online evaluations
    for weight_type in EPOCH_WEIGHTS:
        launch_net_evals(
            out_file=online_eval_dir / f"{type_name}_{weight_type}.hdf5",
            eval_file=ONLINE_EVAL_FILE,
            net_weights=[p / "weights" / f"{weight_type}.eqx" for p in out_dirs],
            train_job_ids=train_job_ids,
        )
