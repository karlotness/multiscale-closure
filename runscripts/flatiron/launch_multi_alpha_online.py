import subprocess
import codecs
import re
import time
import pathlib
import math
import os
import itertools
import operator
import sys

DRY_RUN = True
SCRATCH = pathlib.Path(os.environ["SCRATCH"]).resolve()

# General parameters
ALPHA_PARAMS = [1.0, 5.0, 10.0, 15.0]
SCALE_TESTSETS = {
    64: SCRATCH / "closure/data-rand-eddytojet/factor-1_0/test/op1/data.hdf5",
    96: SCRATCH / "closure/data-size96-rand-eddytojet/largestep/factor-1_0/test/op1/data.hdf5",
}

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


# Identify target directories
candidate_dirs = set()
for root_dir in sys.argv[1:]:
    root_dir = pathlib.Path(root_dir)
    candidate_dirs.update(p.parent.resolve() for p in root_dir.glob("**/poster-plot-data/test.hdf5"))

for candidate in sorted(candidate_dirs):
    # Determine scale
    if m := re.search(r"oneshot-noise-nets-scale(?P<scale>\d+)-", str(candidate)):
        scale = int(m.group("scale"))
        weight_files = sorted(candidate.parent.glob("net*/weights/epoch0050.eqx"))
        for alpha in ALPHA_PARAMS:
            alpha_underscore = str(alpha).replace(".", "_")
            out_file = candidate / f"test-alpha{alpha_underscore}.hdf5"
            if out_file.is_file():
                continue
            launch_online_eval(
                out_file=out_file,
                eval_file=SCALE_TESTSETS[scale],
                weight_files=weight_files,
                param_alpha=alpha,
            )
