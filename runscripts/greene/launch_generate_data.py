import subprocess
import codecs
import re
import time
import pathlib
import math
import os
import itertools

DRY_RUN = True
SCRATCH = pathlib.Path(os.environ["SCRATCH"]).resolve()

# GLOBAL PARAMETERS
OUT_DIR = SCRATCH / "closure" / "data-jet-nowarmup"
BIG_SIZE = 256
SMALL_SIZES = {128, 96, 64, 48}
PRECISION = "double"
DT = "3600.0"
TMAX = "311040000.0"
TWARMUP = "155520000.0"
COARSE_OP = "op1"
SAMPLE_CONFIG = "jet"

# TRAIN PARAMETERS
TRAIN_SLICE_SIZE = 25
TRAIN_NUM_TRAJS = 200
TRAIN_SUBSAMPLE = 8
TRAIN_SEED = 0

# VAL PARAMETERS
VAL_NUM_TRAJS = 3
VAL_SUBSAMPLE = 8
VAL_SEED = 1

# TEST PARAMETERS
TEST_NUM_TRAJS = 10
TEST_SUBSAMPLE = 8
TEST_SEED = 2

src_base = pathlib.Path(__file__).resolve().parent.parent.parent / "src"
container_img = str((SCRATCH / "closure" / "closure.sif").resolve())
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


def launch_generate_data(
    *,
    out_dir,
    seed,
    data_config,
    num_trajs,
    subsample,
    job_name="gen-data",
    small_sizes=SMALL_SIZES,
    dt=DT,
    tmax=TMAX,
    twarmup=TWARMUP,
    precision=PRECISION,
    coarse_op=COARSE_OP,
    big_size=BIG_SIZE,
    traj_slice=None,
    dependency_ids=None
):
    out_dir = str(pathlib.Path(out_dir).resolve())
    small_size_arg = " ".join(map(str, sorted(small_sizes)))
    args = [
        "python",
        "generate_data.py",
        str(out_dir),
        "qg",
        f"{seed:d}",
        "--config", data_config,
        "--precision", precision,
        f"--num_trajs={num_trajs:d}",
        "--dt", str(dt),
        "--tmax", str(tmax),
        "--twarmup", str(twarmup),
        "--coarse_op", coarse_op,
        "--subsample", str(subsample),
        "--big_size", str(big_size),
    ]
    args.append("--small_size")
    args.extend(map(str, sorted(small_sizes)))
    if traj_slice:
        args.extend(["--traj_slice", traj_slice])
    return container_cmd_launch(args, dependency_ids=dependency_ids, job_name=job_name, time_limit="12:00:00", cpus=1, mem_gb=20, gpus=1)

def launch_combine_slices(*, out_dir, train_jobs, coarse_op=COARSE_OP):
    out_dir = str((pathlib.Path(out_dir) / coarse_op).resolve())
    combine_job = container_cmd_launch(["python", "generate_data.py", str(out_dir), "combine_qg_slice"], job_name="qg-combine", cpus=2, mem_gb=20, time_limit="8:00:00", dependency_ids=train_jobs)
    sbatch_launch(["--wrap", f"rm {out_dir}/data-slice*.hdf5"], dependency_ids=[combine_job], job_name="cleanup-qg-combine", cpus=1, mem_gb=1, time_limit="0:30:00", gpus=0)
    return combine_job


def launch_shuffle_data(*, out_dir, data_gen_job, coarse_op=COARSE_OP):
    out_dir = str((pathlib.Path(out_dir) / coarse_op).resolve())
    return container_cmd_launch(["python", "shuffle_data.py", str(out_dir)], job_name="qg-shuf", cpus=4, gpus=0, mem_gb=8, time_limit="12:00:00", dependency_ids=[data_gen_job])


def dry_run_mkdir(dir_path):
    print("mkdir -p", f"'{pathlib.Path(dir_path).resolve()}'")
    if not DRY_RUN:
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def clip(a, a_min, a_max):
    return min(max(a, a_min), a_max)


# LAUNCH MAIN train set generation (including shuffling + subsetting)
num_train_slices = TRAIN_NUM_TRAJS // TRAIN_SLICE_SIZE + (1 if TRAIN_NUM_TRAJS % TRAIN_SLICE_SIZE else 0)
train_out_dir = OUT_DIR / "train"
train_job_ids = []
dry_run_mkdir(train_out_dir / COARSE_OP)
for slice_num in range(num_train_slices):
    slice_start = clip(slice_num * TRAIN_SLICE_SIZE, 0, TRAIN_NUM_TRAJS)
    slice_end = clip((1 + slice_num) * TRAIN_SLICE_SIZE, 0, TRAIN_NUM_TRAJS)
    slice_job_id = launch_generate_data(
        out_dir=train_out_dir,
        seed=TRAIN_SEED,
        data_config=SAMPLE_CONFIG,
        num_trajs=TRAIN_NUM_TRAJS,
        subsample=TRAIN_SUBSAMPLE,
        job_name="qg-gen-train",
        traj_slice=f"{slice_start}:{slice_end}",
    )
    train_job_ids.append(slice_job_id)
# Combine slices
combine_job_id = launch_combine_slices(
    out_dir=train_out_dir,
    train_jobs=train_job_ids,
)
# Shuffle train data
launch_shuffle_data(
    out_dir=train_out_dir,
    data_gen_job=combine_job_id,
)

# LAUNCH validation
val_out_dir = OUT_DIR / "val"
dry_run_mkdir(val_out_dir / COARSE_OP)
launch_generate_data(
    out_dir=val_out_dir,
    seed=VAL_SEED,
    data_config=SAMPLE_CONFIG,
    num_trajs=VAL_NUM_TRAJS,
    subsample=VAL_SUBSAMPLE,
    job_name="qg-gen-val",
)


# Launch test
test_out_dir = OUT_DIR / "test"
dry_run_mkdir(test_out_dir / COARSE_OP)
launch_generate_data(
    out_dir=test_out_dir,
    seed=TEST_SEED,
    data_config=SAMPLE_CONFIG,
    num_trajs=TEST_NUM_TRAJS,
    subsample=TEST_SUBSAMPLE,
    job_name="qg-gen-test",
)
