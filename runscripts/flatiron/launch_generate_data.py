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
OUT_DIR = SCRATCH / "closure" / "data-size96-rand-eddytojet"
SMALL_SIZES = {96}
EXPANSION_LEVELS = {"1.0"}
PRECISION = "double"
DT = "3600.0"
TMAX = "470160000.0"
TWARMUP = "155520000.0"
COARSE_OP = "op1"

# TRAIN PARAMETERS
TRAIN_SLICE_SIZE = 60
TRAIN_NUM_TRAJS = 275
TRAIN_SUBSAMPLE = 1000
TRAIN_SMALLSTEP_SUBSAMPLE = 10
TRAIN_SEED = 0
TRAIN_SUBSET_SIZES = [100]

# VAL PARAMETERS
VAL_NUM_TRAJS = 4
VAL_SUBSAMPLE = 1000
VAL_SEED = 1

# TEST PARAMETERS
TEST_NUM_TRAJS = 16
TEST_SUBSAMPLE = 8
TEST_SEED = 2

src_base = pathlib.Path(__file__).resolve().parent.parent.parent / "src"
container_img = str((SCRATCH / "closure" / "closure.sif").resolve())
dry_run_counter = 123000

def sbatch_launch(args, *, dependency_ids=None, time_limit=None, cwd=None, env_overrides=None):
    global dry_run_counter
    if dependency_ids is None:
        dependency_ids = []
    deps = ":".join(f"{did}" for did in dependency_ids)
    if deps:
        dep_args = ["--dependency", f"afterok:{deps}", "--kill-on-invalid-dep", "yes"]
    else:
        dep_args = []
    if time_limit is not None:
        time_args = [f"--time={time_limit}"]
    else:
        time_args = []
    args = ["--parsable"] + time_args + dep_args + [str(a) for a in args]
    print("sbatch", " ".join(f"'{a}'" for a in args))
    if not DRY_RUN:
        # Actually launch
        if env_overrides:
            proc_env = dict(os.environ).copy()
            proc_env.update(env_overrides)
        else:
            proc_env = None
        proc = subprocess.run(["sbatch"] + args, check=True, capture_output=True, cwd=cwd, env=proc_env)
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


def launch_generate_data(*, out_dir, seed, data_config, num_trajs, subsample, job_name="gen-data", small_sizes=SMALL_SIZES, dt=DT, tmax=TMAX, twarmup=TWARMUP, precision=PRECISION, coarse_op=COARSE_OP, traj_slice=None, dependency_ids=None):
    out_dir = str(pathlib.Path(out_dir).resolve())
    traj_slice_arg = f"--traj_slice '{traj_slice}'" if traj_slice else ""
    small_size_arg = " ".join(map(str, sorted(small_sizes)))
    launch_cmd = f"""singularity exec --nv "{container_img}" python generate_data.py "{out_dir}" qg {seed} --config "{data_config}" --precision {precision} --num_trajs={num_trajs} {traj_slice_arg} --coarse_op {coarse_op} --subsample {subsample} --small_size {small_size_arg} --dt {dt} --tmax {tmax} --twarmup {twarmup}"""
    args = ["--wrap", launch_cmd, "--job-name", str(job_name), "--cpus-per-task", "1", "--mem", "20G", "--gpus", "1", "--partition", "gpu"]
    return sbatch_launch(args, dependency_ids=dependency_ids, time_limit="12:00:00", cwd=src_base, env_overrides={"JAX_ENABLE_X64": "True", "JAX_DEFAULT_DTYPE_BITS": "32"})

def launch_combine_slices(*, out_dir, train_jobs, coarse_op=COARSE_OP):
    out_dir = str((pathlib.Path(out_dir) / coarse_op).resolve())
    launch_cmd = f"""singularity exec "{container_img}" python generate_data.py "{out_dir}" combine_qg_slice && rm {out_dir}/data-slice*.hdf5"""
    args = ["--wrap", launch_cmd, "--job-name", "qg-combine", "--cpus-per-task", "2", "--mem", "20G"]
    return sbatch_launch(args, dependency_ids=train_jobs, time_limit="8:00:00", cwd=src_base, env_overrides={"JAX_ENABLE_X64": "True", "JAX_DEFAULT_DTYPE_BITS": "32"})


def launch_shuffle_data(*, out_dir, data_gen_job, coarse_op=COARSE_OP):
    out_dir = str((pathlib.Path(out_dir) / coarse_op).resolve())
    launch_cmd = f"""singularity exec "{container_img}" python shuffle_data.py "{out_dir}" """
    args = ["--wrap", launch_cmd, "--job-name", "qg-shuf", "--cpus-per-task", "4", "--mem", "8G"]
    return sbatch_launch(args, dependency_ids=[data_gen_job], time_limit="12:00:00", cwd=src_base, env_overrides={"JAX_ENABLE_X64": "True", "JAX_DEFAULT_DTYPE_BITS": "32"})


def launch_subset_shuffle(*, source_data_dir, out_dir, data_gen_job, subset_size, coarse_op=COARSE_OP):
    in_file = str((pathlib.Path(source_data_dir) / coarse_op / "data.hdf5").resolve())
    out_file = ((pathlib.Path(out_dir) / coarse_op / "data.hdf5").resolve())
    out_shuf_dir = str((pathlib.Path(out_dir) / coarse_op).resolve())
    dry_run_mkdir(out_shuf_dir)
    launch_cmd = f"""singularity exec "{container_img}" python subset_shuffle_data.py "{in_file}" "{out_file}" "{subset_size}" && singularity exec "{container_img}" python shuffle_data.py "{out_shuf_dir}" && rm "{out_file}" """
    args = ["--wrap", launch_cmd, "--job-name", f"qg-subshuf-{subset_size}", "--cpus-per-task", "4", "--mem", "8G"]
    return sbatch_launch(args, dependency_ids=[data_gen_job], time_limit="12:00:00", cwd=src_base, env_overrides={"JAX_ENABLE_X64": "True", "JAX_DEFAULT_DTYPE_BITS": "32"})


def clip(a, a_min, a_max):
    return min(max(a, a_min), a_max)


# LAUNCH MAIN train set generation (including shuffling + subsetting)
for expansion_level, step_size in itertools.product(EXPANSION_LEVELS, ["large", "small"]):
    level_underscore = expansion_level.replace(".", "_")
    expansion_out_dir = OUT_DIR / f"{step_size}step" / f"factor-{level_underscore}"
    data_config = f"rand-eddy-to-jet-{expansion_level}"
    # Launch train runs
    num_train_slices = TRAIN_NUM_TRAJS // TRAIN_SLICE_SIZE + (1 if TRAIN_NUM_TRAJS % TRAIN_SLICE_SIZE else 0)
    train_job_ids = []
    train_out_dir = expansion_out_dir / "train"
    dry_run_mkdir(train_out_dir)
    for slice_num in range(num_train_slices):
        slice_start = clip(slice_num * TRAIN_SLICE_SIZE, 0, TRAIN_NUM_TRAJS)
        slice_end = clip((1 + slice_num) * TRAIN_SLICE_SIZE, 0, TRAIN_NUM_TRAJS)
        slice_job_id = launch_generate_data(
            out_dir=train_out_dir,
            seed=TRAIN_SEED,
            data_config=data_config,
            num_trajs=TRAIN_NUM_TRAJS,
            subsample=TRAIN_SUBSAMPLE if step_size != "small" else TRAIN_SMALLSTEP_SUBSAMPLE,
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
    # Launch subsetting jobs
    for subset_size in TRAIN_SUBSET_SIZES:
        subset_out_dir = expansion_out_dir / f"train{subset_size}"
        launch_subset_shuffle(
            source_data_dir=train_out_dir,
            out_dir=subset_out_dir,
            data_gen_job=combine_job_id,
            subset_size=subset_size,
        )

    if step_size != "small":
        # Launch validation
        val_out_dir = expansion_out_dir / "val"
        dry_run_mkdir(val_out_dir)
        launch_generate_data(
            out_dir=val_out_dir,
            seed=VAL_SEED,
            data_config=data_config,
            num_trajs=VAL_NUM_TRAJS,
            subsample=VAL_SUBSAMPLE,
            job_name="qg-gen-val",
        )

        # Launch test
        test_out_dir = expansion_out_dir / "test"
        dry_run_mkdir(test_out_dir)
        launch_generate_data(
            out_dir=test_out_dir,
            seed=TEST_SEED,
            data_config=data_config,
            num_trajs=TEST_NUM_TRAJS,
            subsample=TEST_SUBSAMPLE,
            job_name="qg-gen-test",
        )
