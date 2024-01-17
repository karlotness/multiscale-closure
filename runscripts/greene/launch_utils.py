import codecs
import pathlib
import re
import subprocess
import time


DRY_RUN = True
dry_run_counter = 123000


def enable_real_launch():
    global DRY_RUN
    DRY_RUN = False


global_delays = {}
def global_delay(delay, key):
    global global_delays
    now = time.monotonic()
    prev = global_delays.get(key)
    global_delays[key] = now
    if prev is not None:
        time.sleep(max(0, delay - (now - prev)))


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
        global_delay(0.5, "sbatch")
        proc = subprocess.run(["sbatch"] + args, check=True, capture_output=True)
        output = codecs.decode(proc.stdout, encoding="utf8").strip()
        m = re.match(r"^\s*(?P<jobid>[^;]+)(?:;|$)", output)
        if m:
            job_id = m.group("jobid").strip()
        else:
            raise ValueError(f"could not parse {output}")
    else:
        dry_run_counter += 1
        job_id = str(dry_run_counter)
    print(f"# Launched with id: '{job_id}'")
    return job_id


def raw_cmd_launch(args, *, dependency_ids=None, time_limit=None, job_name=None, cpus=1, gpus=0, mem_gb=2):
    cmd_str = " ".join(str(a) for a in args)
    if "\"" in cmd_str:
        raise ValueError(f"command must not contain double quotes: {cmd_str}")
    return sbatch_launch(
        [f"--wrap=\"{cmd_str}\""],
        dependency_ids=dependency_ids,
        time_limit=time_limit,
        job_name=job_name,
        cpus=cpus,
        gpus=gpus,
        mem_gb=mem_gb,
    )


def copy_dir_launch(src, dst, *, dependency_ids=None, job_name=None):
    src = pathlib.Path(src).absolute()
    dst = pathlib.Path(dst).absolute()
    return raw_cmd_launch(
        [f"cp -a '{src}' '{dst}'"],
        dependency_ids=dependency_ids,
        time_limit="00:30:00",
        job_name=job_name,
        cpus=1,
        gpus=0,
        mem_gb=2,
    )


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
