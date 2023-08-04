import os
import pathlib
import time
import subprocess
import itertools
import codecs
import re
import operator

DRY_RUN = True
NUM_REPEATS = 5
SCRATCH = pathlib.Path(os.environ["SCRATCH"])

TRAIN_FILE=SCRATCH / "closure/data-rand-eddytojet/factor-1_0/train100/op1/shuffled.hdf5"
LIVE_DATA_FILE=SCRATCH / "closure/data-smallstep-rand-eddytojet/factor-1_0/train100/op1/shuffled.hdf5"
VAL_FILE=SCRATCH / "closure/data-rand-eddytojet/factor-1_0/val/op1/data.hdf5"
TEST_TRAIN_FILE=SCRATCH / "closure/data-rand-eddytojet/factor-1_0/test-trainset/op1/data.hdf5"
TEST_HELDOUT_FILE=SCRATCH / "closure/data-rand-eddytojet/factor-1_0/test/op1/data.hdf5"

assert all(p.is_file() for p in [TRAIN_FILE, LIVE_DATA_FILE, VAL_FILE, TEST_TRAIN_FILE, TEST_HELDOUT_FILE])

LAUNCH_TIME = time.strftime("%Y%m%d-%H%M%S")
OUT_DIR=SCRATCH / "closure" / "run_outputs"/ f"test-unet-basictrain-{LAUNCH_TIME}"
dry_run_counter = 1000


def sbatch_launch(args, dependency_ids=None):
    global dry_run_counter
    if dependency_ids is None:
        dependency_ids = []
    deps = ":".join(f"{did}" for did in dependency_ids)
    if deps:
        dep_args = ["--dependency", f"afterok:{deps}", "--kill-on-invalid-dep", "yes"]
    else:
        dep_args = []
    args = ["--parsable"] + dep_args + list(args)
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


eval_job_ids = []
eval_weights = []
for repeat in range(NUM_REPEATS):
    run_dir = (OUT_DIR / f"net{repeat}").resolve()
    if not DRY_RUN:
        run_dir.mkdir(exist_ok=False, parents=True)
    job_id = sbatch_launch(["run-unet-basictrain.sh", str(run_dir), str(TRAIN_FILE.parent), str(VAL_FILE.parent)])
    eval_weights.append(run_dir / "weights" / "epoch0100.eqx")
    eval_job_ids.append(job_id)
# Launch evaluation jobs
for eval_file in [TEST_TRAIN_FILE, TEST_HELDOUT_FILE]:
    dataset_name = eval_file.parent.parent.name
    eval_out_file = OUT_DIR / "poster-plot-data" / f"{dataset_name}.hdf5"
    if not DRY_RUN:
        eval_out_file.parent.mkdir(exist_ok=True, parents=True)
    sbatch_launch(["poster-plot.sh", str(eval_out_file), str(eval_file)] + [str(w) for w in eval_weights], dependency_ids=eval_job_ids)
