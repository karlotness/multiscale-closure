import os
import pathlib
import time
import subprocess
import itertools
import codecs
import re
import operator

DRY_RUN = True
NUM_REPEATS = 1
NUM_SNAPSHOTS = operator.index(round(87 * 2))
SCRATCH = pathlib.Path(os.environ["SCRATCH"])

TRAIN_FILE=SCRATCH / "closure/data-rand-eddytojet/factor-1_0/train100/op1/shuffled.hdf5"
LIVE_DATA_FILE=SCRATCH / "closure/data-smallstep-rand-eddytojet/factor-1_0/train100/op1/shuffled.hdf5"
VAL_FILE=SCRATCH / "closure/data-rand-eddytojet/factor-1_0/val/op1/data.hdf5"
TEST_TRAIN_FILE=SCRATCH / "closure/data-rand-eddytojet/factor-1_0/test-trainset/op1/data.hdf5"
TEST_HELDOUT_FILE=SCRATCH / "closure/data-rand-eddytojet/factor-1_0/test/op1/data.hdf5"

assert all(p.is_file() for p in [TRAIN_FILE, LIVE_DATA_FILE, VAL_FILE, TEST_TRAIN_FILE, TEST_HELDOUT_FILE])

LAUNCH_TIME = time.strftime("%Y%m%d-%H%M%S")
OUT_DIR=SCRATCH / "closure" / "run_outputs"/ f"continue-noise-matched-nn-{LAUNCH_TIME}"
ORIG_NETS = [
    SCRATCH / "closure/run_outputs/run-varied-data-size-20230626-174855/rand-eddytojet/size100-scale64/net2/weights/epoch0025.eqx",
    SCRATCH / "closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net0/weights/best_loss.eqx",
    SCRATCH / "closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net2/weights/best_loss.eqx",
    SCRATCH / "closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net3/weights/best_loss.eqx",
    SCRATCH / "closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net3/weights/epoch0045.eqx",
    SCRATCH / "closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net4/weights/best_loss.eqx",
    SCRATCH / "closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net4/weights/epoch0035.eqx",
    SCRATCH / "closure/run_outputs/run-varied-data-size-20230705-174209/rand-eddytojet/size100-scale64/net4/weights/epoch0045.eqx",
]
STEP_LAYER_VARS = [
    (1, [0.00797639, 0.0038702]),
    (5, [0.01694924, 0.00856806]),
    (10, [0.03143954, 0.01700634]),
    (20, [0.07309306, 0.04431301]),
    (30, [0.12395001, 0.07433827]),
]
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


for config_i, (num_steps, var_values) in enumerate(STEP_LAYER_VARS):
    spec_str = "q_64=" + (",".join(map(str, var_values)))
    config_dir = OUT_DIR / f"steps{num_steps:04}"
    for name, spec_str, gen_mode in [("net-roll", "", "network-noise-onestep"), ("gaussian", spec_str, "schedule-only")]:
        mode_dir = config_dir / name
        eval_weights = []
        eval_job_ids = []
        for orig_net in ORIG_NETS:
            orig_net_name = "-".join(itertools.chain(orig_net.parts[-6:-2], [orig_net.stem], ["continue"]))
            start_weight_dir = mode_dir / orig_net_name
            for repeat in range(NUM_REPEATS):
                run_dir = (start_weight_dir / f"trial{repeat}").resolve()
                if not DRY_RUN:
                    run_dir.mkdir(exist_ok=False, parents=True)
                # Launch job
                job_id = sbatch_launch(["run-noise-match-net-roll-train.sh", str(run_dir), str(TRAIN_FILE.parent), str(VAL_FILE.parent), str(LIVE_DATA_FILE.parent), str(orig_net), gen_mode, spec_str, num_steps, NUM_SNAPSHOTS])
                eval_weights.append(run_dir / "weights" / "epoch0100.eqx")
                eval_job_ids.append(job_id)
        # Launch evaluation jobs
        for eval_file in [TEST_TRAIN_FILE, TEST_HELDOUT_FILE]:
            dataset_name = eval_file.parent.parent.name
            eval_out_file = mode_dir / "poster-plot-data" / f"{dataset_name}.hdf5"
            if not DRY_RUN:
                eval_out_file.parent.mkdir(exist_ok=True, parents=True)
            sbatch_launch(["poster-plot.sh", str(eval_out_file), str(eval_file)] + [str(w) for w in eval_weights], dependency_ids=eval_job_ids)
