import pathlib
import subprocess
import time
import os
import itertools

SCRATCH = pathlib.Path(os.environ["SCRATCH"])
DRY_RUN = True
EVAL_SET = SCRATCH / "closure/data-rand-eddytojet/factor-1_0/test"
assert EVAL_SET.is_dir()
WEIGHT_FILE = "epoch0100"


def sbatch_launch(args):
    print("sbatch", " ".join(f"'{a}'" for a in args))
    if not DRY_RUN:
        # Actually launch
        subprocess.run(["sbatch"] + args, check=True)
        time.sleep(0.5)


def dry_run_mkdir(dir_path):
    print("mkdir", f"'{pathlib.Path(dir_path).resolve()}'")
    if not DRY_RUN:
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


RUN_DIRS = {
    SCRATCH / "closure/run_outputs/continue-runs-20230712-014853",
    SCRATCH / "closure/run_outputs/continue-runs-noisy-20230713-153026",
    SCRATCH / "closure/run_outputs/continue-live-net-rollout-runs-20230725-042112",
}
for run_dir, noise_mode, candidates in itertools.product(
    {"continue-live-data-runs-20230718-021748", "continue-live-data-runs-20230718-124918", "continue-live-net-rollout-runs-20230725-042112", "continue-live-net-rollout-runs-10step17traj-20230725-153307", "continue-live-net-rollout-runs-2step85traj-20230725-153137"},
    ["add-noise", "noiseless"],
    [17, 20, 30, 34, 50, 85],
):
    possible_path = SCRATCH / "closure/run_outputs" / run_dir / noise_mode / f"candidates{candidates}"
    if possible_path.is_dir():
        RUN_DIRS.add(SCRATCH / "closure/run_outputs" / run_dir / noise_mode / f"candidates{candidates}")


# Locate network directories
net_experiment_dirs = set()
for path in RUN_DIRS:
    net_experiment_dirs.update(p.parent.parent.parent for p in path.glob(f"**/{WEIGHT_FILE}.eqx"))

for experiment_dir in map(pathlib.Path, net_experiment_dirs):
    out_file = experiment_dir / "online-ke-testset" / f"ke-{WEIGHT_FILE}.hdf5"
    if out_file.is_file():
        print(f"Skipping {experiment_dir}")
        continue
    net_weights = [str(p.resolve()) for p in experiment_dir.glob(f"*/weights/{WEIGHT_FILE}.eqx")]
    dry_run_mkdir(out_file.parent)
    sbatch_launch(["run-online-ke-data.sh", str(out_file.resolve()), EVAL_SET] + net_weights)
