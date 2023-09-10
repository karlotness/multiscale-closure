import os
import pathlib
import time
import subprocess
import itertools
import re

DRY_RUN = True
SCRATCH = pathlib.Path(os.environ["SCRATCH"])
EVAL_FILE = SCRATCH / "closure" / "data-eddyonly" / "test" / "op1" / "data.hdf5"
EPOCH_WEIGHTS = ["best_loss", "interval"]
BASE_DIRS = [
    SCRATCH / "closure" / "run_outputs" / "run-all-nowarmup-20230531-100641",
    SCRATCH / "closure" / "run_outputs" / "run-all-noresidual-nowarmup-20230605-181442",
]

def sbatch_launch(args):
    args = [str(a) for a in args]
    print("sbatch", " ".join(f"'{a}'" for a in args))
    if not DRY_RUN:
        # Actually launch
        subprocess.run(["sbatch"] + args, check=True)
        time.sleep(0.5)


def dry_run_mkdir(dir_path):
    print("mkdir", f"'{pathlib.Path(dir_path).resolve()}'")
    if not DRY_RUN:
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def launch_net_evals(out_file, eval_file, net_weights):
    assert not pathlib.Path(out_file).exists()
    assert pathlib.Path(eval_file).is_file()
    assert all(pathlib.Path(p).is_file() for p in net_weights)
    dataset_name = eval_file.parent.parent.name
    sbatch_launch(["online-eval.sh", out_file, eval_file] + list(net_weights))


for base_dir in BASE_DIRS:
    dry_run_mkdir(base_dir / "online-eval-results")


for epoch_weight, base_dir in itertools.product(EPOCH_WEIGHTS, BASE_DIRS):
    net_groups = {}
    for weight_file in base_dir.glob(f"*/weights/{epoch_weight}.eqx"):
        weight_exp_name = weight_file.parent.parent.name
        if not re.match(r"(?:^sequential)|(?:.+-direct\d+-)", weight_exp_name):
            continue
        m = re.match(r"^(?P<prefix>.+-(?:small|medium))-(?P<num>\d+)$", weight_exp_name)
        group = m.group("prefix")
        if group not in net_groups:
            net_groups[group] = set()
        net_groups[group].add(weight_file)
    # Prep launches
    for group_name, group_nets in net_groups.items():
        out_file = base_dir / "online-eval-results" / f"{group_name}_{epoch_weight}.hdf5"
        launch_net_evals(out_file, EVAL_FILE, sorted(group_nets))
