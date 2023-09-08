import os
import pathlib
import time
import subprocess
import itertools

DRY_RUN = True
SCRATCH = pathlib.Path(os.environ["SCRATCH"])


EVAL_FILES = [
    SCRATCH / "closure/data-rand-eddytojet/factor-1_0/test/op1/data.hdf5",
    SCRATCH / "closure/data-rand-eddytojet/factor-1_0/test-trainset/op1/data.hdf5"
]
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

EXPERIMENT_DIRS = [
    SCRATCH / "closure" / "run_outputs" / exp_name for exp_name in [
        "continue-live-net-shortstep-rollout-runs-20230731-021307",
        "continue-live-net-shortstep-rollout-runs-20230728-012247",
        "continue-live-net-shortstep-rollout-runs-20230727-145733",
        "continue-live-data-runs-20230718-021748",
        "continue-live-data-runs-20230718-124918",
        "continue-live-net-rollout-runs-20230725-042112",
        "continue-live-net-rollout-runs-10step17traj-20230725-153307",
        "continue-live-net-rollout-runs-2step85traj-20230725-153137",
        "continue-runs-20230712-014853",
        "continue-runs-noisy-20230713-153026",
    ]
]

def sbatch_launch(args):
    print("sbatch", " ".join(f"'{a}'" for a in args))
    if not DRY_RUN:
        # Actually launch
        subprocess.run(["sbatch"] + args, check=True)
        time.sleep(0.5)


def launch_net_evals(out_file_name, eval_file, net_weights):
    dataset_name = eval_file.parent.parent.name
    out_file = SCRATCH / "closure" / "run_outputs" / "poster-plot-data" / f"{out_file_name}-{dataset_name}.hdf5"
    sbatch_launch(["poster-plot.sh", out_file, eval_file] + list(net_weights))


# Launch original nets
for eval_file in EVAL_FILES:
    launch_net_evals("orig-nets", eval_file, ORIG_NETS)

# Launch for other experiments
run_counter = 0
for eval_file in EVAL_FILES:
    for exp_path in EXPERIMENT_DIRS:
        # Locate networks and group into launches
        launch_groups = {}
        for weight_path in exp_path.glob("**/epoch0100.eqx"):
            if weight_path.parent.parent.parent.name.endswith("-continue"):
                continue
            group_key = str(weight_path.parent.parent.parent.parent.resolve())
            if group_key not in launch_groups:
                launch_groups[group_key] = set()
            launch_groups[group_key].add(weight_path)
        for launch_group, nets in launch_groups.items():
            launch_key = f"launch{run_counter:d}-{exp_path.name}"
            net_paths = sorted(nets)
            launch_net_evals(launch_key, eval_file, net_paths)
            run_counter += 1
