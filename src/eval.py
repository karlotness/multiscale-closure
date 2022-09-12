import pathlib
import argparse
import functools
import utils
import logging
import json
import jax
import jax.numpy as jnp
import flax.serialization
import numpy as np
from flax.core import frozen_dict
from systems.qg.loader import SimpleQGLoader, qg_model_from_hdf5
from systems.qg import utils as qg_utils
import utils
from methods import ARCHITECTURES


parser = argparse.ArgumentParser(description="Evaluate neural networks for closure")
parser.add_argument("net_dir", type=str, help="Directory to store output (created if non-existing)")
parser.add_argument("eval_set", type=str, help="Directory with training examples")
parser.add_argument("saved_weight_type", type=str, help="Directory with validation examples")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])


def load_network(weight_dir, weight_type, small_model):
    weight_dir = pathlib.Path(weight_dir)
    dummy_q = jnp.zeros((small_model.nz, small_model.ny, small_model.nx))
    dummy_u = jnp.zeros((small_model.nz, small_model.ny, small_model.nx))
    dummy_v = jnp.zeros((small_model.nz, small_model.ny, small_model.nx))
    # Load weights
    with open(weight_dir / f"{weight_type}.json", "r", encoding="utf") as args_file:
        args = json.load(args_file)
    net = ARCHITECTURES[args["architecture"]](**args["params"])
    # Initialize the network with dummy values
    rng = jax.random.PRNGKey(seed=0)
    match net.param_type:
        case "uv":
            params = net.init(rng, dummy_u, dummy_v, False)
        case "q":
            params = net.init(rng, dummy_q, dummy_u, dummy_v, False)
        case _:
            raise ValueError(f"invalid parameterization type {net.param_type}")
    # Now, load from saved values
    if "batch_stats" not in params:
        params = params.copy("batch_stats"=frozen_dict.freeze({}))
    with open(weight_dir / f"{weight_type}.flaxnn", "rb") as weights_file:
        flax.serialization.from_bytes(params, weights_file.read())
    batch_stats = params["batch_stats"]
    params = params["params"]
    return net, params, batch_stats


def make_eval_traj_computer(net, params, batch_stats, small_model, loss_funcs_dict):
    apply_fn = functools.partial(net.apply, method=net.parameterization)
    memory_init_fn = functools.partial(net.apply, method=net.init_memory)

    def do_eval(traj):
        first_step = qg_utils.slice_kernel_state(traj, 0)
        tail_steps = qg_utils.slice_kernel_state(traj, slice(1, None))
        num_steps = traj.shape[0] - 1
        new_states, _last_memory = qg_utils.get_online_rollout(first_step, num_steps, apply_fn, params, small_model, memory_init_fn, batch_stats, net.param_type, False)
        # Compute losses
        return {
            loss_name: jax.vmap(loss_func)(tail_steps.q, new_states.q)
            for loss_name, loss_func in loss_func_dict.items()
        }

    return do_eval


def main():
    args = parser.parse_args()
    # Prepare directory paths
    net_dir = pathlib.Path(args.net_dir)
    weight_dir = net_dir / "weights"
    weight_data = weight_dir / f"{args.saved_weight_type}.flaxnn"
    weight_json = weight_dir / f"{args.saved_weight_type}.json"
    # Check that we can find both weight files
    if not weight_data.is_file() or not weight_json.is_file():
        raise ValueError(f"weight files for {args.saved_weight_type} do not both exist")
    # Prepare our particular output directory
    out_dir = net_dir / "eval" / args.saved_weight_type
    # Make our directory, but only the last two levels (we don't want to create net_dir if it doesn't exist)
    out_dir.parent.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    # Set up logging
    utils.set_up_logging(level=args.log_level, out_file=out_dir/"run.log")
    logger = logging.getLogger("main")
    logger.info("Arguments: %s", vars(args))
    # Find the training data file
    eval_file = pathlib.Path(args.eval_set) / "data.hdf5"
    # Create the model
    eval_small_model = qg_model_from_hdf5(file_path=eval_file, model="small")
    # Load the network
    net, params, batch_stats = load_network(weight_dir, args.saved_weight_type, eval_small_model)
    json_results = {"traj_info": {}}
    bulk_results = {}
    # Evaluate over each trajectory in turn
    with SimpleQGLoader(eval_file) as eval_loader:
        eval_func = jax.jit(
            make_eval_traj_computer(
                net=net,
                params=params,
                batch_stats=batch_stats,
                small_model=eval_small_model,
                loss_funcs_dics={
                    "mse": None,
                    "relerr": None,
                },
            )
        )
        for traj_num in range(eval_loader.num_trajs):
            logger.debug("Starting to load trajectory %d", traj_num)
            traj = loader.get_trajectory(traj_num)
            logger.debug("Finished loading trajectory %d", traj_num)
            logger.info("Starting rollout of trajectory %d", traj_num)
            traj_result = jax.device_get(eval_func(traj))
            logger.info("Finished rollout of trajectory %d", traj_num)
            # Report some results for the losses
            json_results["traj_info"][traj_num]["loss_keys"] = {}
            for loss_name, loss_result in traj_result.items():
                loss_key = f"{traj_num:05}_{loss_name}"
                json_results["traj_info"][traj_num]["loss_keys"] = {"loss_name": loss_key}
                bulk_results[loss_key] = loss_result
            horizon_report_components = []
            traj_horizons = {loss_name: {} for loss_name in traj_result.keys()}
            for horizon in sorted({5, 10, 25, 50, 100, 250, 500, 750, 1000}):
                for loss_name in sorted(traj_result.keys()):
                    horiz_loss = np.mean(traj_result[loss_name][:horizon])
                    traj_horizons[loss_name][horizon] = horiz_loss
                    horizon_report_components.append(f"{loss_name}: {horiz_loss:<15.5g}")
                # Build report string
                logger.info("Traj %d horizon %d: %s", traj_num, horizon, " ".join(horizon_report_components))
            json_results["traj_info"][traj_num]["loss_horizons"] = traj_horizons
            json_results["traj_horizons"][traj_num] = traj_horizons
    # Save results
    np.savez(out_dir / "results.npz", **bulk_results)
    with open(out_dir / "results.json", "w", encoding="utf8") as json_file:
        json.dump(json_results, json_file)
    logger.info("Finished evaluation")

if __name__ == "__main__":
    main()
