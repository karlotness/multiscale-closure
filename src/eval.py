import pathlib
import argparse
import functools
import utils
import logging
import json
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from systems.qg.loader import SimpleQGLoader, qg_model_from_hdf5
from systems.qg import utils as qg_utils
import utils
import math
import dataclasses
from methods.gz_fcnn import GZFCNN
from train_sdegm import make_sampler, make_scalers
import jax_utils


parser = argparse.ArgumentParser(description="Evaluate neural networks for closure")
parser.add_argument("net_dir", type=str, help="Directory to store output (created if non-existing)")
parser.add_argument("eval_set", type=str, help="Directory with training examples")
parser.add_argument("saved_weight_type", type=str, help="Directory with validation examples")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])


def mse_loss(real_step, est_step):
    err = jnp.abs(est_step - real_step)
    return jnp.mean(err ** 2)


def relerr_loss(real_step, est_step):
    err = jnp.abs(est_step - real_step)
    return jnp.mean(err / jnp.abs(real_step))


def load_network(weight_dir, weight_type, small_model):
    net = GZFCNN(
        img_size=64,
        n_layers_in=4,
        n_layers_out=2,
        key=jax.random.PRNGKey(0),
    )
    # Ensure all weights are float32
    def leaf_map(leaf):
        if isinstance(leaf, jnp.ndarray):
            if leaf.dtype == jnp.dtype(jnp.float64):
                return leaf.astype(jnp.float32)
            if leaf.dtype == jnp.dtype(jnp.complex128):
                return leaf.astype(jnp.complex64)
        return leaf

    net = jax.tree_util.tree_map(leaf_map, net)
    net = eqx.tree_deserialise_leaves(weight_dir / f"{weight_type}.eqx", net)
    return net


def make_eval_traj_computer(small_model, num_steps, observe_interval=1, q_param_func=None):
    total_steps = num_steps * observe_interval

    def state_scan(carry, x):
        state, param_state = carry

        def wrap_q_param(system_state):
            nonlocal param_state
            dq, param_state = q_param_func(system_state, param_state)
            return dq

        new_state = small_model.step_forward(state, q_param_func=wrap_q_param if q_param_func is not None else None)
        return (new_state, param_state), new_state

    def do_eval(initial_state, param_state=None):
        (final_state, final_param_state), observed_states = jax_utils.strided_scan(
            state_scan,
            init=(initial_state, param_state),
            xs=None,
            length=num_steps,
            stride=observe_interval,
        )
        return observed_states

    return do_eval


def make_net_eval_traj_computer(net, small_model, num_steps, observe_interval=1, dt=0.01):
    q_scaler, forcing_scaler = make_scalers()
    draw_samples = make_sampler(
        dt=dt,
        sample_shape=(2, 64, 64),
        q_scaler=q_scaler,
        forcing_scaler=forcing_scaler,
    )
    DummyTrainState = dataclasses.make_dataclass("DummyTrainState", ["net"])

    def q_param(state, key):
        train_state = DummyTrainState(net=net)
        # Add batch and time dimensions
        batch_q = jnp.expand_dims(state.q, (0, 1))
        rng = key
        # Draw single sample for q subgrid forcing
        samples, new_rng = draw_samples(train_state, batch_q, rng)
        return samples.squeeze(0), new_rng

    return make_eval_traj_computer(
        small_model=small_model,
        num_steps=num_steps,
        observe_interval=observe_interval,
        q_param_func=q_param,
    )


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
    git_info = utils.get_git_info(base_logger=logger)
    if git_info is not None:
        logger.info(
            "Running on commit %s (%s worktree)",
            git_info.hash,
            "clean" if git_info.clean_worktree else "dirty"
        )
    logger.info("Saving results to: %s", out_dir)
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
                loss_funcs_dict={
                    "mse": mse_loss,
                    "relerr": relerr_loss,
                },
            )
        )
        logger.info("Starting to evaluate on %d trajectories of %d steps", eval_loader.num_trajs, eval_loader.num_steps)
        for traj_num in range(eval_loader.num_trajs):
            logger.debug("Starting to load trajectory %d", traj_num)
            traj = eval_loader.get_trajectory(traj_num)
            logger.debug("Finished loading trajectory %d", traj_num)
            logger.info("Starting rollout of trajectory %d", traj_num)
            traj_result = jax.device_get(eval_func(traj))
            logger.info("Finished rollout of trajectory %d", traj_num)
            # Report some results for the losses
            json_results["traj_info"][traj_num] = {}
            json_results["traj_info"][traj_num]["loss_keys"] = {}
            for loss_name, loss_result in traj_result.items():
                loss_key = f"{traj_num:05}_{loss_name}"
                json_results["traj_info"][traj_num]["loss_keys"][loss_name] = loss_key
                bulk_results[loss_key] = loss_result
            traj_horizons = {loss_name: {} for loss_name in traj_result.keys()}
            for horizon in sorted({5, 10, 25, 50, 100, 250, 500, 750, 1000}):
                horizon_report_components = []
                for loss_name in sorted(traj_result.keys()):
                    horiz_loss = float(np.mean(traj_result[loss_name][:horizon]))
                    traj_horizons[loss_name][horizon] = horiz_loss
                    horizon_report_components.append(f"{loss_name}: {horiz_loss:<15.5g}")
                # Build report string
                logger.info("Traj %05d horizon %5d: %s", traj_num, horizon, " ".join(horizon_report_components))
            json_results["traj_info"][traj_num]["loss_horizons"] = traj_horizons
    # Save results
    np.savez(out_dir / "results.npz", **bulk_results)
    with open(out_dir / "results.json", "w", encoding="utf8") as json_file:
        json.dump(json_results, json_file)
    logger.info("Finished evaluation")

if __name__ == "__main__":
    main()
