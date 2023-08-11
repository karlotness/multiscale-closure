import jax
import jax.numpy as jnp
import numpy as np
import utils
import math
import itertools
import pathlib
import logging
import sys
import argparse
import cascaded_eval
from cascaded_online_eval import make_parameterized_stepped_model
from eval import load_network
from train import load_model_params, make_basic_coarsener, determine_required_fields, determine_channel_size
from online_ensemble_compare import make_ke_time_computer, DATA_SUBSAMPLE_FACTOR, LoadedNetwork
from systems.qg import utils as qg_utils, loader
import equinox as eqx
import h5py

parser = argparse.ArgumentParser(description="Compare offline and online metrics for individual networks vs ensemble")
parser.add_argument("out_file", type=str, help="Directory to store evaluation results")
parser.add_argument("eval_set", type=str, help="Directory with evaluation samples")
parser.add_argument("nets", type=str, nargs="+", help="Trained network weight files")
parser.add_argument("--dt", type=float, default=3600.0, help="Time step size")
parser.add_argument("--rollout_subsample", type=int, default=DATA_SUBSAMPLE_FACTOR, help="Stride to use when rolling out simulations")
parser.add_argument("--t_metric_start", type=float, default=155520000.0, help="Time at which we should start averaging")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--traj_limit", type=int, help="Max trajectories to run", default=None)


@eqx.filter_jit
def model_rollout(loaded_net, initial_q, sys_params, dt, num_steps, subsampling, skip_steps, state_map_fn=None):
    ln = loaded_net
    small_size = int(ln.net_info["output_channels"][0][-2:])
    net_rollout_fn = make_parameterized_stepped_model(
        nets=[ln.net],
        net_data=[ln.net_data],
        model_params=ln.model_params,
        qg_model_args=qg_utils.qg_model_to_args(ln.model_params.qg_models[small_size]),
        dt=dt,
        state_map_fn=state_map_fn,
    )
    rolled_out_traj = net_rollout_fn(
        initial_q,
        num_steps=num_steps,
        subsampling=subsampling,
        sys_params=sys_params,
        skip_steps=skip_steps,
    )
    if state_map_fn is None:
        return rolled_out_traj.q
    else:
        return rolled_out_traj


def main():
    args = parser.parse_args()
    # Prepare our particular output directory
    out_path = pathlib.Path(args.out_file)
    # Make our directory, but only the last two levels (we don't want to create net_dir if it doesn't exist)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    # Set up logging
    utils.set_up_logging(level=args.log_level, out_file=out_path.parent/f"run-{out_path.stem}.log")
    logger = logging.getLogger("main")
    logger.info("Arguments: %s", vars(args))
    git_info = utils.get_git_info(base_logger=logger)
    if git_info is not None:
        logger.info(
            "Running on commit %s (%s worktree)",
            git_info.hash,
            "clean" if git_info.clean_worktree else "dirty"
        )
    if not utils.check_environment_variables(base_logger=logger):
        sys.exit(1)
    logger.info("Saving results to: %s", out_path)
    # Find the training data file
    eval_file = pathlib.Path(args.eval_set) / "data.hdf5"
    # Load the networks
    loaded_nets = []
    logger.info("Loading networks")
    for net_path in map(pathlib.Path, args.nets):
        logger.info("Loading network %s", net_path)
        net, net_info = load_network(net_path)
        loaded_nets.append(
            LoadedNetwork(
                net=net,
                net_info=net_info,
                net_data=cascaded_eval.NetData(
                    input_channels=net_info["input_channels"],
                    output_channels=net_info["output_channels"],
                    processing_size=net_info["processing_size"]
                ),
                net_path=net_path.resolve(),
                model_params=load_model_params(net_info["train_path"], eval_path=eval_file),
            )
        )
        del loaded_nets[-1].model_params.qg_models["big_model"]
    small_size = determine_channel_size(loaded_nets[0].net_info["output_channels"][0])
    dt_name_slug = str(int(args.dt))
    logger.info("Using time step dt=%f", args.dt)
    # Load data
    required_fields = determine_required_fields(loaded_nets[0].net_data.input_channels)
    required_fields.update(["q", f"q_total_forcing_{small_size}", "rek", "beta", "delta"])
    with loader.SimpleQGLoader(eval_file, fields=required_fields) as data_loader, h5py.File(out_path, "w", libver="latest") as out_file:
        logger.info("Using data set with %d trajectories of %d steps", data_loader.num_trajs, data_loader.num_steps)
        args_group = out_file.create_group("args")
        args_group.create_dataset("eval_file", data=str(eval_file))
        args_group.create_dataset("dt", data=args.dt)
        args_group.create_dataset("rollout_subsample", data=args.rollout_subsample)
        args_group.create_dataset("t_metric_start", data=args.t_metric_start)
        # ONLINE TESTS
        time_ke_computer = jax.jit(make_ke_time_computer(loaded_nets[0].model_params.qg_models[small_size]))
        logger.info("Running online tests")
        num_trajs_to_run = data_loader.num_trajs
        if args.traj_limit is not None:
            num_trajs_to_run = min(num_trajs_to_run, args.traj_limit)
            logger.info("Limiting run to %d trajectories", num_trajs_to_run)
        for traj in range(num_trajs_to_run):
            logger.info("Testing trajectory %d", traj)
            traj_data = data_loader.get_trajectory(traj)
            data_q = traj_data.q
            traj_sys_params = jax.tree_map(lambda d: d[0, 0, 0, 0], traj_data.sys_params)
            ref_subsample = math.ceil(args.rollout_subsample / (DATA_SUBSAMPLE_FACTOR * (args.dt / 3600.0)))
            num_steps = math.ceil(data_loader.num_steps * DATA_SUBSAMPLE_FACTOR / (args.dt / 3600.0))
            skip_steps = math.ceil(args.t_metric_start / args.dt)
            coarsener = make_basic_coarsener(data_q.shape[-1], small_size, loaded_nets[0].model_params)
            initial_q = coarsener(data_q[0])
            # Reference trajectory
            logger.info("Coarsening reference trajectory %d", traj)
            ref_traj = jax.vmap(coarsener)(data_q[(skip_steps // DATA_SUBSAMPLE_FACTOR)::ref_subsample])
            logger.info("Finished coarsening reference trajectory %d", traj)
            traj_group = out_file.create_group(f"traj{traj:05d}")
            path_group = traj_group.create_group("paths")
            net_times = np.linspace(args.t_metric_start, args.t_metric_start + (args.dt * data_loader.num_steps), ref_traj.shape[0], endpoint=False)
            traj_group.create_dataset("times", data=net_times)
            # Roll out trajectories with network corrections
            for label, rollout_q, net_path in itertools.chain(
                    [("reference", ref_traj, None)],
                    (
                        (
                            str(ln.net_path.relative_to(ln.net_path.parent.parent.parent)).replace("/", "-"),
                            model_rollout(
                                loaded_net=ln,
                                initial_q=initial_q,
                                sys_params=traj_sys_params,
                                dt=args.dt,
                                num_steps=num_steps,
                                subsampling=args.rollout_subsample,
                                skip_steps=skip_steps,
                            ),
                            ln.net_path
                        )
                        for ln in loaded_nets
                    )
            ):
                kes = time_ke_computer(rollout_q)
                traj_group.create_dataset(label, data=kes)
                if net_path is not None:
                    path_group.create_dataset(label, data=str(net_path))
                out_file.flush()
                logger.info("Computed KE traj %d and net %s", traj, label)

    logger.info("Finished evaluation")


if __name__ == "__main__":
    main()
