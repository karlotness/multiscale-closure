import sys
import argparse
import logging
import pathlib
import itertools
import math
import functools
import operator
import numpy as np
import utils
import jax_cfd
import jax_cfd.base
import jax
import jax.numpy as jnp
import equinox as eqx
import haiku as hk
import dataclasses
import powerpax
import h5py
import gin
from systems.ns import loader as ns_loader, config as ns_config, metrics as ns_metrics
from train import determine_output_size, determine_channel_layers, make_basic_coarsener, make_chunk_from_batch, remove_residual_from_output_chunk, determine_channel_size, determine_required_fields
from cascaded_train import name_remove_residual, split_chunk_into_channels
from online_data_eval import load_networks


parser = argparse.ArgumentParser(description="Compute statistics for NS nets")
parser.add_argument("out_file", type=str, help="File to store results")
parser.add_argument("eval_file", type=str, help="File to use for evaluation reference data")
parser.add_argument("net_weights", type=str, nargs="+", help="Network weights to evaluate")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--trajectory_batch_size", type=int, default=2, help="Number of trajectories to batch together")
parser.add_argument("--start_step_seed", type=int, default=54321, help="RNG seed for drawing start steps")
parser.add_argument("--rollout_length_limit", type=float, default=5.0, help="RNG seed for drawing start steps")
parser.add_argument("--num_rollouts", type=int, default=20, help="Number of short trajectories to roll out")


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class PackedModelData:
    nets: list
    net_data: list
    model_params: object

    def tree_flatten(self):
        return ((self.nets, self.net_data, self.model_params), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        nets, net_data, model_params = children
        return cls(nets=nets, net_data=net_data, model_params=model_params)


def make_network_results_computer(nets, net_data, model_params):

    def compute_results(state):
        assert len(state) == 2
        u, v = state
        size = u.shape[-1]
        was_double = (u.dtype == jnp.dtype(jnp.float64))
        dummy_batch = ns_loader.LoadedState(
            u=jnp.expand_dims(u.data, 0).astype(jnp.float32),
            v=jnp.expand_dims(v.data, 0).astype(jnp.float32),
            u_corr=None,
            v_corr=None,
        )
        alt_sources = {}
        for net, data in zip(nets, net_data, strict=True):
            output_size = determine_output_size(data.output_channels)
            input_chunk = make_chunk_from_batch(
                channels=data.input_channels,
                batch=dummy_batch,
                model_params=model_params,
                processing_size=data.processing_size,
                alt_source=alt_sources,
            )
            predictions=jax.vmap(net)(input_chunk)
            predictions = jax.vmap(make_basic_coarsener(data.processing_size, output_size, model_params))(predictions)
            # Process predictions and add to alt_sources
            predictions = split_chunk_into_channels(
                channels=data.output_channels,
                chunk=remove_residual_from_output_chunk(
                    output_channels=data.output_channels,
                    output_chunk=predictions,
                    batch=dummy_batch,
                    model_params=model_params,
                    processing_size=output_size,
                    alt_source=alt_sources,
                )
            )
            alt_sources.update({name_remove_residual(k): v for k, v in predictions.items()})
        # Remove batch dimension from each result and return
        uv_corr = jnp.squeeze(alt_sources[f"ns_uv_corr_{size:d}"], 0)
        assert uv_corr.shape[0] == 2
        u_unscaler = model_params.size_stats[size].field_stats("u_corr").scale_from_standard
        v_unscaler = model_params.size_stats[size].field_stats("v_corr").scale_from_standard
        u_corr = u_unscaler(uv_corr[0])
        v_corr = v_unscaler(uv_corr[1])
        if was_double:
            u_corr = u_corr.astype(jnp.float64)
            v_corr = v_corr.astype(jnp.float64)
        # Repack as grid variables
        return jax_cfd.base.initial_conditions.wrap_variables(
            (u_corr, v_corr),
            state[0].array.grid,
            [s.bc for s in state],
            [s.array.offset for s in state],
        )

    return compute_results


@gin.register
def ns_equinox_corrector(grid, dt, physics_specs, ns_eqx_module=None):
    results_comp = make_network_results_computer(ns_eqx_module.nets, ns_eqx_module.net_data, ns_eqx_module.model_params)

    def corrector(state):
        assert len(state) == 2
        return results_comp(state)

    return corrector


def make_traj_eval_fn(time_subsample_factor, model_cls, grid, bcs, offsets):

    def step_fn(carry, xs, model):
        assert xs is None
        next_step = model.advance(carry)
        return next_step, carry

    def traj_eval_fn(traj_data, ns_packed_net):
        assert traj_data.u.shape == traj_data.v.shape
        big_model = model_cls(ns_eqx_module=ns_packed_net)
        num_save_steps = traj_data.u.shape[0]
        ref_u = traj_data.u
        ref_v = traj_data.v
        start_vals = (ref_u[0], ref_v[0])
        v0 = tuple(
            jnp.expand_dims(v.data, 0) for v in
            jax_cfd.base.initial_conditions.wrap_variables(
                start_vals,
                grid,
                bcs,
                offsets,
            )
        )
        enc_v0 = big_model.encode(v0)
        _, traj = powerpax.sliced_scan(
            functools.partial(
                step_fn,
                model=big_model,
            ),
            enc_v0,
            None,
            length=num_save_steps * time_subsample_factor,
            start=0,
            step=time_subsample_factor,
        )
        traj = tuple(t.data for t in traj)
        u, v = traj
        assert u.shape == ref_u.shape
        assert v.shape == ref_v.shape

        # Compute some initial metrics
        total_ke = jax.vmap(lambda u, v: ns_metrics.ke(u, v).sum())(u, v)
        ref_ke = jax.vmap(lambda u, v: ns_metrics.ke(u, v).sum())(ref_u, ref_v)
        vort2d = jax.vmap(
            functools.partial(ns_metrics.vort2d, grid=grid)
        )(u, v)
        ref_vort2d = jax.vmap(
            functools.partial(ns_metrics.vort2d, grid=grid)
        )(ref_u, ref_v)
        vort_corr = jax.vmap(ns_metrics.correlate)(vort2d, ref_vort2d)
        metrics = {
            "total_ke": total_ke,
            "ref_ke": ref_ke,
            "vort_corr": vort_corr,
        }
        return traj, metrics

    def hk_transform(traj_data, ns_packed_net):
        return hk.without_apply_rng(
            hk.transform(
                functools.partial(
                    traj_eval_fn,
                    traj_data=traj_data,
                    ns_packed_net=ns_packed_net,
                )
            )
        ).apply({})

    def vec_fn(ns_packed_net, traj_data):
        return jax.vmap(
            functools.partial(
                hk_transform,
                ns_packed_net=ns_packed_net,
            )
        )(traj_data)

    return vec_fn


def main():
    args = parser.parse_args()
    # Set up logging
    out_file = pathlib.Path(args.out_file)
    eval_file = pathlib.Path(args.eval_file)
    log_file = out_file.parent / f"run-{out_file.stem}.log"
    utils.set_up_logging(level=args.log_level, out_file=log_file)
    logger = logging.getLogger("main")
    # Log basic information
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
    if not eval_file.is_file():
        logger.error("Eval file %s does not exist", eval_file)
        sys.exit(2)
    if out_file.is_file():
        logger.error("Output file %s already exists", out_file)
        sys.exit(3)

    # Load networks
    logger.info("Loading %d networks", len(args.net_weights))
    loaded_nets = load_networks(net_paths=args.net_weights, eval_file=eval_file, logger=logger.getChild("load-nets"))
    net_ids = []
    counter = 0
    for ln in loaded_nets:
        if ln.net_path == "null net":
            net_ids.append("null")
        else:
            net_ids.append(f"net{counter}")
            counter += 1
    # Determine evaluation size
    input_sizes = set(
        determine_channel_size(c) for c in ln.net_data[0].input_channels for ln in loaded_nets
    )
    if len(input_sizes) != 1:
        raise ValueError(f"Ambiguous/inconsistent input size {input_sizes}")
    main_size = input_sizes.pop()
    logger.info("Running with MAIN_SIZE=%d", main_size)
    # Determine required fields
    required_fields = determine_required_fields(
        itertools.chain.from_iterable(
            itertools.chain(ln.net_data[0].input_channels, ln.net_data[-1].output_channels) for ln in loaded_nets
        )
    )
    logger.info("Required fields: %s", required_fields)

    # Determine required eval parameters
    with h5py.File(eval_file, "r") as data_file:
        params_group = data_file[f"sz{main_size}"]["params"]
        ndim = params_group["ndim"][()].item()
        dt = params_group["dt"][()].item()
        stable_time_step = params_group["stable_time_step"][()].item()
        time_subsample_factor = params_group["time_subsample_factor"][()].item()
        maximum_velocity = params_group["maximum_velocity"][()].item()
        viscosity = params_group["viscosity"][()].item()
        full_config_str = params_group["full_config_str"].asstr()[()]
        domain_size_mul = data_file[f"sz{main_size}"]["params"]["domain_size_multiple"][()].item()
        u_offset = tuple(data_file[f"sz{main_size}"]["trajs"][f"traj00000_u"].attrs["offset"])
        v_offset = tuple(data_file[f"sz{main_size}"]["trajs"][f"traj00000_v"].attrs["offset"])
    logger.info("Eval set ndim=%d", ndim)
    logger.info("Eval set dt=%f", dt)
    logger.info("Time subsample factor=%d", time_subsample_factor)
    logger.info("Domain size mul=%d", domain_size_mul)

    total_rollout_steps = math.ceil(args.rollout_length_limit / dt)
    saved_rollout_steps = len(range(0, total_rollout_steps, time_subsample_factor))
    logger.info("Rolling out %d steps (sampling %d)", total_rollout_steps, saved_rollout_steps)

    gin.parse_config(full_config_str)
    gin.parse_config(ns_config.make_eval_model_config())

    # Construct jax-cfd model objects
    physics_specs = jax_cfd.ml.physics_specifications.get_physics_specs()
    grid = ns_config.make_grid(size=main_size, grid_domain_scale=domain_size_mul, ndim=ndim)
    big_model_cls = jax_cfd.ml.model_builder.get_model_cls(grid, dt, physics_specs)
    offsets = [u_offset, v_offset]
    bcs = [jax_cfd.base.boundaries.periodic_boundary_conditions(grid.ndim) for _ in range(grid.ndim)]

    traj_eval_fn = eqx.filter_jit(
        make_traj_eval_fn(
            time_subsample_factor=time_subsample_factor,
            model_cls=big_model_cls,
            grid=grid,
            bcs=bcs,
            offsets=offsets,
        )
    )

    # Open output file
    with h5py.File(out_file, "w") as out_file:
        logger.info("Storing basic parameters")
        params_group = out_file.create_group("params")
        params_group.create_dataset("eval_file", data=str(eval_file.resolve()))
        paths_group = params_group.create_group("paths")
        for net_id, ln in zip(net_ids, loaded_nets, strict=True):
            paths_group.create_dataset(net_id, data=str(ln.net_path))
        logger.info("Finished storing parameters")
        # Do network processing
        with ns_loader.SimpleNSLoader(eval_file, fields=required_fields) as data_set:
            # Select the steps to run
            data_num_steps = data_set.num_steps
            data_num_trajs = data_set.num_trajs
            valid_steps = data_num_steps - saved_rollout_steps
            total_valid_steps = valid_steps * data_num_trajs

            step_rng = np.random.Generator(
                np.random.PCG64(seed=args.start_step_seed)
            )
            start_steps = step_rng.integers(
                low=0,
                high=total_valid_steps,
                size=args.num_rollouts,
                dtype=np.uint64,
                endpoint=False,
            )
            trajs, steps = np.divmod(start_steps, valid_steps)
            # Record sources
            params_group.create_dataset("trajs", data=trajs)
            params_group.create_dataset("steps", data=steps)

            results_group = out_file.create_group("results")

            for traj_num in range(trajs.shape[0]):
                results_group.create_group(f"traj{traj_num:05d}")

            for net_id, ln in zip(net_ids, loaded_nets, strict=True):
                logger.info("Evaluating net %s", net_id)
                # Pack network for JIT functions
                ns_packed_net = PackedModelData(
                    nets=ln.net,
                    net_data=ln.net_data,
                    model_params=ln.model_params,
                )

                for traj_batch in itertools.batched(enumerate(zip(trajs, steps, strict=True)), args.trajectory_batch_size):
                    # Load data
                    logger.info("Loading next %d trajectories", len(traj_batch))
                    traj_data = []
                    for _, (traj, step) in traj_batch:
                        traj_data.append(
                            data_set.get_trajectory(
                                traj,
                                start=operator.index(step),
                                end=operator.index(step) + saved_rollout_steps,
                            )
                        )
                    traj_data = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *traj_data)

                    logger.info("Processing batch")
                    rollout, metrics = traj_eval_fn(ns_packed_net, traj_data)
                    logger.info("Storing metrics")

                    # Store results
                    for idx, (traj_num, (_, _)) in enumerate(traj_batch):
                        results, results_metrics = jax.tree_util.tree_map(lambda v: v[idx], (rollout, metrics))
                        traj_group = results_group[f"traj{traj_num:05d}"]
                        net_group = traj_group.create_group(net_id)
                        net_group.create_dataset("total_ke", data=np.asarray(results_metrics["total_ke"]))
                        net_group.create_dataset("ref_ke", data=np.asarray(results_metrics["ref_ke"]))
                        net_group.create_dataset("vort_corr", data=np.asarray(results_metrics["vort_corr"]))

                        # Compute energy spectrum
                        specs, k = ns_metrics.energy_spectrum(
                            u_traj=np.asarray(results[0]),
                            v_traj=np.asarray(results[1]),
                            grid=grid,
                            time_traj=((dt * time_subsample_factor) * np.arange(results[0].shape[0])),
                        )
                        net_group.create_dataset("energy_spectra", data=np.asarray(specs))
                        net_group.create_dataset("spec_k", data=np.asarray(k))

                        # Report decorr time
                        logger.info("Traj %d: decorr steps %d", traj_num, np.count_nonzero(np.asarray(results_metrics["vort_corr"]) >= 0.5))

                    # Drop results
                    rollout = None
                    metrics = None

    logger.info("Finished computing stats")

if __name__ == "__main__":
    main()
