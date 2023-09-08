import itertools
import re
import functools
import jax
import jax.numpy as jnp
import pyqg_jax
from train import make_chunk_from_batch, determine_channel_size, determine_output_size, make_basic_coarsener, remove_residual_from_output_chunk
from cascaded_train import split_chunk_into_channels, name_remove_residual
from cascaded_eval import load_networks
from systems.qg.loader import SnapshotStates
from systems.qg import coarsen
import jax_utils


def make_network_results_computer(nets, net_data, model_params):

    def compute_results(q, sys_params={}):
        q_was_double = (q.dtype == jnp.dtype(jnp.float64))
        q = jnp.expand_dims(q, 0)
        sys_params = jax.tree_map(lambda a: jnp.expand_dims(a, 0), sys_params)
        # Scale to standard
        q = jax.vmap(model_params.scalers.q_scalers[q.shape[-1]].scale_to_standard)(q)
        q = q.astype(jnp.float32)
        dummy_batch = SnapshotStates(q=None, q_total_forcings={}, sys_params=sys_params)
        # Pre-populate alt_sources with input q value
        alt_sources = {
            f"q_{q.shape[-1]}": q,
        }
        results = {}
        for net, data in zip(nets, net_data, strict=True):
            output_size = determine_output_size(data.output_channels)
            input_chunk = make_chunk_from_batch(
                channels=data.input_channels,
                batch=dummy_batch,
                model_params=model_params,
                processing_size=data.processing_size,
                alt_source=alt_sources,
            )
            predictions = jax.vmap(net)(input_chunk)
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
            results.update({name_remove_residual(k): v for k, v in predictions.items()})
            alt_sources.update(results)
        if q_was_double:
            # Cast back up to float64 if q was double precision
            results = {k: v.astype(jnp.float64) for k, v in results.items()}
        # Remove batch dimension from each result and return
        return {k: jnp.squeeze(v, 0) for k, v in results.items()}

    return compute_results


def make_forcing_computer(nets, net_data, model_params):

    def compute_forcing(q, sys_params={}):
        results = make_network_results_computer(nets, net_data, model_params)(q, sys_params=sys_params)
        # Final step, extract the forcing from alt_sources
        out_val = jnp.expand_dims(results[f"q_total_forcing_{q.shape[-1]}"], 0)
        out_val = jax.vmap(model_params.scalers.q_total_forcing_scalers[out_val.shape[-1]].scale_from_standard)(out_val)
        return jnp.squeeze(out_val, 0)

    return compute_forcing


def make_net_param_func(nets, net_data, model_params):

    @pyqg_jax.parameterizations.q_parameterization
    def net_param_func(model_state, param_aux, model, sys_params={}):
        compute_forcing_fn = make_forcing_computer(nets, net_data, model_params)
        return compute_forcing_fn(model_state.q, sys_params=sys_params), None

    return net_param_func


def make_parameterized_stepped_model(nets, net_data, model_params, qg_model_args, dt, state_map_fn=None):

    @functools.partial(jax.jit, static_argnums=(1, 2, 4))
    def model_stepper(initial_q, num_steps, subsampling=1, sys_params={}, skip_steps=0):
        nonlocal state_map_fn
        assert all(v.ndim == 0 for v in sys_params.values())
        assert num_steps > skip_steps
        new_model_params = qg_model_args.copy()
        new_model_params.update(sys_params)
        fixed_sys_params = {k: jnp.full((1, 1, 1), fill_value=v) for k, v in sys_params.items()}
        model = pyqg_jax.steppers.SteppedModel(
            pyqg_jax.parameterizations.ParameterizedModel(
                pyqg_jax.qg_model.QGModel(
                    **new_model_params,
                ),
                param_func=functools.partial(
                    make_net_param_func(
                        nets=nets,
                        net_data=net_data,
                        model_params=model_params,
                    ),
                    sys_params=fixed_sys_params,
                )
            ),
            pyqg_jax.steppers.AB3Stepper(dt=dt),
        )

        # Wrap in model states
        inner_state = model.model.model.create_initial_state(jax.random.PRNGKey(0))
        inner_state = inner_state.update(q=initial_q)
        state = model.initialize_stepper_state(
            model.model.initialize_param_state(inner_state)
        )
        # Step through time
        def step_state(carry, _x):
            old_state = carry
            new_state = model.step_model(old_state)
            if state_map_fn is None:
                out_value = old_state.state.model_state
            else:
                out_value = state_map_fn(old_state.state.model_state)
            return new_state, out_value

        def skip_states(carry, _x):
            new_state, _y = step_state(carry, _x)
            return new_state, None

        # Skip the warmup steps, if any
        if skip_steps > 0:
            state, _states = jax.lax.scan(
                skip_states,
                state,
                None,
                length=skip_steps,
            )

        _last_state, out_results = jax_utils.strided_scan(
            step_state,
            state,
            None,
            length=num_steps - skip_steps,
            stride=subsampling,
        )

        return out_results

    return model_stepper


def make_null_model_stepper(qg_model_args, dt):
    model = pyqg_jax.steppers.SteppedModel(
        pyqg_jax.qg_model.QGModel(
            **qg_model_args,
        ),
        pyqg_jax.steppers.AB3Stepper(dt=dt),
    )

    @functools.partial(jax.jit, static_argnums=(1, 2))
    def model_stepper(initial_q, num_steps, subsampling=1):
        # Wrap in model states
        inner_state = model.model.create_initial_state(jax.random.PRNGKey(0))
        inner_state = inner_state.update(q=initial_q)
        state = model.initialize_stepper_state(inner_state)
        # Step through time
        def step_state(carry, _x):
            old_state = carry
            new_state = model.step_model(old_state)
            return new_state, old_state.state

        _last_state, states = jax_utils.strided_scan(
            step_state,
            state,
            None,
            length=num_steps,
            stride=subsampling,
        )

        return states

    return model_stepper
