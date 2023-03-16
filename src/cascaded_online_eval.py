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

    def compute_results(q):
        q = jnp.expand_dims(q, 0)
        # Scale to standard
        q = jax.vmap(model_params.scalers.q_scalers[q.shape[-1]].scale_to_standard)(q)
        dummy_batch = SnapshotStates(q=None, q_total_forcings={})
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
        # Remove batch dimension from each result and return
        return {k: jnp.squeeze(v, 0) for k, v in results.items()}

    return compute_results


def make_forcing_computer(nets, net_data, model_params):

    def compute_forcing(q):
        results = make_network_results_computer(nets, net_data, model_params)(q)
        # Final step, extract the forcing from alt_sources
        out_val = jnp.expand_dims(results[f"q_total_forcing_{q.shape[-1]}"], 0)
        out_val = jax.vmap(model_params.scalers.q_total_forcing_scalers[out_val.shape[-1]].scale_from_standard)(out_val)
        return jnp.squeeze(out_val, 0)

    return compute_forcing


def make_net_param_func(nets, net_data, model_params):

    @pyqg_jax.parameterizations.q_parameterization
    def net_param_func(model_state, param_aux, model):
        compute_forcing_fn = make_forcing_computer(nets, net_data, model_params)
        return compute_forcing_fn(model_state.q), None

    return net_param_func


def make_parameterized_stepped_model(nets, net_data, model_params, qg_model_args, dt):
    model = pyqg_jax.steppers.SteppedModel(
        pyqg_jax.parameterizations.ParameterizedModel(
            pyqg_jax.qg_model.QGModel(
                **qg_model_args,
            ),
            param_func=make_net_param_func(
                nets=nets,
                net_data=net_data,
                model_params=model_params,
            ),
        ),
        pyqg_jax.steppers.AB3Stepper(dt=dt),
    )

    @functools.partial(jax.jit, static_argnums=(1, 2))
    def model_stepper(initial_q, num_steps, subsampling=1):
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
            return new_state, old_state.state.model_state

        _last_state, states = jax_utils.strided_scan(
            step_state,
            state,
            None,
            length=num_steps,
            stride=subsampling,
        )

        return states

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
