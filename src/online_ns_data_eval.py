import jax_cfd
import jax_cfd.base
import jax
import jax.numpy as jnp
import dataclasses
import gin
from systems.ns import loader as ns_loader
from train import determine_output_size, determine_channel_layers, make_basic_coarsener, make_chunk_from_batch, remove_residual_from_output_chunk
from cascaded_train import name_remove_residual, split_chunk_into_channels


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
