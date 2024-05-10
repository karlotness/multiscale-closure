import jax_cfd
import jax_cfd.ml
import jax_cfd.base
import math
import operator
import textwrap
import gin
import typing
import functools


def make_grid(size):
    return jax_cfd.base.grids.Grid(((operator.index(size),)*2), domain=([(0, 2*math.pi)]*2))


def make_generation_config(viscosity=0.001):
    return textwrap.dedent(f"""\
    # Macros:
    # ==============================================================================
    C_INTERPOLATION_MODULE = @interpolations.transformed
    CONVECTION_MODULE = @advections.self_advection
    DENSITY = 1.0
    DIFFUSION_MODULE = @diffusions.solve_fast_diag
    FORCING_MODULE = @forcings.kolmogorov_forcing
    NS_MODULE = @equations.modular_navier_stokes_model
    PRESSURE_MODULE = @pressures.fast_diagonalization
    U_INTERPOLATION_MODULE = @interpolations.linear

    # Parameters for get_model_cls:
    # ==============================================================================
    get_model_cls.model_cls = @ModularStepModel

    # Parameters for get_physics_specs:
    # ==============================================================================
    get_physics_specs.physics_specs_cls = @NavierStokesPhysicsSpecs

    # Parameters for implicit_diffusion_navier_stokes:
    # ==============================================================================
    implicit_diffusion_navier_stokes.diffusion_module = %DIFFUSION_MODULE

    # Parameters for kolmogorov_forcing:
    # ==============================================================================
    kolmogorov_forcing.linear_coefficient = -0.1
    kolmogorov_forcing.scale = 1.0
    kolmogorov_forcing.wavenumber = 4

    # Parameters for modular_advection:
    # ==============================================================================
    modular_advection.c_interpolation_module = %C_INTERPOLATION_MODULE
    modular_advection.u_interpolation_module = %U_INTERPOLATION_MODULE

    # Parameters for modular_navier_stokes_model:
    # ==============================================================================
    modular_navier_stokes_model.convection_module = %CONVECTION_MODULE
    modular_navier_stokes_model.equation_solver = \
        @equations.implicit_diffusion_navier_stokes
    modular_navier_stokes_model.pressure_module = %PRESSURE_MODULE

    # Parameters for ModularStepModel:
    # ==============================================================================
    ModularStepModel.advance_module = %NS_MODULE
    ModularStepModel.decoder_module = @decoders.aligned_array_decoder
    ModularStepModel.encoder_module = @encoders.aligned_array_encoder

    # Parameters for NavierStokesPhysicsSpecs:
    # ==============================================================================
    NavierStokesPhysicsSpecs.density = %DENSITY
    NavierStokesPhysicsSpecs.forcing_module = %FORCING_MODULE
    NavierStokesPhysicsSpecs.viscosity = {viscosity:f}

    # Parameters for self_advection:
    # ==============================================================================
    self_advection.advection_module = @advections.modular_advection

    # Parameters for transformed:
    # ==============================================================================
    transformed.base_interpolation_module = @interpolations.lax_wendroff
    transformed.transformation = @interpolations.tvd_limiter_transformation
    """)


def make_eval_model_config(corrector_type="ns_learned_corrector_v2", network_type="ns_equinox_corrector", model_cls="NSModularStepModel"):
    return textwrap.dedent(f"""\
    # Specification of the correction.
    {corrector_type}.base_solver_module = @modular_navier_stokes_model
    {corrector_type}.corrector_module = @{network_type}

    {model_cls}.encoder_module = @aligned_array_encoder
    {model_cls}.decoder_module = @aligned_array_decoder
    {model_cls}.advance_module = @{corrector_type}
    get_model_cls.model_cls = @{model_cls}
    """)


@gin.register
class NSModularStepModel(jax_cfd.ml.model_builder.ModularStepModel):
    """Dynamical model based on independent encoder/decoder/step components."""

    def __init__(
        self,
        grid: jax_cfd.base.grids.Grid,
        dt: float,
        physics_specs: jax_cfd.ml.physics_specifications.BasePhysicsSpecs,
        advance_module=gin.REQUIRED,
        encoder_module=gin.REQUIRED,
        decoder_module=gin.REQUIRED,
        name: typing.Optional[str] = None,
        ns_eqx_module = None,
    ):
        """Constructs an instance of a class."""
        super().__init__(
            grid=grid,
            dt=dt,
            physics_specs=physics_specs,
            advance_module=functools.partial(advance_module, ns_eqx_module=ns_eqx_module),
            encoder_module=encoder_module,
            decoder_module=decoder_module,
            name=name,
        )


@gin.register
def ns_learned_corrector_v2(
    grid: jax_cfd.base.grids.Grid,
    dt: float,
    physics_specs: jax_cfd.ml.physics_specifications.BasePhysicsSpecs,
    base_solver_module: typing.Callable,
    corrector_module: typing.Callable,
    ns_eqx_module=None,
):
    return jax_cfd.ml.equations.learned_corrector_v2(
        grid=grid,
        dt=dt,
        physics_specs=physics_specs,
        base_solver_module=base_solver_module,
        corrector_module=functools.partial(corrector_module, ns_eqx_module=ns_eqx_module),
    )
