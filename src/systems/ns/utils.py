import contextlib
import importlib
import gin
import jax_cfd
from .config import make_generation_config, make_grid


@contextlib.contextmanager
def temp_gin_config():
    cache = gin.config_str()
    gin.clear_config()
    try:
        yield
    finally:
        gin.clear_config()
        gin.parse_config(cache)


def make_train_encoder(size: int):
    _ = importlib.import_module("jax_cfd.ml")
    with temp_gin_config():
        gin.parse_config(make_generation_config())
        grid = make_grid(size)
        return gin.configurable(jax_cfd.ml.encoders.aligned_array_encoder)(
            grid=grid,
            dt=None,
            physics_specs=jax_cfd.ml.physics_specifications.get_physics_specs(),
        )
