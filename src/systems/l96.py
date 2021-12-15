import jax
import jax.numpy as jnp
import itertools


def attach_to_object(self):
    def decorator(func):
        setattr(self, func.__name__, func)
        return func
    return decorator


def step_postprocess(x):
    return x

# Shape of x[(X 0, Y_fast 1,...j), variables k]
# Parameters:
# f   - constant forcing term
# h_x - coupling term
# h_y - coupling term
# eps - scale separation


class MultiscaleL96System:
    def __init__(self, f=10, h_x=-0.8, h_y=1, eps=2e-5):
        self.f = f
        self.h_x = h_x
        self.h_y = h_y
        self.eps = eps
        attach_to_object(self)(step_postprocess)

        @attach_to_object(self)
        def time_derivative(x):
            assert x.shape[-1] > 3
            state_x = x[0]
            state_y = x[1:].T
            term_1_x = jnp.roll(state_x, -1) - jnp.roll(state_x, 2)
            x_dot = term_1_x * jnp.roll(state_x, 1) - state_x + self.f + self.h_x * jnp.mean(state_y, axis=1)
            y_dot = (1/self.eps) * (-1 * jnp.roll(state_y, -1) * (jnp.roll(state_y, -2) - jnp.roll(state_y, 1)) - state_y + self.h_y * jnp.expand_dims(state_x, axis=1))
            return jnp.concatenate((jnp.expand_dims(x_dot, axis=0), y_dot.T))


class BasicL96System:
    def __init__(self, f=10):
        self.f = f
        attach_to_object(self)(step_postprocess)

        @attach_to_object(self)
        def time_derivative(x):
            assert x.shape[-1] > 3
            term_1 = jnp.roll(x, -1, axis=-1) - jnp.roll(x, 2, axis=-1)
            return term_1 * jnp.roll(x, 1, axis=-1) - x + self.f
