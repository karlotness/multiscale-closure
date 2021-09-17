import jax


def euler_step(x, dt, time_deriv_func, postprocess_func=None):
    ret = x + dt * time_deriv_func(x)
    if postprocess_func is not None:
        ret = postprocess_func(ret)
    return ret


def rk4_step(x, dt, time_deriv_func, postprocess_func=None):
    k_1 = time_deriv_func(x)
    k_2 = time_deriv_func(x + 0.5 * dt * k_1)
    k_3 = time_deriv_func(x + 0.5 * dt * k_2)
    k_4 = time_deriv_func(x + dt * k_3)
    ret = x + (1 / 6) * dt * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
    if postprocess_func is not None:
        ret = postprocess_func(ret)
    return ret


def ssprk3_step(x, dt, time_deriv_func, postprocess_func=None):
    k_1 = (1 / 6) * time_deriv_func(x)
    k_2 = (1 / 6) * time_deriv_func(x + dt * k_1)
    k_3 = (2 / 3) * time_deriv_func(x + dt * (1 / 4) * (k_1 + k_2))
    ret = x + dt * (k_1 + k_2 + k_3)
    if postprocess_func is not None:
        ret = postprocess_func(ret)
    return ret


INTEGRATORS = {
    "euler": euler_step,
    "rk4": rk4_step,
    "ssprk3": ssprk3_step,
}


def make_integrator(integrator, time_deriv_func, postprocess_func=None):
    try:
        integrator_func = INTEGRATORS[integrator]
    except KeyError as e:
        raise ValueError(f"Unknown integrator: {integrator}") from e
    def integrate(x0, dt, num_steps):
        def step_function(step, _):
            new_step = integrator_func(step, dt, time_deriv_func, postprocess_func)
            return new_step, new_step
        _, further_steps = jax.lax.scan(step_function, init=x0, xs=None, length=num_steps)
        return further_steps
    return integrate
