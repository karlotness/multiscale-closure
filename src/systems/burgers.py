import jax
import jax.numpy as jnp
import itertools


def attach_to_object(self):
    def decorator(func):
        setattr(self, func.__name__, func)
        return func
    return decorator


def _brute_force_compute_wave_num_corresp(wave_numbers):
    wave_numbers = wave_numbers.to_py().tolist()
    wn_idx = {wn: i for i, wn in enumerate(wave_numbers)}
    corresp = [[] for _ in wave_numbers]
    # Find correspondences
    for (ai, wa), (bi, wb) in itertools.combinations_with_replacement(enumerate(wave_numbers), 2):
        if wa + wb in wn_idx:
            idx = wn_idx[wa + wb]
            corresp[idx].append((ai, bi))
    max_len = max(len(sl) for sl in corresp)
    # Pad each sublist to max_len
    for sl in corresp:
        pad_len = max_len - len(sl)
        sl.extend([(-1, -1)] * pad_len)
    return jnp.moveaxis(jnp.array(corresp, dtype=jnp.int32), -1, 0)


def _truncate_frequencies(full_freqs, wave_numbers):
    return jnp.take(full_freqs, wave_numbers)


def _untruncate_frequencies(trunc_freqs, wave_numbers, space_size):
    new_freqs = jnp.zeros_like(trunc_freqs, shape=(space_size, ))
    new_freqs = new_freqs.at[wave_numbers].set(trunc_freqs)
    return new_freqs


class BurgersSystem:
    def __init__(self, nu=0.2, n_space_grid=1024, freq_span=512):
        assert 2 * freq_span <= n_space_grid
        # Configure constants
        self.nu = 0.2
        self._wave_numbers = jnp.array(range(-freq_span, freq_span), dtype=jnp.int32)
        self._wave_nums_corresp = _brute_force_compute_wave_num_corresp(self._wave_numbers)
        self._dealias_selector = jnp.abs(self._wave_numbers) > (1 * freq_span / 3)

        # Produce free functions
        @attach_to_object(self)
        def spatial_to_frequency(state):
            return _truncate_frequencies(jnp.fft.fft(state), self._wave_numbers)

        @attach_to_object(self)
        def frequency_to_spatial(x):
            return jnp.real(jnp.fft.ifft(_untruncate_frequencies(x, self._wave_numbers, n_space_grid)))

        @attach_to_object(self)
        def time_derivative(x):
            term_1 = -self.nu * (self._wave_numbers ** 2) * x
            indexing_freqs = jnp.pad(x, (0, 1))
            sub_term = (jnp.take(indexing_freqs, self._wave_nums_corresp[0], axis=0)
                        * jnp.take(indexing_freqs, self._wave_nums_corresp[1], axis=0))
            term_2 = (-1j * self._wave_numbers) * jnp.sum(sub_term, axis=1)
            return term_1 + term_2

        @attach_to_object(self)
        def step_postprocess(x):
            x = x.at[self._dealias_selector].set(0)
            return x
