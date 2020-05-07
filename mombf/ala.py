"""ALA functions."""
from jax import numpy as jnp
from jax import vmap

custom_dot = vmap(lambda g, H: jnp.dot(jnp.dot(g, H), g), (0, 0), 0)

def marghood_ala(b0, logpost, glogpost, hlogpost, *args):
    glik = glogpost(b0, *args)
    Hlik = -hlogpost(b0, *args)
    f0 = logpost(b0, *args)
    ans = (
        f0
        + 0.5 * custom_dot(glik, Hlik)
        + 0.5 * (jnp.log(2 * jnp.pi) - jnp.log(jnp.linalg.det(Hlik)))
    )
    return ans
