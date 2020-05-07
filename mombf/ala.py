"""ALA functions."""
from jax import numpy as jnp


def marghood_ala(b0, logpost, glogpost, hlogpost):
    glik = glogpost(b0)
    Hlik = -hlogpost(b0)
    f0 = logpost(b0)
    ans = (
        f0
        + 0.5 * jnp.dot(jnp.dot(glik, Hlik), glik)
        + 0.5 * (jnp.log(2 * jnp.pi) - jnp.log(jnp.linalg.det(Hlik)))
    )
    return ans
