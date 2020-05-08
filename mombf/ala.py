"""ALA functions."""
from jax import numpy as jnp
from jax import vmap

custom_dot = vmap(lambda g, H: jnp.dot(jnp.dot(g, H), g), (0, 0), 0)
custom_prod = vmap(lambda g, H: jnp.dot(H, g), (0, 0), 0)


def marghood_ala_post(b0, logpost, glogpost, hlogpost, *args):
    gpost = glogpost(b0, *args)
    Hpost = hlogpost(b0, *args)
    Hinv = jnp.linalg.inv(Hpost)
    f0 = logpost(b0, *args)
    ans = (
        f0
        + 0.5 * custom_dot(gpost, Hinv)
        + 0.5 * (jnp.log(2 * jnp.pi) - jnp.log(jnp.linalg.det(Hpost)))
    )
    return ans


def marghood_ala_lik(b0, logl, glogl, hlogl, logpr, argsl, argspr):
    glik = glogl(b0, *argsl)
    Hlik = hlogl(b0, *argsl)
    Hinv = jnp.linalg.inv(Hlik)
    btilde = b0 - custom_prod(Hinv, glik)
    f0 = logl(b0, *argsl) + logpr(btilde, *argspr)
    ans = (
        f0
        + 0.5 * (
            custom_dot(glik, Hinv)
            + jnp.log(2 * jnp.pi)
            - jnp.log(jnp.linalg.det(Hlik))
        )
    )
    return ans
