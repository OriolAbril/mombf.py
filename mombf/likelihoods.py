from jax import numpy as jnp
from jax.lax import cond
from jax.tree_util import Partial


def normal_log_lik(beta, phi, y, X):
    Xtbeta = jnp.dot(X, beta)
    p = beta.size
    lik = cond(
        jnp.allclose(beta, 0),
        p,
        lambda x: -0.5 * x * jnp.log(2 * jnp.pi),
        phi,
        lambda ph: -0.5
        * (p * jnp.log(2 * jnp.pi) + jnp.dot((y - Xtbeta), y - Xtbeta) / ph),
    )
    return lik


def logistic_log_lik(beta, ytX, X):
    return jnp.sum(jnp.dot(ytX, beta)) - jnp.sum(jnp.log(1 + jnp.exp(jnp.dot(X, beta))))


def logistic_log_lik_cond(beta, ytX, X, n):
    aux_fun = Partial(logistic_log_lik, ytX=ytX, X=X)
    lik = cond(jnp.all(beta == 0), n * jnp.log(2), lambda x: x, beta, aux_fun)
    return -lik


def poisson_log_lik(beta, ytX, X, fact_y):
    return jnp.dot(ytX, beta) - jnp.sum(jnp.exp(jnp.dot(X, beta))) - fact_y


def poisson_log_lik_cond(beta, ytX, X, n, fact_y):
    aux_fun = Partial(poisson_log_lik, ytX=ytX, X=X, n=n, fact_y=fact_y)
    lik = cond(jnp.all(beta == 0), beta, lambda x: n + fact_y, beta, aux_fun)
    return -lik
