from jax import numpy as jnp
from jax.lax import cond


def normal_log_lik(beta, phi, y, X):
    Xtbeta = jnp.dot(X, beta)
    p = beta.size
    lik = cond(
        jnp.allclose(beta, 0),
        p, lambda x: -.5*x*jnp.log(2*jnp.pi),
        phi, lambda ph: -.5*(p*jnp.log(2*jnp.pi)+jnp.dot((y - Xtbeta), y-Xtbeta)/ph)
    )
    return lik


def logistic_log_lik(beta, ytX, n, X):
    aux_fun = lambda x: -jnp.sum(jnp.dot(ytX, x)) + jnp.sum(jnp.log(1+jnp.exp(jnp.dot(X, x))))
    lik = cond(
        jnp.allclose(beta, 0),
        n, lambda x: x * jnp.log(2),
        beta, aux_fun
    )
    return lik

def poisson_log_lik(beta, ytX, n, X, fact_y):
    aux_fun = lambda x: -jnp.dot(ytX, x) + jnp.sum(jnp.exp(jnp.dot(X, x))) + fact_y
    lik = cond(
        jnp.allclose(beta, 0),
        n, lambda x: x + fact_y,
        beta, aux_fun
    )
    return lik
