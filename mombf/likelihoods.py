from jax import numpy as jnp
from jax.lax import cond
from jax.tree_util import Partial


def normal_log_lik(beta, phi, y, X):
    Xtbeta = jnp.dot(X, beta)
    p = beta.size
    lik = cond(
        jnp.allclose(beta, 0),
        p, lambda x: -.5*x*jnp.log(2*jnp.pi),
        phi, lambda ph: -.5*(p*jnp.log(2*jnp.pi)+jnp.dot((y - Xtbeta), y-Xtbeta)/ph)
    )
    return lik

def logistic_helper(beta, ytX, X):
    return (
        -jnp.sum(jnp.dot(ytX, beta)) +
        jnp.sum(jnp.log(1+jnp.exp(jnp.dot(X, beta))))
    )


def logistic_log_lik(beta, ytX, X, n):
    aux_fun = Partial(logistic_helper, ytX=ytX, X=X)
    lik = cond(
        jnp.allclose(beta, 0),
        n*jnp.log(2), lambda x: x,
        beta, aux_fun
    )
    return lik

def poisson_log_lik(beta, ytX, X, n, fact_y):
    aux_fun = lambda x: -jnp.dot(ytX, x) + jnp.sum(jnp.exp(jnp.dot(X, x))) + fact_y
    print(aux_fun(beta))
    lik = cond(
        jnp.allclose(beta, 0),
        beta, lambda x: n + fact_y,
        beta, aux_fun
    )
    return lik
