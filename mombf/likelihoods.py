from jax import numpy as jnp

def normal_log_lik(gamma, beta, y, X, phi):
    beta_g = jnp.where(gamma, beta, 0)
    Xtbeta_g = jnp.dot(X, beta_g)
    return -.5*jnp.dot((y - Xtbeta_g), y-Xtbeta_g)/phi

def poisson_log_lik(gamma, beta, y, X):
    pass

def logistic_log_lik(gamma, beta, y, X):
    pass
