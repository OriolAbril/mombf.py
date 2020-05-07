# pylint: disable=too-many-locals, cell-var-from-loop, too-many-arguments
"""Model Selection utilities."""
from jax import numpy as jnp
from jax.scipy.special import gammaln
from jax import vmap, jit, grad, hessian

from .likelihoods import poisson_log_lik, logistic_log_lik
from .priors import normalprior, gmomprior
from .ala import marghood_ala
from .utils import get_group_zellner, get_p_j


def modelSelection(X, y, models, prior="mom", family="logistic", groups=None):
    n_models, p = models.shape
    if groups is None:
        groups = jnp.arange(p)
    W = get_group_zellner(groups, X, prior == "mom")
    p_j = get_p_j(groups).reshape((1, p))
    modelprobs = jnp.empty(n_models)
    n = y.shape[-1]
    fact_y = jnp.sum(gammaln(y + 1))
    for n_vars in range(1, p + 1):
        model_mask = models.sum(axis=1) == n_vars
        if model_mask.sum() == 0:
            continue
        models_iter = models[model_mask, :]
        X_iter = X[models_iter]
        ytX = jnp.dot(y, X_iter)
        W_iter = vmap(lambda w, mask: w[:, mask][mask, :], (None, 0), 0)(W, models_iter)
        if prior == "mom":
            Winv_iter = jnp.linalg.inv(W_iter)
            p_j_iter = p_j[model_mask]
            prior_fun = lambda b: gmomprior(b, 1, 1, W_iter, Winv_iter, p_j_iter)
        elif prior == "normal":
            prior_fun = lambda b: normalprior(b, 1, 1, W_iter)
        if family == "poisson":
            _logpost = lambda b: prior_fun(b) + poisson_log_lik(
                b, ytX, n, X_iter, fact_y
            )
        elif family == "logistic":
            _logpost = lambda b: prior_fun(b) + logistic_log_lik(b, ytX, n, X_iter)
        logpost = vmap(_logpost, 0, 0)
        glogpost = vmap(grad(_logpost), 0, 0)
        hlogpost = vmap(hessian(_logpost), 0, 0)
        b0 = jnp.zeros(models_iter.shape)
        margs = jit(marghood_ala(b0, logpost, glogpost, hlogpost))
        modelprobs = modelprobs.at[model_mask].set(margs)
    return models, modelprobs
