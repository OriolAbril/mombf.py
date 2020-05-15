# pylint: disable=too-many-locals, cell-var-from-loop, too-many-arguments
"""Model Selection utilities."""
from itertools import combinations

from jax import numpy as jnp
from jax.scipy.special import gammaln
from jax import vmap, jit, grad, hessian
from jax.tree_util import Partial

from .likelihoods import poisson_log_lik, logistic_log_lik
from .priors import normalprior, gmomprior, gmomprior_correction
from .ala import marghood_ala_post, marghood_ala_lik
from .utils import (
    get_group_zellner,
    get_p_j,
    apply_mask_2d,
    apply_mask_matrix,
    apply_mask_1d,
)


def modelSelection(
    X, y, models=None, prior="mom", family="logistic", groups=None, method="post"
):
    n, p = X.shape
    if models is None:
        n_models = 2 ** p
        model_i = 0
    else:
        n_models = models.shape[0]
    if groups is None:
        groups = jnp.arange(p)
    W, Winv = get_group_zellner(groups, X, prior == "mom")
    p_j = get_p_j(groups)
    modelprobs = jnp.empty(n_models)
    fact_y = jnp.sum(gammaln(y + 1))
    ytX = jnp.dot(y, X)
    if prior == "mom":
        XtX = jnp.dot(X.T, X)
    ## define vmapped functions ##
    # likelihood helpers
    if family == "poisson":
        _loglik = lambda b, ytx, x: poisson_log_lik(b, ytx, x, fact_y=fact_y)
    elif family == "logistic":
        _loglik = logistic_log_lik
    if method == "post":
        _logpost = lambda b, ytx, x, w: (_loglik(b, ytx, x) + normalprior(b, 1, 1, w))
        vmaparg = (0, 0, 0, 0)
        logpost = jit(vmap(_logpost, vmaparg, 0))
        glogpost = jit(vmap(grad(_logpost, argnums=0), vmaparg, 0))
        hlogpost = jit(vmap(hessian(_logpost, argnums=0), vmaparg, 0))
        jitted_ala = jit(
            Partial(
                marghood_ala_post, logpost=logpost, glogpost=glogpost, hlogpost=hlogpost
            )
        )
    elif method == "lik":
        logpr = jit(vmap(lambda b, w: normalprior(b, 1, 1, w), (0, 0), 0))
        loglik = jit(vmap(_loglik, (0, 0, 0), 0))
        gloglik = jit(vmap(grad(_loglik, argnums=0), (0, 0, 0), 0))
        hloglik = jit(vmap(hessian(_loglik, argnums=0), (0, 0, 0), 0))
        jitted_ala = jit(
            Partial(
                marghood_ala_lik,
                logl=loglik,
                glogl=gloglik,
                hlogl=hloglik,
                logpr=logpr,
            )
        )
    ## loop, we need equal shapes for vmap, hence calculations are vectorized ##
    ## on a number of selected variables basis ##
    for n_vars in range(1, p + 1):
        if models is None:
            models_iter = jnp.array(list(combinations(jnp.arange(p), n_vars)))
            model_mask = jnp.full((n_models,), False)
            model_mask = model_mask.at[model_i : models_iter.shape[0]].set(True)
            model_i += models_iter.shape[0]
        else:
            model_mask = models.sum(axis=1) == n_vars
            if model_mask.sum() == 0:
                continue
            models_iter = (
                jnp.arange(p)
                .reshape((1, -1))[models[model_mask, :]]
                .reshape((-1, n_vars))
            )
        b0 = jnp.zeros(models_iter.shape)
        X_iter = apply_mask_2d(X, models_iter)
        ytX_iter = apply_mask_1d(ytX, models_iter)
        W_iter = apply_mask_matrix(W, models_iter)
        if method == "post":
            args = [ytX_iter, X_iter, W_iter]
            margs = jitted_ala(b0=b0, args=args)
        elif method == "lik":
            argspr = (W_iter,)
            argsl = (ytX_iter, X_iter)
            margs = jitted_ala(b0=b0, argsl=argsl, argspr=argspr)
        if prior == "mom":
            p_j_iter = apply_mask_1d(p_j, models_iter)
            Winv_iter = apply_mask_matrix(Winv, models_iter)
            XtX_iter = apply_mask_matrix(XtX, models_iter)
            margs += vmap(gmomprior_correction, (None, 0, 0, 0, 0), 0)(
                1, Winv_iter, p_j_iter, XtX_iter, ytX_iter
            )
        modelprobs = modelprobs.at[model_mask].set(margs)
    return models, modelprobs
