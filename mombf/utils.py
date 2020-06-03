"""Utilities."""
from jax import numpy as jnp
from jax import vmap, jit
from jax.lax import cond


def get_p_j(groups):
    _, counts = jnp.unique(groups, return_counts=True)
    return jnp.array([p_j for p_j in counts for _ in range(p_j)])


def get_group_zellner(groups, X, isgmom=False):
    """Note that V=(XtX)^-1 and Vinv=XtX."""
    n, p = X.shape
    Vinv = jnp.zeros((p, p))
    V = jnp.zeros((p, p))
    for group, p_j in zip(*jnp.unique(groups, return_counts=True)):
        mask = jnp.arange(p)[groups == group]
        X_j = X[:, mask]
        p_term = cond(isgmom, p_j, lambda x: x, p_j, lambda x: x + 2)
        aux = jnp.dot(X_j.T, X_j) * n / p_term
        Vinv = Vinv.at[jnp.ix_(mask, mask)].set(aux)
        V = V.at[jnp.ix_(mask, mask)].set(jnp.linalg.inv(aux))
    return V, Vinv


apply_mask_2d = jit(vmap(lambda x, mask: x[:, mask], (None, 0), 0))
apply_mask_1d = jit(vmap(lambda x, mask: x[mask], (None, 0), 0))
apply_mask_matrix = jit(vmap(lambda w, mask: w[:, mask][mask, :], (None, 0), 0))
