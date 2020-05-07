"""Prior functions available"""
from jax import numpy as jnp
from jax.numpy.linalg import det

def normalprior(beta, phi, g, W):
    """Calculate normal prior.
    To calculate group zellner prior, use ``utils.get_group_zellner``,
    which already adds the n and p_j terms.
    """
    detW = det(W)
    bXtXb = jnp.dot(jnp.dot(beta, W), beta)
    return -.5*(jnp.log(2*jnp.pi) + detW + bXtXb/(phi*g))

def gmomprior_penalty(beta, phi, g, Winv, p_j):
    """Calculate gmom penalty (also includes mom case).
    Note on p_j, it should be a vector of length p
    whose positions are equal to the number of covariates
    in the group. For example:
        groups=(1,1,2,3,4,4,4) -> p_j=(2,2,1,1,3,3,3)
    """
    bXtXinvb = jnp.dot(jnp.dot(beta, Winv), beta)
    return bXtXinvb/(phi*g*p_j)*(p_j+2)

def gmomprior(beta, phi, g, W, Winv, p_j):
    """Calculate gmom (includes mom case).
    See docs on ``gmomprior_penalty``
    """
    ans = normalprior(gamma, beta, phi, g, W)
    return ans + gmomprior_penalty(gamma, beta, phi, g, Winv, p_j)
