"""Test modelSelection."""
import numpy as np
import jax.numpy as jnp
from jax import random
import pytest

from mombf.modSel import modelSelection


@pytest.mark.parametrize("family", ("logistic", "poisson"))
@pytest.mark.parametrize("prior", ("normal",))
@pytest.mark.parametrize("method", ("lik", "post"))
def test_modelSelection(family, prior, method):
    p = 5
    n = 700
    key = random.PRNGKey(0)
    X = random.normal(key, (n, p))
    key, subkey1, subkey2 = random.split(key, 3)
    mu = 1.7 * X[:, 1] - 1.6 * X[:, 2]
    truth = jnp.full((p,), False)
    truth = truth.at[[1, 2]].set(True)
    if family == "logistic":
        y = (mu + random.normal(subkey1, (n,)) > 0).astype(jnp.int32)
    elif family == "poisson":
        y = random.poisson(subkey2, lam=jnp.exp(mu), shape=(n,))
    fmt = f"{{:0{p}b}}"
    gammes = np.array([list(fmt.format(i)) for i in range(2 ** p)])
    gammes = jnp.array(gammes == "0")[:-1, :]

    _, modprobs = modelSelection(
        X, y, gammes, family=family, prior=prior, method=method
    )
    order = jnp.argsort(modprobs)[::-1]
    assert np.all(np.isfinite(modprobs))
    if family == "logistic":
        assert np.all(gammes[order[0], :] == truth), gammes[order[0], :]
