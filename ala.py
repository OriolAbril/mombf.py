"""

exit()
venv_on_jax
python

"""

import numpy as np
import jax.numpy as jnp
from jax import random
from jax import vmap
from jax.tree_util import Partial
from mombf.modSel import modelSelection
from mombf.utils import apply_mask_1d

p = 10
n = 1000
key = random.PRNGKey(0)
X = random.normal(key, (n, p))
key, subkey = random.split(key)
y = ((1.7 * X[:, 1] - 1.6 * X[:, 2] + random.normal(subkey, (n,))) > 0).astype(jnp.int32)
fmt = f"{{:0{p}b}}"
gammes = np.array([list(fmt.format(i)) for i in range(2 ** p)])
gammes = jnp.array(gammes == "0")[:-1,:]

_, modprobs = modelSelection(X, y, gammes, family="logistic", prior="normal", method="post")
order = jnp.argsort(modprobs)[::-1]
(gammes[order, :], modprobs[order])


## Iris datset
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
y = iris.target
# mask = np.isin(y, (0, 1))
mask = np.full_like(y, True, dtype=bool)
y = y[mask] + 1
n = len(y)
df = pd.DataFrame(iris.data[mask, :2], columns=("sepal_length", "sepal_width"))
df["rand"] = np.random.normal(size=n)
df["rand2"] = np.random.exponential(size=n)
X = df[["sepal_length", "sepal_width", "rand"]].values
p = X.shape[1]
fmt = f"{{:0{p}b}}"
gammes = np.array([list(fmt.format(i)) for i in range(2 ** p)])
gammes = jnp.array(gammes == "0")[:-1,:]

_, modprobs = modelSelection(X, y, gammes, family="poisson", prior="normal", method="post")
order = jnp.argsort(modprobs)[::-1]
(gammes[order, :], modprobs[order])

from mombf.likelihoods import logistic_log_lik, poisson_log_lik
from jax import grad, hessian
from jax.scipy.special import gammaln

b0 = jnp.array([0., 0.])
X_g = X[:, :2]
ytX = jnp.dot(y, X_g)
fact_y = jnp.sum(gammaln(y + 1))

-poisson_log_lik(b0, ytX, X_g, fact_y)

g = grad(poisson_log_lik, argnums=0)(b0, ytX, X_g, fact_y)
H = hessian(poisson_log_lik, argnums=0)(b0, ytX, X_g, fact_y)
g, H
jnp.dot(g, H)

jnp.matmul(g.T, jnp.matmul(H, g))

jnp.allclose(jnp.array([0., 0., 0., 0.]), 0)

jnp.all(jnp.array([0., 0., 0., 0.]) == 0)

