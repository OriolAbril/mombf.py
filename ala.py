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
p = 4
n = 1000
key = random.PRNGKey(0)
X = random.normal(key, (n, p))
key, subkey = random.split(key)
y = ((1.7 * X[:, 1] - 1.6 * X[:, 2] + random.normal(subkey, (n,))) > 0).astype(jnp.int32)
fmt = f"{{:0{p}b}}"
gammes = np.array([list(fmt.format(i)) for i in range(2 ** p)])
gammes = jnp.array(gammes == "0")[:-1,:]

_, modprobs = modelSelection(X, y, gammes, family="logistic", prior="mom")
order = jnp.argsort(modprobs)[::-1]
(gammes[order, :], modprobs[order])

func = lambda y, x, z: z*jnp.dot(y, x)
func(y, X, 3)
part = Partial(func, x=X, z=3)
part(y=y, x=X/3)

k = {"a": 1}

k.update(dict(b=3, c=4))

k

