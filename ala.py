"""

venv_on_jax
python

exit()

"""

import numpy as np
import jax.numpy as jnp
from jax import random

from mombf.modSel import modelSelection
from mombf.priors import normalprior

p = 4
n = 200
key = random.PRNGKey(0)
X = random.normal(key, (n, p))
key, subkey = random.split(key)
y = (0.7 * X[:, 1] - 0.6 * X[:, 2] + random.normal(subkey, n) > 0).astype(int)

fmt = f"{{:0{p}b}}"
gammes = np.array([list(fmt.format(i)) for i in range(2 ** p)])
gammes = jnp.array(gammes == "0")
