"""

venv_on_jax
python

exit()

"""

import numpy as np
from jax import numpy as jnp
from jax import lax, grad, jit, vmap
from jax import random
from jax.ops import index_update

from mombf.likelihoods import normal_log_lik
from mombf.utils import get_group_zellner

key = random.PRNGKey(0)
n = 200
p = 5
nrep = 10
X = random.normal(key, (nrep, n, p))
beta = jnp.zeros((nrep, p))
beta = beta.at[:, [0, 1, 2]].set([0.6, 0.7, -0.55])
y = vmap(jnp.dot, (0, 0), 0)(X, beta) + random.normal(key, (nrep, n))
phi = 1

groups = jnp.array([1] * 5 + [2, 3, 4] + [5] * 4 + [6] * 3)[:p]
Vinv_gzell = vmap(get_group_zellner, (None, 0), 0)(groups, X)
V_gzell = jnp.linalg.inv(Vinv_gzell)

fmt = f"{{:0{p}b}}"
gammes = np.array([list(fmt.format(i)) for i in range(2 ** p)])
gammes = jnp.array(gammes == "0")

vgam_norm_log_lik = vmap(normal_log_lik, (0, None, None, None, None), 0)
vrep_vgam_norm_log_lik = jit(vmap(vgam_norm_log_lik, (None, 0, 0, 0, None), 0))

liks = vrep_vgam_norm_log_lik(gammes, beta, y, X, phi)
liks

jit(vmap(grad(normalprior, argnums=1), (None, 0, None, None, 0), 0))(gammes[0], beta, phi, 1, V_gzell)
