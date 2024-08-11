import jax.numpy as jnp
from jax import jit
from jwave_utils import get_data_only

@jit
def linear_loss(speed, speed0, J, dy):
    ntimepoints, nelements = dy.shape
    loss = jnp.linalg.norm(dy - (J @ (speed - speed0).flatten()).reshape((ntimepoints, nelements)), ord='fro')
    return loss
# @jit
# def linear_loss(x, A, b):
#     Ax = jnp.tensordot(A, x, axes=([0, 1, 2], [0, 1, 2]))
#     return jnp.sum((Ax - b) ** 2)

@jit
def nonlinear_loss(speed, data0, density0, domain, time_axis, sources, element_positions):
    predicted = get_data_only(speed, density0, domain, time_axis, sources, element_positions)
    loss = jnp.linalg.norm(data0 - predicted, ord='fro')
    return loss