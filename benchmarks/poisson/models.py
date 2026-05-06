from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, hessian

from pinneval.models import PINN
from pinneval.evaluator import BaseEvaluator



def sample_points_on_all_hypercube_faces(num_points, dim, seed=1234):
    """
    Sample `num_points` points on each of the 2*dim faces of [0,1]^dim.

    Face ordering:
        for j in range(dim):
            face 2*j     -> x_j = 0
            face 2*j + 1 -> x_j = 1

    Returns
    -------
    X_bc : jnp.ndarray, shape (2*dim*num_points, dim)
    """
    key = jax.random.PRNGKey(seed)
    face_keys = jax.random.split(key, 2 * dim)

    X_faces = []

    for j in range(dim):
        X0 = jax.random.uniform(face_keys[2 * j], (num_points, dim))
        X0 = X0.at[:, j].set(0.0)
        X_faces.append(X0)

        X1 = jax.random.uniform(face_keys[2 * j + 1], (num_points, dim))
        X1 = X1.at[:, j].set(1.0)
        X_faces.append(X1)

    return jnp.concatenate(X_faces, axis=0)


class PoissonND(PINN):
    def __init__(self, config, axes_dict, pde_params_dict=None, initial_dict=None):
        super().__init__(
            config=config,
            axes_dict=axes_dict,
            initial_dict=initial_dict,
            pde_params_dict=pde_params_dict,
        )

        self.dim = pde_params_dict["dim"]
        self.X_star = axes_dict["X"]

        # Fixed boundary points, analogous to the LDC style
        num_points = 64
        self.X_bc = sample_points_on_all_hypercube_faces(
            num_points=num_points,
            dim=self.dim,
            seed=1234,
        )
        print(f"Sampled {self.X_bc.shape[0]} boundary points.")

        # Store true boundary values once
        self.u_bc = self.boundary_values(self.X_bc)

        # Vectorized prediction functions
        self.u_pred_fn = vmap(self.u_net, (None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0))

    def boundary_values(self, X):
        """
        Exact Dirichlet values on the boundary:
            u(x) = sum_i sin(pi/2 * x_i)

        X: shape (..., dim)
        returns: shape (...)
        """
        return jnp.sum(jnp.sin(0.5 * jnp.pi * X), axis=-1)

    def forcing(self, x):
        """
        For -Δu = f with
            u*(x) = sum_i sin(pi/2 * x_i),
        we have
            f(x) = (pi^2 / 4) * sum_i sin(pi/2 * x_i)
        """
        return (jnp.pi**2 / 4.0) * jnp.sum(jnp.sin(0.5 * jnp.pi * x))

    def neural_net(self, params, x):
        _, outputs = self.state.apply_fn(params, x)
        u = outputs[0]
        return u

    def u_net(self, params, x):
        return self.neural_net(params, x)

    def r_net(self, params, x):
        """
        Residual for:
            -Δu - f = 0
        """
        u_hessian = hessian(self.u_net, argnums=1)(params, x)
        lap_u = jnp.trace(u_hessian)
        f = self.forcing(x)
        r = -lap_u - f
        return r

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, point_weights):
        # Boundary condition loss
        u_bc_loss = self.loss_u_bc(params, batch, point_weights)
        res_loss = self.loss_res(params, batch, point_weights)
        loss_dict = {
            "u_bc": u_bc_loss,
            "r": res_loss,
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, U_test):
        u_pred = self.u_pred_fn(params, self.X_star)
        l2_error = jnp.linalg.norm(u_pred - U_test) / jnp.linalg.norm(U_test)
        return l2_error

    @partial(jit, static_argnums=(0,))
    def r_losses_pp(self, params, batch, weights):
        r_pred = vmap(self.r_net, in_axes=(None, 0))(params, batch)
        return r_pred**2 * weights["r"]

    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, point_weights):
        u_bc_loss = self.loss_u_bc(params, batch, point_weights)
        res_loss = self.loss_res(params, batch, point_weights)
        return weights["u_bc"] * u_bc_loss + weights["r"] * res_loss

    @partial(jit, static_argnums=(0,))
    def loss_u_bc(self, params, batch, point_weights):
        u_bc_pred = self.u_pred_fn(params, self.X_bc)
        return jnp.mean((u_bc_pred - self.u_bc) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_res(self, params, batch, point_weights):
        r_pred = self.r_pred_fn(params, batch)
        return jnp.mean((r_pred * point_weights) ** 2)

    @property
    def loss_fns(self):
        return {
            "u_bc": self.loss_u_bc,
            "r": self.loss_res,
        }


class PoissonNDEvaluator(BaseEvaluator):
    def __init__(self, config, model, sol_dict):
        super().__init__(config, model)
        self.U_ref = sol_dict["U"]

    def log_errors(self, params):
        l2_error = self.model.compute_l2_error(params, self.U_ref)
        self.log_dict["l2_error"] = l2_error
