from functools import partial

import jax.numpy as jnp
from jax import jit, grad, vmap

from pinneval.models import PINN
from pinneval.evaluator import BaseEvaluator


class Wave(PINN):
    def __init__(self, config, axes_dict, initial_dict, pde_params_dict):
        super().__init__(config, initial_dict, axes_dict, pde_params_dict=pde_params_dict)

        self.u0 = initial_dict["u0"]
        self.t_star = axes_dict["t"]
        self.x_star = axes_dict["x"]
        self.c = pde_params_dict["c"]

        self.t0 = self.t_star[0]
        self.t1 = self.t_star[-1]

        # Predictions over a grid
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))
        self.r_pred_fn = vmap(vmap(self.r_net, (None, None, 0)), (None, 0, None))

    def u_net(self, params, t, x):
        z = jnp.stack([t, x])
        _, u = self.state.apply_fn(params, z)
        return u[0]

    def u_t_net(self, params, t, x):
        u_t = grad(self.u_net, argnums=1)(params, t, x)
        return u_t

    def r_net(self, params, t, x):

        u_tt = grad(grad(self.u_net, argnums=1), argnums=1)(params, t, x)
        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)

        return u_tt - self.c**2 * u_xx

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, point_weights):
        # Initial condition loss
        u0_pred = vmap(self.u_net, (None, None, 0))(params, self.t0, self.x_star)
        u0_loss = jnp.mean((self.u0 - u0_pred) ** 2)

        u_t0_pred = vmap(self.u_t_net, (None, None, 0))(params, self.t0, self.x_star)
        u_t0_loss = jnp.mean((0 - u_t0_pred) ** 2)

        # Boundary condition loss
        u_bc1_pred = vmap(self.u_net, (None, 0, None))(params, self.t_star, self.x_star[0])
        u_bc2_pred = vmap(self.u_net, (None, 0, None))(params, self.t_star, self.x_star[-1])
        bcs_loss = jnp.mean((u_bc1_pred) ** 2) + jnp.mean((u_bc2_pred) ** 2)

        r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1])
        res_loss = jnp.mean((r_pred * point_weights) ** 2)

        loss_dict = {"u0": u0_loss, "u_t0": u_t0_loss, "res": res_loss, "bcs": bcs_loss}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.t_star, self.x_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error
 
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, point_weights):
        # === your exact math, but return a scalar directly ===
        u0_pred = vmap(self.u_net, (None, None, 0))(params, self.t0, self.x_star)
        u0_loss = jnp.mean((self.u0 - u0_pred) ** 2)

        u_t0_pred = vmap(self.u_t_net, (None, None, 0))(params, self.t0, self.x_star)
        u_t0_loss = jnp.mean((u_t0_pred) ** 2)

        # Boundary condition loss
        u_bc1_pred = vmap(self.u_net, (None, 0, None))(params, self.t_star, self.x_star[0])
        u_bc2_pred = vmap(self.u_net, (None, 0, None))(params, self.t_star, self.x_star[-1])
        bcs_loss = jnp.mean((u_bc1_pred) ** 2) + jnp.mean((u_bc2_pred) ** 2)

        # Residual loss
        r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1])
        res_loss = jnp.mean((r_pred * point_weights) ** 2)

        return weights["res"] * res_loss  + weights["bcs"] * bcs_loss  + weights["u0"] * u0_loss + weights["u_t0"] * u_t0_loss

    @partial(jit, static_argnums=(0,))
    def r_losses_pp(self, params, candidate_batch, weights):
        r_pred = vmap(self.r_net, (None, 0, 0))(params, candidate_batch[:, 0], candidate_batch[:, 1])
        return r_pred**2 * weights["res"]

    def loss_u0(self, params, batch, point_weights):
        u0_pred = vmap(self.u_net, (None, None, 0))(params, self.t0, self.x_star)
        return jnp.mean((self.u0 - u0_pred) ** 2)

    def loss_u_t0(self, params, batch, point_weights):
        u_t0_pred = vmap(self.u_t_net, (None, None, 0))(params, self.t0, self.x_star)
        return jnp.mean(u_t0_pred ** 2)

    def loss_bcs(self, params, batch, point_weights):
        u_bc1_pred = vmap(self.u_net, (None, 0, None))(params, self.t_star, self.x_star[0])
        u_bc2_pred = vmap(self.u_net, (None, 0, None))(params, self.t_star, self.x_star[-1])
        return jnp.mean(u_bc1_pred ** 2) + jnp.mean(u_bc2_pred ** 2)

    def loss_res(self, params, batch, point_weights):
        r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1])
        return jnp.mean((r_pred * point_weights) ** 2)

    @property
    def loss_fns(self):
        return {
            "u0": self.loss_u0,
            "u_t0": self.loss_u_t0,
            "bcs": self.loss_bcs,
            "res": self.loss_res,
        }

class WaveEvaluator(BaseEvaluator):
    def __init__(self, config, model, sol_dict):
        super().__init__(config, model)
        self.u_ref = sol_dict["u"]

    def log_errors(self, params):
        l2_error = self.model.compute_l2_error(params, self.u_ref)
        self.log_dict["l2_error"] = l2_error
