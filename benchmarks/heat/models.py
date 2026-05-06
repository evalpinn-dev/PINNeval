from functools import partial

import jax.numpy as jnp
from jax import jit, jvp, vmap, grad, random

from pinneval.models import PINN
from pinneval.evaluator import BaseEvaluator



def sample_points_in_unit_ball(num_points, dim, seed=1234):
    key = random.PRNGKey(seed)
    key_z, key_r = random.split(key)

    z = random.normal(key_z, (num_points, dim))
    z_norm = jnp.linalg.norm(z, axis=1, keepdims=True)
    dirs = z / z_norm

    u = random.uniform(key_r, (num_points, 1))
    r = u ** (1.0 / dim)
    return dirs * r


def sample_points_on_unit_sphere(num_points, dim, seed=4321):
    key = random.PRNGKey(seed)
    z = random.normal(key, (num_points, dim))
    z_norm = jnp.linalg.norm(z, axis=1, keepdims=True)
    return z / z_norm


class HeatND(PINN):
    def __init__(self, config, axes_dict, pde_params_dict=None, initial_dict=None):
        super().__init__(
            config=config,
            axes_dict=axes_dict,
            initial_dict=initial_dict,
            pde_params_dict=pde_params_dict,
        )

        self.dim = pde_params_dict["dim"]
        print(f"Initialized HeatND with dim={self.dim}")
        self.T = pde_params_dict.get("T", 1.0)

        self.X_star = axes_dict["X"]  # shape (N_test, dim+1)

        # Fixed boundary points on sphere across time
        num_bc = 2048
        Xb = sample_points_on_unit_sphere(num_bc, self.dim, seed=4321)
        tb = self.T * random.uniform(random.PRNGKey(4322), (num_bc, 1))
        self.X_bc = jnp.concatenate([Xb, tb], axis=1)

        # Fixed initial points in unit ball at t=0
        num_ic = 2048
        Xi = sample_points_in_unit_ball(num_ic, self.dim, seed=5432)
        ti = jnp.zeros((num_ic, 1))
        self.X_ic = jnp.concatenate([Xi, ti], axis=1)

        # Store true BC / IC targets once
        self.g_bc = self.boundary_flux_values(self.X_bc)
        self.u_ic = self.exact_u(self.X_ic)

        self.u_pred_fn = vmap(self.u_net, (None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0))
        self.nflux_pred_fn = vmap(self.normal_flux_net, (None, 0))

    def change_params(self, new_pde_params_dict):
        self.dim = new_pde_params_dict["dim"]
        self.T = new_pde_params_dict.get("T", self.T)

    def exact_u(self, XT):
        x = XT[:, :-1]
        t = XT[:, -1:]
        x2 = jnp.sum(x**2, axis=1, keepdims=True)
        return jnp.exp(0.5 * x2 + t).reshape(-1)

    def forcing(self, xt):
        x = xt[:-1]
        t = xt[-1]
        x2 = jnp.sum(x**2)
        return -(1.0 / self.dim) * x2 * jnp.exp(0.5 * x2 + t)

    def boundary_flux_values(self, XT):
        # On the unit sphere, ||x||=1, so ∂u/∂n = exp(1/2 + t)
        x = XT[:, :-1]
        t = XT[:, -1:]
        x2 = jnp.sum(x**2, axis=1, keepdims=True)
        return jnp.exp(0.5 * x2 + t).reshape(-1)

    def neural_net(self, params, xt):
        _, outputs = self.state.apply_fn(params, xt)
        return outputs[0]

    def u_net(self, params, xt):
        return self.neural_net(params, xt)

    def r_net(self, params, xt):
        d = self.dim
        f = lambda xt_: self.u_net(params, xt_)

        # Basis vectors for spatial axes, shape (d, d+1)
        eye_spatial = jnp.eye(d + 1, dtype=xt.dtype)[:d]

        def u_ii(v):
            _, val = jvp(lambda xt_: jvp(f, (xt_,), (v,))[1], (xt,), (v,))
            return val

        lap_u = jnp.sum(vmap(u_ii)(eye_spatial))

        e_t = jnp.zeros_like(xt).at[-1].set(1.0)
        u, u_t = jvp(f, (xt,), (e_t,))

        return (1.0 / d) * lap_u + self.forcing(xt) - u_t

    def normal_flux_net(self, params, xt):
        x = xt[:-1]
        grad_u = grad(self.u_net, argnums=1)(params, xt)
        grad_x = grad_u[: self.dim]

        x_norm = jnp.linalg.norm(x)
        n = x / x_norm
        return jnp.dot(grad_x, n)

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, point_weights):
        # Initial condition loss
        u_ic_pred = self.u_pred_fn(params, self.X_ic)
        u_ic_loss = jnp.mean((u_ic_pred - self.u_ic) ** 2)

        # Neumann boundary loss
        g_bc_pred = self.nflux_pred_fn(params, self.X_bc)
        g_bc_loss = jnp.mean((g_bc_pred - self.g_bc) ** 2)

        # PDE residual loss
        r_pred = vmap(self.r_net, (None, 0))(params, batch)
        res_loss = jnp.mean( (r_pred * point_weights)**2)  # use passed-in weights

        return {
            "u_ic": u_ic_loss,
            "g_bc": g_bc_loss,
            "r": res_loss,
        }

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, U_test):
        u_pred = self.u_pred_fn(params, self.X_star)
        return jnp.linalg.norm(u_pred - U_test) / jnp.linalg.norm(U_test)

    @partial(jit, static_argnums=(0,))
    def r_losses_pp(self, params, candidate_batch, weights):
        r_pred = vmap(self.r_net, in_axes=(None, 0))(params, candidate_batch)
        return r_pred**2 * weights["r"]

    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, point_weights):
        u_ic_pred = self.u_pred_fn(params, self.X_ic)
        u_ic_loss = jnp.mean((u_ic_pred - self.u_ic) ** 2)

        g_bc_pred = self.nflux_pred_fn(params, self.X_bc)
        g_bc_loss = jnp.mean((g_bc_pred - self.g_bc) ** 2)

        r_pred = self.r_pred_fn(params, batch)
        r_loss = jnp.mean((r_pred * point_weights) ** 2)

        return (
            weights["u_ic"] * u_ic_loss
            + weights["g_bc"] * g_bc_loss
            + weights["r"] * r_loss
        )

    @partial(jit, static_argnums=(0,))
    def loss_u_ic(self, params, batch, point_weights):
        u_ic_pred = self.u_pred_fn(params, self.X_ic)
        return jnp.mean((u_ic_pred - self.u_ic) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_g_bc(self, params, batch, point_weights):
        g_bc_pred = self.nflux_pred_fn(params, self.X_bc)
        return jnp.mean((g_bc_pred - self.g_bc) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_r(self, params, batch, point_weights):
        r_pred = self.r_pred_fn(params, batch)
        return jnp.mean((r_pred * point_weights) ** 2)

    @property
    def loss_fns(self):
        return {
            "u_ic": self.loss_u_ic,
            "g_bc": self.loss_g_bc,
            "r": self.loss_r,
        }


class HeatNDEvaluator(BaseEvaluator):
    def __init__(self, config, model, sol_dict):
        super().__init__(config, model)
        self.U_ref = sol_dict["U"]

    def log_errors(self, params):
        self.log_dict["l2_error"] = self.model.compute_l2_error(params, self.U_ref)
