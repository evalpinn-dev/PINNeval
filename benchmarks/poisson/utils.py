import jax
import jax.numpy as jnp


def get_dataset(dim=5, points_per_axis=32, num_test=65536, seed=1234):
    """
    Creates an analytic dataset for the d-dimensional Poisson problem on [0,1]^d
    with manufactured solution

        u(x) = sum_{i=1}^d sin(pi/2 * x_i).

    For low dimensions (dim <= 3), this returns a full tensor-product grid.
    For higher dimensions, it keeps the 1D axis linspaces but uses a sampled
    point cloud for evaluation.

    Parameters
    ----------
    dim : int
        Spatial dimension.
    points_per_axis : int
        Number of points per axis for the stored 1D axes.
    num_test : int
        Number of sampled reference/evaluation points for high-dimensional cases.
    seed : int
        RNG seed for sampled high-dimensional evaluation points.

    Returns
    -------
    sol : dict
        Solution dictionary containing:
            - 'u': exact solution on tensor grid, only for dim <= 3
            - 'U': flattened exact solution values, shape (N_test,)

    axes : dict
        Axes dictionary containing:
            - 'x0', ..., 'x{dim-1}': 1D axis linspaces
            - 'X': evaluation points, shape (N_test, dim)
            - 'grid_shape': only for low-dimensional tensor-grid case

    axis_order : tuple
        Order of the axes, e.g. ('x0', 'x1', ..., 'x{dim-1}').

    pde_extra_params : dict
        Dictionary containing PDE parameters:
            - 'dim': spatial dimension
    """

    def exact_u(X):
        return jnp.sum(jnp.sin(0.5 * jnp.pi * X), axis=-1)

    x_axes = [jnp.linspace(0.0, 1.0, points_per_axis) for _ in range(dim)]
    axis_order = tuple(f"x{i}" for i in range(dim))
    pde_extra_params = {"dim": dim}

    # Low-dimensional case: full tensor grid
    if dim <= 3:
        meshes = jnp.meshgrid(*x_axes, indexing="ij")
        u_exact = sum(jnp.sin(0.5 * jnp.pi * Xk) for Xk in meshes)

        X_flat = jnp.stack([Xk.reshape(-1) for Xk in meshes], axis=-1)
        U_flat = u_exact.reshape(-1)

        sol = {
            "u": u_exact,
            "U": U_flat,
        }

        axes = {
            **{f"x{i}": x_axes[i] for i in range(dim)},
            "X": X_flat,
            "grid_shape": tuple([points_per_axis] * dim),
        }

        return sol, axes, axis_order, pde_extra_params

    # High-dimensional case: sampled evaluation cloud
    key = jax.random.PRNGKey(seed)
    X_test = jax.random.uniform(key, (num_test, dim))
    U_test = exact_u(X_test)

    sol = {
        "U": U_test,
    }

    axes = {
        **{f"x{i}": x_axes[i] for i in range(dim)},
        "X": X_test,
        "grid_shape": tuple([points_per_axis] * dim),
    }

    return sol, axes, axis_order, pde_extra_params