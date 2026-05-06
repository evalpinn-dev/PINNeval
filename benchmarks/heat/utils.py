import jax
import jax.numpy as jnp


def get_dataset(dim=2, T=1.0, points_per_axis=32, num_test=65536, seed=1234):
    """
    Dataset for PINNacle HeatND:
        x in unit ball in R^dim, t in [0, T]
        u(x, t) = exp(||x||^2 / 2 + t)

    For low dimensions (dim <= 2), returns a full tensor grid in (x, t).
    For higher dimensions, keeps per-axis linspaces but uses a sampled point cloud
    inside the space-time domain.

    Returns
    -------
    sol : dict
        'u' : tensor-grid exact solution for low-dimensional case only
        'U' : flattened/sampled exact solution values, shape (N,)
    axes : dict
        'x0', ..., 'x{dim-1}' : 1D linspaces
        't' : 1D time axis
        'X' : evaluation points, shape (N, dim+1), with last column time
        'grid_shape' : only for tensor-grid case
    axis_order : tuple
        ('x0', ..., 'x{dim-1}', 't')
    pde_extra_params : dict
        {'dim': dim, 'T': T}
    """

    def exact_u(XT):
        x = XT[:, :-1]
        t = XT[:, -1:]
        x2 = jnp.sum(x**2, axis=1, keepdims=True)
        return jnp.exp(0.5 * x2 + t).reshape(-1)

    x_axes = [jnp.linspace(-1.0, 1.0, points_per_axis) for _ in range(dim)]
    t_axis = jnp.linspace(0.0, T, points_per_axis)

    axis_order = tuple([f"x{i}" for i in range(dim)] + ["t"])
    pde_extra_params = {"dim": dim, "T": T}

    # Low-dimensional full tensor grid, filtered to the unit ball
    if dim <= 2:
        meshes = jnp.meshgrid(*x_axes, t_axis, indexing="ij")
        coords = [M.reshape(-1) for M in meshes]
        XT_all = jnp.stack(coords, axis=-1)

        x = XT_all[:, :-1]
        keep = jnp.sum(x**2, axis=1) <= 1.0
        XT = XT_all[keep]
        U = exact_u(XT)

        # Optional dense tensor only for dim=1 if you need it; for dim=2 ball mask breaks full shape
        sol = {"U": U}
        axes = {
            **{f"x{i}": x_axes[i] for i in range(dim)},
            "t": t_axis,
            "X": XT,
        }
        return sol, axes, axis_order, pde_extra_params

    # High-dimensional sampled reference cloud inside ball x time interval
    key = jax.random.PRNGKey(seed)
    key_x, key_t = jax.random.split(key)

    # Sample uniformly in the unit ball by normalizing Gaussian directions and scaling radius
    z = jax.random.normal(key_x, (num_test, dim))
    z_norm = jnp.linalg.norm(z, axis=1, keepdims=True)
    dirs = z / z_norm
    # radius distribution for uniform ball
    u = jax.random.uniform(key_t, (num_test, 1))
    r = u ** (1.0 / dim)

    # Need a separate key for time
    key_time = jax.random.PRNGKey(seed + 1)
    t = T * jax.random.uniform(key_time, (num_test, 1))

    X = dirs * r
    XT = jnp.concatenate([X, t], axis=1)
    U = exact_u(XT)

    sol = {"U": U}
    axes = {
        **{f"x{i}": x_axes[i] for i in range(dim)},
        "t": t_axis,
        "X": XT,
    }

    return sol, axes, axis_order, pde_extra_params