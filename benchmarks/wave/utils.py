import jax.numpy as jnp
from jax import vmap

def get_dataset(T=1.0, L=1.0, a=0.5, c=2.0, n_t=200, n_x=128):
    """
    Loads a 1D time-dependent analytic dataset.

    Parameters
    ----------
    T : float
        Final time.
    L : float
        Spatial domain length.
    a : float
        Amplitude parameter of the second mode.
    c : float
        Frequency scaling parameter.
    n_t : int
        Number of time points.
    n_x : int
        Number of spatial points.

    Returns
    -------
    sol : dict
        Solution dictionary containing:
            - 'u': solution values, shape (Nt, Nx)

    axes : dict
        Axes dictionary containing:
            - 't': time axis, shape (Nt,)
            - 'x': spatial axis, shape (Nx,)

    axis_order : tuple
        Order of the axes, given by ('t', 'x').

    pde_extra_params : dict
        Dictionary containing PDE parameters:
            - 'c': frequency scaling parameter.
            - 'a': amplitude parameter of the second mode.
    """

    t_star = jnp.linspace(0.0, T, n_t)
    x_star = jnp.linspace(0.0, L, n_x)

    def u_fn(t, x):
        return (
            jnp.sin(jnp.pi * x) * jnp.cos(c * jnp.pi * t)
            + a * jnp.sin(2 * c * jnp.pi * x) * jnp.cos(4 * c * jnp.pi * t)
        )

    # Evaluate on tensor-product grid (t, x)
    u_exact = vmap(
        vmap(u_fn, in_axes=(None, 0)),
        in_axes=(0, None),
    )(t_star, x_star)

    sol = {
        "u": u_exact,
    }

    axes = {
        "t": t_star,
        "x": x_star,
    }

    axis_order = ("t", "x")

    pde_extra_params = {'c': c, 'a': a}

    return sol, axes, axis_order, pde_extra_params
