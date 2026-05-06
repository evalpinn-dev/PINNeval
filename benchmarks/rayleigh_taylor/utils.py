import jax.numpy as jnp

def get_dataset():
    """
    Loads the Rayleigh–Taylor dataset (high Ra case).

    The spatial coordinates are provided as a flattened (x, y) grid and are
    reconstructed into separable axes.

    Returns
    -------
    sol : dict
        Solution dictionary containing:
            - 'u': x-velocity component
            - 'v': y-velocity component
            - 'p': pressure
            - 'T': temperature
        Each array has shape (Nt, Nx, Ny).

    axes : dict
        Axes dictionary containing:
            - 't': time axis (reset to start at t = 0)
            - 'x': spatial x-axis
            - 'y': spatial y-axis

    axis_order : tuple
        Order of the axes, given by ('t', 'x', 'y').

    pde_extra_params : dict
        Dictionary containing PDE parameters:
            - 'alpha1', 'alpha2', 'alpha3', 'alpha4'
            - 'Ra', 'Pr', 'Ge'
    """

    # ------------------------------------------------------------
    # Load raw data (same as before)
    # ------------------------------------------------------------
    data = jnp.load(
        "data/rayleigh_taylor_high_Ra.npy",
        allow_pickle=True
    ).item()

    

    start_idx = 5

    velocity = jnp.asarray(data["velocity"])[start_idx:]     # (Nt, Npoints, 2)
    pressure = jnp.asarray(data["pressure"])[start_idx:]     # (Nt, Npoints)
    temperature = jnp.asarray(data["temperature"])[start_idx:]

    t = jnp.asarray(data["t"])[start_idx:]
    t = t - t[0]

    coords = jnp.asarray(data["coords"])                      # (Npoints, 2)

    u_ref = velocity[..., 0]
    v_ref = velocity[..., 1]
    p_ref = pressure
    T_ref = temperature

    # ------------------------------------------------------------
    # Assemble outputs
    # ------------------------------------------------------------
    sol = {
        "u": u_ref,
        "v": v_ref,
        "p": p_ref,
        "T": T_ref,
    }

    axes = {
        "t": t,
        "coords": coords,
    }

    axis_order = ("t", "coords")

    pde_param_dict = {
        "alpha1": data["alpha1"],
        "alpha2": data["alpha2"],
        "alpha3": data["alpha3"],
        "alpha4": data["alpha4"],
        "Ra": data["Ra"],
        "Pr": data["Pr"],
        "Ge": data["Ge"],
    }

    return sol, axes, axis_order, pde_param_dict


if __name__ == "__main__":

    import pinneval.train as train
    num_time_windows = 4

    sol_dict, axes_dict, axis_order, pde_params_dict = get_dataset()
    
    # add t to the sol_dict for conflicts of slicing in the last window
    sol_dict["t"] = axes_dict["t"]

    num_time_steps = len(axes_dict["t"]) // num_time_windows

    # shorten t-axis accordingly
    axes_dict["t"] = axes_dict["t"][:num_time_steps+1]

    
    window_error_table = None            
    window_error_history = {}           
    # collect all final predictions for final evaluation, that is every key in sol_dict except t
    final_predictions = {key: [] for key in sol_dict if key != "t"}

    for idx in range(num_time_windows):
        # Get the reference solution for the current time window
        sol_dict_window = train.get_sol_dict_window(sol_dict, num_time_steps, idx)
        for key in sol_dict_window:
            if key != "T":
                continue
                
            norm = jnp.linalg.norm(sol_dict_window[key])
            #print(f"Time window {idx+1}, {key} norm: {norm:.3e}")
            print(f"{norm:.3e}")


    
    

