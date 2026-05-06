import scipy.io
import jax.numpy as jnp


def get_dataset():
    """
    Loads the Ginzburg-Landau dataset
    Returns:
        sol: dict, solution dictionary containing 'u', 'v'
        axes: dict, axes dictionary containing 't_star', 'x_star', 'y_star'
        axis_order: tuple, order of the axes 
        pde_extra_params: dict, containing 'eps' and 'k'
    """
    data = scipy.io.loadmat("data/ginzburg_landau_square.mat")

    u_ref = data["usol"]
    v_ref = data["vsol"]

    # PDE parameters
    eps = data["eps"].flatten()[0]
    k = data["k"].flatten()[0]

    t_star = data["t"].flatten()
    x_star = data["x"].flatten()
    y_star = data["y"].flatten()

    sol = {
        "u": jnp.asarray(u_ref),  # Exclude last time step for training
        "v": jnp.asarray(v_ref),

    }
    axes = {
        "t": jnp.asarray(t_star),  # Exclude last time step for training
        "x": jnp.asarray(x_star),
        "y": jnp.asarray(y_star),
    }
    axis_order = ("t", "x", "y")
    pde_extra_params = {
        "eps": eps,
        "k": k,
    }
    # Return the processed data
    return sol, axes, axis_order, pde_extra_params

      

if __name__ == "__main__":
    import pinneval.train as train
    num_time_windows = 5

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
            if key == "t":
                continue
                
            norm = jnp.linalg.norm(sol_dict_window[key])
            print(f"Time window {idx+1}, {key} norm: {norm:.3e}")
    
    
            
