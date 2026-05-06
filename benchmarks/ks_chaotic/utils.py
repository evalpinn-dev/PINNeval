import scipy.io
import jax.numpy as jnp

def get_dataset():
    """
    Loads the Kuramoto-Sivashinsky dataset

    Returns
    -------
    sol : dict
        Solution dictionary containing:
            - 'u': solution array with shape (Nt, Nx)
    axes : dict
        Axes dictionary containing:
            - 't': time axis
            - 'x': spatial x-axis
    axis_order : tuple
        Order of the axes, given by ('t', 'x').
    pde_extra_params : None
    """
    data = scipy.io.loadmat("data/ks_chaotic.mat")
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()

    sol = {
        "u": jnp.asarray(u_ref),
    }
    axes = {
        "t": jnp.asarray(t_star),
        "x": jnp.asarray(x_star),
    }
    axis_order = ("t", "x")
    pde_extra_params = None

    return sol, axes, axis_order, pde_extra_params


def get_dataset_and_solution():
    sol_dict, axes_dict, axis_order, pde_params_dict = get_dataset()
    # put solution into pde_params_dict 
    pde_params_dict = {"u": sol_dict["u"]}
    return sol_dict, axes_dict, axis_order, pde_params_dict

if __name__ == "__main__":
    import pinneval.train as train
    num_time_windows = 10

    sol_dict, axes_dict, axis_order, pde_params_dict = get_dataset()
    
    # add t to the sol_dict for conflicts of slicing in the last window
    sol_dict["t"] = axes_dict["t"]
    print(f"Time axis: {axes_dict['t']}")

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
    
