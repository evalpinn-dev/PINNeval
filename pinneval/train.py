import os
import time
from jax import vmap
import optax
import jax
import jax.numpy as jnp

from functools import partial

import ml_collections
import wandb
import pinneval
from pinneval.domain_importance_adapt import IC_Axis_Sampler, get_sampler

from pinneval.log_utils import Logger
from pinneval.loss_balancing import create_loss_balancer, FixedBalancer

from flax.training import checkpoints



def save_checkpoint(state, workdir, keep=5):
    # Create the workdir if it doesn't exist.
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    # Save the checkpoint.
    if jax.process_index() == 0:
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step=step, keep=keep, overwrite=True)


def resample_samplers(model, samplers):
    if isinstance(samplers, dict):
        sampler = samplers["res"]
    else:
        sampler = samplers

    sampler.resample(model, model.state.params, model.state.weights)

    return model

def first_order_step(model, samplers, config, step, first_order_steps, balancer):
    # Handle potentially multiple samplers
    if step == 1 and config.dia.type == 'vRBA':
        if isinstance(samplers, dict):
            sampler = samplers["res"]
            sampler.prime(model, model.state.params, model.state.weights)
        else:
            samplers.prime(model, model.state.params, model.state.weights)

    if isinstance(samplers, dict):
        batch = {}
        for key, sampler in samplers.items():
            batch[key] = next(sampler)
    else:
        batch = next(samplers)

    if step == 1:
        balancer.init(model, model.state.params, batch, model.state.point_weights)
        

    if config.dia.type == "vRBA":
            model.state = model.state.update_point_weights(
                point_weights=samplers.get_lambdas()
            )

    model.state = model.step(model.state, batch, model.state.point_weights)

    # Update weights at the correct interval (skip last step to avoid a wasted update)
    if step % config.weighting.update_every_steps == 0 and step < first_order_steps - 1:
        new_weights = balancer.update(model, model.state.params, batch, model.state.point_weights, model.state.weights)
        model.state = model.state.replace(weights=new_weights)

    model = resample_samplers(model, samplers)
    
    return model, batch, samplers

def logging_and_saving(model, config, step, batch, evaluator, logger, start_time):
    if jax.process_index() == 0:
        if step % config.logging.log_every_steps == 0:
            # Get the first replica of the state and batch
            log_dict = evaluator(model.state, batch)
            wandb.log(log_dict, step)
            end_time = time.time()
            logger.log_iter(step, start_time, end_time, log_dict)

    # Saving
    if config.saving.save_every_steps is not None:
        if (step + 1) % config.saving.save_every_steps == 0:
            ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
            save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)
    
    return logger

def second_order_step(model, config, batch, step_offset, evaluator,
                      logger, log, max_iter):

    def lbfgs_log(params, i, logger, log, batch, start_time):
        if jax.process_index() == 0 and log:
            # Get the first replica of the state and batch
            state = model_state.replace(params=params)
            log_dict = evaluator(state, batch)
            wandb.log(log_dict, i)
            end_time = time.time()
            logger.log_iter(i, start_time, end_time, log_dict)
        return logger


    @partial(jax.jit,) 
    def lbfgs_update(params, opt_state, loss, grads, weights, batch, point_weights):

        value_fn = lambda p: model.loss(p, weights, batch, point_weights)

        updates, opt_state = opt.update(
            grads, opt_state, params,
            value=loss, grad=grads, value_fn=value_fn
        )
        params_new = optax.apply_updates(params, updates)

        return params_new, opt_state
    
    model_state = model.state
    weights = model.state.weights
    params = model.state.params
    point_weights = model.state.point_weights

    opt = get_quasi_newton_opt(config.optim.optimizer)
    opt_state= opt.init(params)

    for i in range(max_iter):
        start_time = time.time()
        loss, grads = jax.value_and_grad(model.loss)(params, weights, batch, point_weights)

        params_new, opt_state_new = lbfgs_update(params, opt_state, loss, grads, weights, batch, point_weights)
        # check if the update failed
        # compute the loss after the update
        loss_new = model.loss(params_new, weights, batch, point_weights)
        # if NaN then failed
        if not jnp.isfinite(loss_new):
            # no fallback to safe_broyden.
            print(f"NaN loss detected at iteration {i}. Stopping optimization.")
            return model.state.replace(params=params), logger, i
        else:
            params = params_new
            opt_state = opt_state_new
        #logging the state
        if (i + 1) % config.logging.log_every_steps == 0:
            lbfgs_log(params, i+step_offset, logger, log, batch, start_time)
        if hasattr(opt_state, 'failed') and opt_state.failed:
            print(f"Terminating at iteration {i} due to failure in line search.") 
            break

    return model_state.replace(params=params), logger, i

def get_quasi_newton_opt(optimizer_name):
    if "Lbfgs" in optimizer_name:
        opt = optax.lbfgs(learning_rate=1e-1, memory_size=500)

    elif "Broyden1" in optimizer_name:
        from pinneval.broyden import broyden
        opt = broyden(name="SSBroyden1", c1=1e-4, c2=0.9, alpha0=1.0, tau=0.5, eps=1e-12)

    elif "Broyden2" in optimizer_name:
        from pinneval.broyden import broyden
        opt = broyden(name="SSBroyden2", c1=1e-4, c2=0.9, alpha0=1.0, tau=0.5, eps=1e-12) 

    elif "Broyden3" in optimizer_name:
        from pinneval.broyden import broyden
        opt = broyden(name="SSBroyden3", c1=1e-4, c2=0.9, alpha0=1.0, tau=0.5, eps=1e-12)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return opt
    
def make_domain(axes, axis_order):
    """
    Construct domain array [[min, max], ...] in fixed axis order.

    Parameters
    ----------
    axes : dict
        Mapping axis name -> coordinate array
    axis_order : tuple
        Desired axis ordering, e.g. ("t", "x", "y")
    time_extension : float
        Extend t_max by time_extension * dt

    Returns
    -------
    domain : jnp.ndarray, shape e.g. (ndim, 2)
    """
    bounds = []

    eps = 0
    for axis in axis_order:
        if axis not in axes:
            continue
        
        arr = axes[axis]
        if axis == 'coords':
            for i in range(arr.shape[1]):
                min_axis = jnp.min(arr[:, i])
                max_axis = jnp.max(arr[:, i])
                bounds.append(jnp.array([min_axis+eps, max_axis-eps]))
        else:        
            a0 = arr[0]
            a1 = arr[-1]
            bounds.append(jnp.array([a0, a1]))

    return jnp.stack(bounds)

def make_initial_dict(sol_dict: dict[str, jnp.ndarray], axes_dict: dict[str, jnp.ndarray] = None):
    """
    Construct initial condition dictionary from solution dictionary.
    If no time axis is present, return None.
    """
    if "t" not in axes_dict:
        return None
    
    make_initial_dict = {}
    for key, val in sol_dict.items():
        make_initial_dict[key + "0"] = val[0, ...]
    return make_initial_dict

def get_time_window_bounds(num_time_points, num_time_windows):
    if num_time_windows < 1:
        raise ValueError("num_time_windows must be at least 1.")
    if num_time_points < 2:
        raise ValueError("At least two time points are required for time-window training.")

    num_intervals = num_time_points - 1
    if num_time_windows > num_intervals:
        raise ValueError(
            "num_time_windows cannot exceed the number of time intervals "
            f"({num_intervals})."
        )

    base_steps = num_intervals // num_time_windows
    remainder = num_intervals % num_time_windows

    bounds = []
    start = 0
    for idx in range(num_time_windows):
        interval_len = base_steps + (1 if idx < remainder else 0)
        stop = start + interval_len + 1
        bounds.append((start, stop))
        start = stop - 1

    return bounds


def get_sol_dict_window_by_indices(sol_dict, start, stop):
    sol_dict_window = {}
    for key, val in sol_dict.items():
        if key == 't':
            sol_dict_window[key] = val[start:stop]
            #print("Time window:", sol_dict_window[key][0], "to", sol_dict_window[key][-1])
            # shift to start to 0
            sol_dict_window['t'] = sol_dict_window['t'] - sol_dict_window['t'][0]
        else:
            sol_dict_window[key] = val[start:stop, ...]
    return sol_dict_window


def get_sol_dict_window(sol_dict, num_time_steps, idx):
    start = num_time_steps * idx
    stop = num_time_steps * (idx + 1) + 1
    return get_sol_dict_window_by_indices(sol_dict, start, stop)

def physics_informed_init(config, axes_dict, initial_dict, pde_params_dict, model_cls):
    """
    Perform physics-informed initialization for the given model.
    Parameters
    ----------
    config : ml_collections.ConfigDict
        Configuration dictionary.
    axes_dict : dict
        Dictionary of axes arrays.
    initial_dict : dict
        Dictionary of initial condition arrays.
    pde_params_dict : dict
        Dictionary of PDE parameters.
    model_cls : class
        Model class to instantiate.
    time_stride : int
        Stride for time axis when building inputs.
    """
    print("Use physics-informed initialization...")
    # if hatattr config.pi_init_stride then time_stride = config.pi_init_stride else time_stride = 10
    if hasattr(config, "pi_init_stride"):
        time_stride = config.pi_init_stride
    else:
        time_stride = 10

    config.arch.pi_init = None  # reset any existing pi_init
    # Build the model
    model = model_cls(config=config, axes_dict=axes_dict, initial_dict=initial_dict, pde_params_dict=pde_params_dict)
    params = model.state.params

    # Build the inputs for the feature matrix
    t = axes_dict["t"][::time_stride]
    t_scaled = t / t[-1]

    # Spatial axes (everything except time)
    spatial_axes = [k for k in axes_dict.keys() if k != "t"]
    # only do if coords is not in spatial_axes
    spatial_grids = [axes_dict[k] for k in spatial_axes]
    
    # build inputs
    if 'coords' in spatial_axes:
        coords = axes_dict['coords']                # (N_coords, d)
        N_t = t_scaled.shape[0]
        N_coords = coords.shape[0]

        t_rep = jnp.repeat(t_scaled, repeats=N_coords)     # (N_t*N_coords,)
        coords_rep = jnp.tile(coords, (N_t, 1))            # (N_t*N_coords, d)

        inputs = jnp.concatenate([t_rep[:, None], coords_rep], axis=1)  # (N_t*N_coords, 1+d)
    else:
        # Meshgrid over all axes (independent spatial axes)
        mesh = jnp.meshgrid(t_scaled, *spatial_grids, indexing="ij")
        inputs = jnp.stack([m.reshape(-1) for m in mesh], axis=1)       # (N_samples, 1+ndim)

    N_samples = inputs.shape[0]

    # build targets
    field_names = list(initial_dict.keys())
    targets_cols = []

    if 'coords' in spatial_axes:
        # fields should be defined on coords: (N_coords,) or (N_coords, k)
        for name in field_names:
            field = jnp.asarray(initial_dict[name])

            if field.shape[0] != N_coords:
                raise ValueError(
                    f"Field '{name}' first dim must be N_coords={N_coords}, got {field.shape}."
                )

            # replicate across time in time-major order: [t0 all coords, t1 all coords, ...]
            if field.ndim == 1:
                col = jnp.tile(field, (N_t,))              # (N_t*N_coords,)
                targets_cols.append(col)
            else:
                # vector-valued per coord: (N_coords, k) -> (N_t*N_coords, k)
                col = jnp.tile(field, (N_t, 1))            # (N_t*N_coords, k)
                # If you truly want one column per field, you must decide how to handle k>1.
                # Here we flatten k into multiple target columns:
                targets_cols.append(col)

        # stack targets: scalar fields -> (N_samples, n_fields)
        # vector fields -> concatenate their columns
        stacked = []
        for col in targets_cols:
            if col.ndim == 1:
                stacked.append(col[:, None])
            else:
                stacked.append(col)  # (N_samples, k)
        targets = jnp.concatenate(stacked, axis=1)

    else:
        # grid case: field shaped (Nx, Ny, ...)
        for name in field_names:
            field = jnp.asarray(initial_dict[name])
            tiled = jnp.tile(field, (t_scaled.shape[0],) + (1,) * field.ndim)
            targets_cols.append(tiled.reshape(-1))          # (N_samples,)

        targets = jnp.stack(targets_cols, axis=1)           # (N_samples, n_fields)

    # final sanity check
    if targets.shape[0] != N_samples:
        raise ValueError(f"targets has {targets.shape[0]} rows, but inputs has {N_samples} rows.")

    # Compute feature matrix
    feat_matrix, _ = vmap(model.state.apply_fn, (None, 0))(params, inputs)




    # Solve for coefficients using least squares
    coeffs = []
    for i in range(targets.shape[1]):
        c, res, rank, s = jnp.linalg.lstsq(
            feat_matrix,
            targets[:, i],
            rcond=None,
        )
        coeffs.append(c)

        print(
            f"PI-init residual field {i}:",
            res if res.size > 0 else "rank-deficient",
        )

    coeffs = jnp.stack(coeffs, axis=1)
    # Set the pi_init in the config
    config.arch.pi_init = coeffs 

    del model, params 

    model = model_cls(config=config, axes_dict=axes_dict, initial_dict=initial_dict, pde_params_dict=pde_params_dict)
    return model

def train_and_evaluate(config: ml_collections.ConfigDict,
                        first_order_steps: int,
                        second_order_steps: int,
                        model: pinneval.models.PINN,
                        samplers: dict,
                        evaluator: pinneval.evaluator.BaseEvaluator,
                        step_offset: int,
                        logger: Logger,
                        log: bool = True,
                        balancer=None):

    if balancer is None:
        balancer = create_loss_balancer(config, dict(config.weighting.init_weights))

    if first_order_steps > 0:
        print(f"Using first-order optimizer for {first_order_steps} steps...")

        for step in range(1, first_order_steps+1):
            start_time = time.time()
            # First-order optimization step
            model, batch, samplers = first_order_step(model, samplers, config, step, first_order_steps, balancer)

            # Logging and saving
            if log: 
                logger = logging_and_saving(model, config, step + step_offset, batch, evaluator, logger, start_time)

        step_offset += first_order_steps

    stats = jax.local_devices()[0].memory_stats()
    peak_gib = stats['peak_bytes_in_use'] / 1024**3
    print(f"Peak GPU memory usage: {peak_gib:.2f} GiB") 
    if second_order_steps > 0:
        balancer = FixedBalancer()  # freeze weights during second-order optimization
        print(f"Using second-order optimizer for {second_order_steps} steps...")
        # if samplers are dict then resample only the residual sampler
        # test if the config has initial_sampler attribute and if it's true set the ics sampler to full
        if hasattr(config.dia, "initial_sampler") and config.dia.initial_sampler:
            samplers['ics'].mode = 'full'
        if isinstance(samplers, dict):
            model = resample_samplers(model, samplers)
            batch = {}
            for key, sampler in samplers.items():
                batch[key] = next(sampler)
        else:
            model = resample_samplers(model, samplers)
            batch = next(samplers)
        model.state, logger, steps_done = \
                second_order_step(model=model,
                                config=config,
                                batch=batch, 
                                step_offset=step_offset, 
                                evaluator=evaluator,
                                logger=logger, 
                                log=log, 
                                max_iter=second_order_steps)
        step_offset += steps_done
    
    if hasattr(config.logging, "log_preds") and config.logging.log_preds:
        params = model.state.params
        evaluator.log_preds(params)
        
    stats = jax.local_devices()[0].memory_stats()
    peak_gib = stats['peak_bytes_in_use'] / 1024**3
    print(f"Peak GPU memory usage: {peak_gib:.2f} GiB") 
        
    return model, step_offset, logger
    
def init_and_train_and_evaluate(config: ml_collections.ConfigDict, 
                                dataset_fn,
                                model_cls,
                                evaluator_cls,
                                log=True,):
    # Initialize W&B
    wandb_config = config.wandb
    if log:
        wandb.init(project=wandb_config.project, name=wandb_config.name, reinit=True)

    # Initialize logger
    logger = Logger()

    # Get the dataset, the domain and the initial condition points

    sol_dict, axes_dict, axis_order, pde_params_dict = dataset_fn()
    
    
    dom = make_domain(axes_dict, axis_order)
    initial_dict = make_initial_dict(sol_dict, axes_dict=axes_dict)

    # Initialize the sampler
    res_sampler = get_sampler(config, dom)

    # Initialize the model
    if hasattr(config, "use_pi_init") and config.use_pi_init:
        model = physics_informed_init(config, axes_dict, initial_dict, pde_params_dict, 
                                            model_cls)
    else:
        model = model_cls(config=config, axes_dict=axes_dict, 
                        initial_dict=initial_dict, pde_params_dict=pde_params_dict)

    #print number of trainable parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(model.state.params))
    print(f"Number of trainable parameters: {num_params}")
    # Initialize evaluator
    evaluator = evaluator_cls(config=config, model=model, sol_dict=sol_dict)


    # train and evaluate
    model, _, logger = train_and_evaluate(config=config,
                                            first_order_steps=config.optim.first_order_steps,
                                            second_order_steps=config.optim.second_order_steps,
                                            model=model,
                                            samplers=res_sampler,
                                            evaluator=evaluator,
                                            step_offset=0,
                                            logger=logger,
                                            log=log)
    if config.saving.num_keep_ckpts > 0:
            print("Saving checkpoint")
            ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
            save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)
    return model, evaluator

def train_curriculum(config: ml_collections.ConfigDict, 
                     changing_model_param,
                     model_cls,
                     dataset_fn,
                     evaluator_cls,
                     log=True,):
    
    assert len(config.optim.first_order_steps) == len(changing_model_param)
    # Initialize W&B
    wandb_config = config.wandb
    if log:
        wandb.init(project=wandb_config.project, name=wandb_config.name, reinit=True)

    # Initialize logger
    logger = Logger()
    old_model_params = None

    step_offset = 0
    all_first_order_steps = config.optim.first_order_steps
    all_second_order_steps = config.optim.second_order_steps

    # run 
    for first_order_steps, second_order_steps, parameter in \
        zip(all_first_order_steps, all_second_order_steps, changing_model_param):

        param_value = parameter
        print(f"Training with parameter value: {param_value}")

        # Get the dataset, the domain and the initial condition points
        sol_dict, axes_dict, axis_order, pde_params_dict = dataset_fn(param_value)
        dom = make_domain(axes_dict, axis_order)
        initial_dict = make_initial_dict(sol_dict, axes_dict=axes_dict)

        # Initialize the sampler
        res_sampler = get_sampler(config, dom)
        
        # Initialize the model
        model = model_cls(config=config, axes_dict=axes_dict, initial_dict=initial_dict, pde_params_dict=pde_params_dict)
        

        # if we have a pre-trained model, load the params
        if old_model_params is not None:
            new_model = model_cls(config=config, axes_dict=axes_dict, initial_dict=initial_dict, pde_params_dict=pde_params_dict)
            new_model.state = model.state.replace(params=old_model_params)
            model = new_model
        
        # Initialize evaluator
        evaluator = evaluator_cls(config=config, model=model, sol_dict=sol_dict)

        
        model, step_offset, logger = train_and_evaluate(config=config,
                                                first_order_steps=first_order_steps,
                                                second_order_steps=second_order_steps,
                                                model=model,
                                                samplers=res_sampler,
                                                evaluator=evaluator,
                                                step_offset=step_offset,
                                                logger=logger,
                                                log=log)

        old_model_params = model.state.params
    
    if config.saving.num_keep_ckpts > 0:
            print("Saving checkpoint")
            ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
            save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)
    return model, evaluator

def track_time_windows_errors(window_error_table, window_error_history, log_dict, idx):

    # Keep only error metrics
    error_dict = {
        key: float(value)
        for key, value in log_dict.items()
        if "error" in key.lower()
    }

    # --- Lazy table creation (after first window) ---
    if window_error_table is None:
        # Table columns: window_idx + all error keys
        columns = ["row_type", "window_idx"] + list(error_dict.keys())
        window_error_table = wandb.Table(columns=columns, log_mode="MUTABLE")

        # Initialize history storage
        for key in error_dict:
            window_error_history[key] = []

    # --- Add row for this window ---
    row = ["window", idx] + [error_dict[k] for k in error_dict]
    window_error_table.add_data(*row)

    # --- Store for averaging ---
    for key, value in error_dict.items():
        window_error_history[key].append(value)

    # --- Log table (safe to call every window) ---
    wandb.log({"windows/l2_error_table": window_error_table})
    return window_error_table, window_error_history

def train_time_windows(config: ml_collections.ConfigDict, 
                     model_cls,
                     dataset_fn,
                     evaluator_cls,
                     log=True,): 
    if log:
        wandb_config = config.wandb
        wandb.init(project=wandb_config.project, name=wandb_config.name)

    sol_dict, axes_dict, axis_order, pde_params_dict = dataset_fn()
    initial_dict = make_initial_dict(sol_dict, axes_dict=axes_dict)

    # add t to the sol_dict for conflicts of slicing in the last window
    sol_dict["t"] = axes_dict["t"]

    time_window_bounds = get_time_window_bounds(
        len(axes_dict["t"]),
        config.training.num_time_windows,
    )
    interval_lengths = [stop - start - 1 for start, stop in time_window_bounds]
    if len(set(interval_lengths)) > 1:
        print(
            "Using uneven time windows to cover all time points. "
            f"Intervals per window: {interval_lengths}"
        )

    # Each window resets its local t-axis to start at zero.
    step_offset = 0
    logger = Logger()
    
    window_error_table = None            
    window_error_history = {}      
         
    # collect all final predictions for final evaluation, that is every key in sol_dict except t
    final_predictions = {key: [] for key in sol_dict if key != "t"}
    ref_solution = {key: [] for key in sol_dict if key != "t"}

    for idx, (window_start, window_stop) in enumerate(time_window_bounds):
        # Get the reference solution for the current time window
        sol_dict_window = get_sol_dict_window_by_indices(
            sol_dict,
            window_start,
            window_stop,
        )
        axes_dict['t'] = sol_dict_window['t']   

        # check first if config has the attr then if they are true
        if hasattr(config, "use_pi_init") and config.use_pi_init:
            model = physics_informed_init(config, axes_dict, initial_dict, pde_params_dict, 
                                              model_cls)
        else:
            model = model_cls(config=config, axes_dict=axes_dict, 
                          initial_dict=initial_dict, pde_params_dict=pde_params_dict)
            
        
        evaluator = evaluator_cls(config=config, model=model, sol_dict=sol_dict_window)

        dom = make_domain(axes_dict, axis_order)
        res_sampler = get_sampler(config, dom)
        # if inital or bc sampler do those somehow
        if hasattr(config.dia, "initial_sampler") and config.dia.initial_sampler:
            ICSampler = IC_Axis_Sampler(
                init_dict=initial_dict,
                axes_dict=axes_dict,
                batch_size=config.dia.initial_batch_size,
                mode=config.dia.initial_sampler_mode,
            )
            samplers = {
                "res": res_sampler,
                "ics": iter(ICSampler),
            }
        else:
            samplers = res_sampler

        model, step_offset, logger = train_and_evaluate(config=config,
                                                first_order_steps=config.optim.first_order_steps,
                                                second_order_steps=config.optim.second_order_steps,
                                                model=model,
                                                samplers=samplers,
                                                evaluator=evaluator,
                                                step_offset=step_offset,
                                                logger=logger,
                                                log=log)
        
        # compute l2 error for the current time window
        evaluator.log_errors(model.state.params)  
        log_dict = evaluator.log_dict
        window_error_table, window_error_history = track_time_windows_errors(window_error_table, window_error_history, log_dict, idx)
        
        stats = jax.local_devices()[0].memory_stats()
        peak_gib = stats['peak_bytes_in_use'] / 1024**3
        print(f"Peak GPU memory usage: {peak_gib:.2f} GiB") 

        # collect final predictions
        preds = model.pred_domain(model.state.params)
        # unpack the dict and append to final_predictions
        for key in final_predictions:
            if idx != config.training.num_time_windows - 1:
                # exclude last time step
                final_predictions[key].append(preds[key][:-1])
                ref_solution[key].append(sol_dict_window[key][:-1])
            else:
                final_predictions[key].append(preds[key])  
                ref_solution[key].append(sol_dict_window[key])  
        # Save checkpoint for the current time window
        if config.saving.num_keep_ckpts > 0:
            print("Saving checkpoint for time window {}".format(idx+1))
            ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "time_window_{}".format(idx+1))
            save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

        # Update the initial condition for the next time window
        new_initial_dict = model.pred_last_t(model.state.params)
        # check if new_init dict contains NaNs
        for key in new_initial_dict:
            if jnp.isnan(new_initial_dict[key]).any():
                raise ValueError(f"NaN values found in the new initial condition for key '{key}' at time window {idx+1}.")
        # check if keys match

        assert set(new_initial_dict.keys()) == set(initial_dict.keys()), "Keys of the new and old initial dict do not match!"
        initial_dict = new_initial_dict
        del model

    
    print("Training on all time windows completed. Evaluating final predictions...")
    # for all keys in final_predictions stack along time axis and save to wandb
    for key in final_predictions:
        full_pred = jnp.concatenate(final_predictions[key], axis=0)
        full_ref_solution = jnp.concatenate(ref_solution[key], axis=0)
        # print shape
        print(f"Final prediction shape for {key}: {full_pred.shape}")
        rel_l2_error = jnp.linalg.norm(full_ref_solution - full_pred) / jnp.linalg.norm(full_ref_solution)
        wandb.run.summary[f"final_rel_l2_error_{key}"] = float(rel_l2_error)
        print(f"Final relative L2 error for {key}: {rel_l2_error:.3e}")
