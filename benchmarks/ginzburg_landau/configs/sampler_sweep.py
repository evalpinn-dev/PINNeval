import ml_collections

import jax.numpy as jnp


def get_other_config(seed=42, hidden_dim=64, num_layers=5):
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "GL-diaSweep"
    wandb.tag = None

    # Physics-informed initialization
    config.use_pi_init = True

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = num_layers
    arch.hidden_dim = hidden_dim
    arch.out_dim = 2
    arch.activation = "swish"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (jnp.pi, jnp.pi), "axis": (1, 2)}
    )
    arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 1.0, "embed_dim": hidden_dim}
    )
    arch.reparam = None

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Soap" 
    optim.beta1 = 0.95
    optim.beta2 = 0.95
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.schedule_free = False
    optim.decay_steps = 1000 
    optim.warmup_steps = 2000 
    config.optim.first_order_steps = 40_000
    config.optim.second_order_steps = 0

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "gradnorm"
    weighting.init_weights = ml_collections.ConfigDict({"u_ic": 100.0, "v_ic": 100.0, "ru": 1.0, "rv": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 400

    config.training = training = ml_collections.ConfigDict()
    training.num_time_windows = 5
    
    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = None
    saving.num_keep_ckpts = 0

    # Input shape for initializing Flax models
    arch.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = seed

    return config


def get_config(dia_method, seed):
    config = get_other_config(seed=seed)

    config.wandb.name = "{dia}_Soap_{seed}_{hidden_dim}_{num_layers}".format(dia=dia_method, seed=seed, hidden_dim=config.arch.hidden_dim, num_layers=config.arch.num_layers)

    config.dia = dia = ml_collections.ConfigDict()

    if dia_method == "vRBA":
        dia.type = "vRBA"
        dia.batch_size = 64*128
        dia.batch_size_eval = 4*64*128

    elif dia_method == "uniform":
        dia.type = "uniform"
        dia.batch_size = 64*128
        dia.batch_size_eval = None

    elif dia_method == "R3":
        dia.type = "R3"
        dia.batch_size = 64*128
        dia.batch_size_eval = None
    
    elif dia_method == "las":
        dia.type = "las"
        dia.batch_size = 64*128
        dia.batch_size_eval = None
        dia.beta = 0.2
        dia.tau = 0.002

    elif dia_method == "rad":
        dia.type = "rad"
        dia.batch_size = 64*128
        dia.batch_size_eval = 4*64*128


    return config
