import ml_collections

import jax.numpy as jnp


def get_other_config(seed=42, hidden_dim=64, num_layers=5, dim=5):
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "Heat-WeightingSweep"
    wandb.tag = None

    # Physics-informed initialization
    config.use_pi_init = False

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = num_layers
    arch.hidden_dim = hidden_dim
    arch.out_dim = 1
    arch.activation = "tanh"
    arch.periodicity = None
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
    optim.decay_steps = 500
    optim.warmup_steps = 500
    config.optim.first_order_steps = 40_000
    config.optim.second_order_steps = 0

    # Domain Importance adapation
    config.dia = dia = ml_collections.ConfigDict()
    dia.type = "uniform"
    dia.batch_size = 64 * 128
    dia.batch_size_eval = None

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
    arch.input_dim = dim

    # Integer for PRNG random seed.
    config.seed = seed

    return config


def get_config(weighting_scheme, seed, dim):
    config = get_other_config(seed=seed, dim=dim)

    config.wandb.name = "{weighting_scheme}_Soap_{seed}_{hidden_dim}_{num_layers}".format(
        weighting_scheme=weighting_scheme,
        seed=seed,
        hidden_dim=config.arch.hidden_dim,
        num_layers=config.arch.num_layers,
    )

    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.init_weights = ml_collections.ConfigDict(
        {"u_ic": 100.0, "g_bc": 50.0, "r": 1.0}
    )

    if weighting_scheme == "gradnorm":
        weighting.scheme = "gradnorm"
        weighting.momentum = 0.9
        weighting.update_every_steps = 1000

    elif weighting_scheme == "fixed":
        weighting.scheme = "fixed"
        weighting.momentum = None
        weighting.update_every_steps = 1e10

    elif weighting_scheme == "relobralo":
        weighting.scheme = "relobralo"
        weighting.update_every_steps = 1
        weighting.T = 1e-1
        weighting.rho = 0.999
        weighting.alpha = 0.99
        weighting.momentum = 0

    elif weighting_scheme == "softadapt":
        weighting.scheme = "softadapt"
        weighting.update_every_steps = 1
        weighting.T = 1e-1
        weighting.rho = 1
        weighting.alpha = 0.9
        weighting.momentum = 0

    elif weighting_scheme == "gradnorm_mtl":
        weighting.scheme = "gradnorm_mtl"
        weighting.gradnorm_mtl_lr = 1e-2
        weighting.update_every_steps = 100
        weighting.gradnorm_mtl_alpha = 2.0
        weighting.momentum = 0

    return config
