import ml_collections

import jax.numpy as jnp


def get_other_config(seed=42):
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "RT-NNSweep"
    wandb.tag = None

    # Physics-informed initialization
    config.use_pi_init = False

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Soap"  # "Soap_SSBroyden2"
    optim.beta1 = 0.95
    optim.beta2 = 0.95
    optim.eps = 1e-8
    optim.learning_rate = 3e-3
    optim.decay_rate = 0.9
    optim.schedule_free = False
    optim.decay_steps = 2000
    optim.warmup_steps = 2000
    config.optim.first_order_steps = 25_000
    config.optim.second_order_steps = 0

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "gradnorm"
    weighting.init_weights = ml_collections.ConfigDict(
        {
            "u_ic": 100.0,
            "v_ic": 100.0,
            "temp_ic": 10.0,
            "u_bc": 100.0,
            "v_bc": 100.0,
            "temp_bc": 10.0,
            "ru": 1.0,
            "rv": 1.0,
            "rc": 1.0,
            "re": 1.0,
        }
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 400

    # Domain Importance adapation
    config.dia = dia = ml_collections.ConfigDict()
    dia.type = "uniform"
    dia.batch_size = 64 * 128
    dia.batch_size_eval = None

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size_per_device = 8192
    training.num_time_windows = 7

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

    # Integer for PRNG random seed.
    config.seed = seed

    return config


def get_config(nn_arch, seed):
    config = get_other_config(seed=seed)

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.out_dim = 4
    arch.input_dim = 3
    arch.activation = "swish"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (2 * jnp.pi,), "axis": (1,)}
    )
    arch.reparam = None

    if nn_arch == "MLP_small":
        arch.arch_name = "Mlp"
        arch.num_layers = 5
        arch.hidden_dim = 32
        arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 32})

    elif nn_arch == "MLP_med":
        arch.arch_name = "Mlp"
        arch.num_layers = 5
        arch.hidden_dim = 64
        arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 64})

    elif nn_arch == "MLP_large":
        arch.arch_name = "Mlp"
        arch.num_layers = 5
        arch.hidden_dim = 128
        arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 128})

    elif nn_arch == "ModifiedMLP_small":
        arch.arch_name = "ModifiedMlp"
        arch.num_layers = 4
        arch.hidden_dim = 32
        arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 32})

    elif nn_arch == "ModifiedMLP_med":
        arch.arch_name = "ModifiedMlp"
        arch.num_layers = 4
        arch.hidden_dim = 64
        arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 64})

    elif nn_arch == "ModifiedMLP_large":
        arch.arch_name = "ModifiedMlp"
        arch.num_layers = 4
        arch.hidden_dim = 128
        arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 128})

    elif nn_arch == "PirateNet_small":
        arch.arch_name = "PirateNet"
        arch.num_layers = 3
        arch.hidden_dim = 24
        arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 24})

    elif nn_arch == "PirateNet_med":
        arch.arch_name = "PirateNet"
        arch.num_layers = 3
        arch.hidden_dim = 48
        arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 48})

    elif nn_arch == "PirateNet_large":
        arch.arch_name = "PirateNet"
        arch.num_layers = 3
        arch.hidden_dim = 96
        arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 96})

    config.wandb.name = "{nn}_Soap_{seed}_{hidden_dim}_{num_layers}".format(
        nn=nn_arch,
        seed=seed,
        hidden_dim=config.arch.hidden_dim,
        num_layers=config.arch.num_layers,
    )

    return config
