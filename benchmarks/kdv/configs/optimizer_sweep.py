import ml_collections

import jax.numpy as jnp


def get_other_config(seed=42):
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "KdV-OptSweep"
    wandb.tag = None

    # Physics-informed initialization
    config.use_pi_init = True
    config.pi_init_type = "initial_condition"

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "gradnorm"
    weighting.init_weights = ml_collections.ConfigDict({"ics": 10.0, "res": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    # Domain Importance adapation
    config.dia = dia = ml_collections.ConfigDict()
    dia.type = "las"
    dia.batch_size = 64 * 128
    dia.batch_size_eval = None
    dia.beta = 0.2
    dia.tau = 0.002

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


def get_config(opt, seed):
    config = get_other_config(seed=seed)

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.out_dim = 1
    arch.input_dim = 2
    arch.activation = "tanh"
    arch.periodicity = ml_collections.ConfigDict({"period": (jnp.pi,), "axis": (1,)})
    arch.reparam = None

    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = opt

    if opt in ["Adam", "Soap", "Muon"]:
        # Architecture setting for first-order / quasi-second-order methods.
        arch.arch_name = "PirateNet"
        arch.num_layers = 3
        arch.hidden_dim = 256
        arch.fourier_emb = ml_collections.ConfigDict(
            {"embed_scale": 1, "embed_dim": 256}
        )

    if opt == "Adam":
        optim.beta1 = 0.9
        optim.beta2 = 0.999
        optim.learning_rate = 1e-3
        optim.eps = 1e-8
        optim.decay_rate = 0.9
        optim.schedule_free = False
        optim.decay_steps = 2000
        optim.warmup_steps = 2000
        optim.first_order_steps = 100_000
        optim.second_order_steps = 0

    elif "Soap" in opt:
        optim.beta1 = 0.95
        optim.beta2 = 0.95
        optim.learning_rate = 3e-3
        optim.eps = 1e-8
        optim.decay_rate = 0.9
        optim.schedule_free = False
        optim.decay_steps = 2000
        optim.warmup_steps = 2000
        optim.first_order_steps = 100_000
        optim.second_order_steps = 0

    if opt in ["Soap_SSBroyden2"]:
        arch.arch_name = "Mlp"
        arch.num_layers = 5
        arch.hidden_dim = 64
        arch.fourier_emb = ml_collections.ConfigDict(
            {"embed_scale": 1, "embed_dim": 64}
        )
        optim.first_order_steps = 8_000
        optim.second_order_steps = 12_000

    config.wandb.name = "{opt}_{nn}_{hidden_dim}_{num_layers}_{seed}".format(
        nn=arch.arch_name,
        opt=opt,
        seed=seed,
        hidden_dim=arch.hidden_dim,
        num_layers=arch.num_layers,
    )

    return config
