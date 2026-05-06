import ml_collections

import jax.numpy as jnp


def get_config(batch_size=8192, hidden_dim=64, num_layers=4, seed=42):
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Wave"
    wandb.name = "Mlp_soap_broyden_{}_{}_{}_{}".format(hidden_dim, num_layers, batch_size, seed)
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
    # arch.periodicity = ml_collections.ConfigDict(
    #     {"period": (2 * jnp.pi,), "axis": (1,)})
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 10.0, "embed_dim": hidden_dim})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Soap_SSBroyden2"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.warmup_steps = 5000
    optim.schedule_free = False
    optim.first_order_steps = 5_000
    optim.second_order_steps = 10_000

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "gradnorm"
    weighting.init_weights = ml_collections.ConfigDict({"u0": 1.0, "u_t0": 1.0,  "res": 1.0, "bcs": 1.0})
    weighting.momentum = 0.8
    weighting.update_every_steps = 400

    weighting.use_causal = False
    weighting.causal_tol = 1.0
    weighting.num_chunks = 16

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_nonlinearities = False
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False

    # Sampler
    config.dia = sampler = ml_collections.ConfigDict()
    sampler.type = "rad"
    sampler.batch_size = batch_size
    sampler.batch_size_eval = int(batch_size *1.5)
    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 10

    # # Input shape for initializing Flax models
    arch.input_dim = 2

    # Integer for PRNG random seed.
    config.seed = seed

    return config
