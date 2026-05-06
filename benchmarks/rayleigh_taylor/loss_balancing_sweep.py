import os

# Deterministic
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC
# os.environ['JAX_ENABLE_X64'] = 'True'

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
import jax
jax.config.update("jax_default_matmul_precision", "highest")

import pinneval.train as train
import wandb



# Import your config builder
from configs import loss_balancing_sweep

import models
from utils import get_dataset


def dataset_fn(*args):
    return get_dataset()


def main():
    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    loss_balancing = ["gradnorm", "gradnorm_mtl", "softadapt", "relobralo", "fixed"]
    for seed in seeds:
        for scheme in loss_balancing:
            config = loss_balancing_sweep.get_config(weighting_scheme=scheme, seed=seed)

            train.train_time_windows(
                config,
                dataset_fn=get_dataset,
                model_cls=models.NavierStokes,
                evaluator_cls=models.NavierStokesEvaluator,
                log=True,
            )

            stats = jax.local_devices()[0].memory_stats()
            peak_gib = stats["peak_bytes_in_use"] / 1024**3
            print(f"Peak GPU memory usage: {peak_gib:.2f} GiB")

            wandb.finish()


if __name__ == "__main__":
    main()
