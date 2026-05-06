import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC

import jax
jax.config.update("jax_default_matmul_precision", "highest")

import pinneval.train as train
import wandb

# Import your config builder
from configs import loss_balancing_sweep

import models
from utils import get_dataset


def main():
    # Choose your config here
    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    loss_balancing = ["softadapt", "relobralo", "fixed", "gradnorm", "gradnorm_mtl"]
    for seed in seeds:
        for scheme in loss_balancing:
            config = loss_balancing_sweep.get_config(weighting_scheme=scheme, seed=seed)

            train.train_time_windows(
                config,
                dataset_fn=get_dataset,
                model_cls=models.KS,
                evaluator_cls=models.KSEvaluator,
                log=True,
            )

            stats = jax.local_devices()[0].memory_stats()
            peak_gib = stats["peak_bytes_in_use"] / 1024**3
            print(f"Peak GPU memory usage: {peak_gib:.2f} GiB")

            wandb.finish()


if __name__ == "__main__":
    main()
