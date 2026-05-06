import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"

import jax
jax.config.update("jax_default_matmul_precision", "highest")

import pinneval.train as train
import wandb
from configs import nn_sweep

import models
from utils import get_dataset


def main():
    seeds = [2, 3, 5, 7, 11]
    nn_archs = ["MLP_small", "MLP_med", "MLP_large",
                "ModifiedMLP_small", "ModifiedMLP_med", "ModifiedMLP_large",
                "PirateNet_small", "PirateNet_med", "PirateNet_large"]
    for seed in seeds:
        for nn_arch in nn_archs:
            dim = 5
            config = nn_sweep.get_config(nn_arch=nn_arch, seed=seed, dim=dim + 1)

            train.init_and_train_and_evaluate(
                config,
                dataset_fn=lambda: get_dataset(dim=dim),
                model_cls=models.HeatND,
                evaluator_cls=models.HeatNDEvaluator,
                log=True,
            )

            stats = jax.local_devices()[0].memory_stats()
            peak_gib = stats["peak_bytes_in_use"] / 1024**3
            print(f"Peak GPU memory usage: {peak_gib:.2f} GiB")

            wandb.finish()


if __name__ == "__main__":
    main()
