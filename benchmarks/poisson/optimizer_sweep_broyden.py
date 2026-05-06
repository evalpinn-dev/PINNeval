import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"

# Set to float64 for better stability of second-order optimizers.
os.environ["JAX_ENABLE_X64"] = "True"

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import pinneval.train as train
import wandb
from configs import optimizer_sweep

import models
from utils import get_dataset


def main():
    seeds = [2, 3, 5, 7, 11]
    opts = ["Soap_SSBroyden2"]
    dim = 5

    for seed in seeds:
        for opt in opts:
            config = optimizer_sweep.get_config(opt=opt, seed=seed, dim=dim)

            train.init_and_train_and_evaluate(
                config,
                dataset_fn=lambda: get_dataset(dim=dim),
                model_cls=models.PoissonND,
                evaluator_cls=models.PoissonNDEvaluator,
                log=True,
            )

            stats = jax.local_devices()[0].memory_stats()
            peak_gib = stats["peak_bytes_in_use"] / 1024**3
            print(f"Peak GPU memory usage: {peak_gib:.2f} GiB")

            wandb.finish()


if __name__ == "__main__":
    main()
