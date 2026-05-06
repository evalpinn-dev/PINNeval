import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC

import jax
jax.config.update("jax_default_matmul_precision", "highest")
 
import pinneval.train as train
import wandb
from configs import optimizer_sweep

import models
from utils import get_dataset

def main():
    dim=5
    seeds = [2, 3, 5, 7, 11]
    opts = ["Soap"]
    for seed in seeds:
        for opt in opts:
            config = optimizer_sweep.get_config(opt=opt, seed=seed, dim=dim+1)

            train.init_and_train_and_evaluate(config, 
                                dataset_fn=lambda: get_dataset(dim=dim),
                                model_cls=models.HeatND,
                                evaluator_cls=models.HeatNDEvaluator,
                                log=True)
        
            stats = jax.local_devices()[0].memory_stats()
            peak_gib = stats['peak_bytes_in_use'] / 1024**3
            print(f"Peak GPU memory usage: {peak_gib:.2f} GiB") 
            
            wandb.finish()
    
if __name__ == "__main__":
    main()

