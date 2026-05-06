import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC


os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
import jax
jax.config.update("jax_default_matmul_precision", "highest")
 
import pinneval.train as train
import wandb

# Import your config builder
from configs import sampler_sweep

import models
from utils import get_dataset

def main():
    seeds = [2, 3, 5, 7, 11]
    dias = ["uniform", "vRBA", "R3", "rad", "las"]
    for seed in seeds:
        for dia in dias:
            dim=5
            config = sampler_sweep.get_config(dia_method=dia, seed=seed, dim=dim+1)

            train.init_and_train_and_evaluate(config, 
                                dataset_fn= lambda: get_dataset(dim=dim),
                                model_cls=models.HeatND,
                                evaluator_cls=models.HeatNDEvaluator,
                                log=True)
        
            stats = jax.local_devices()[0].memory_stats()
            peak_gib = stats['peak_bytes_in_use'] / 1024**3
            print(f"Peak GPU memory usage: {peak_gib:.2f} GiB") 
            
            wandb.finish()
    
if __name__ == "__main__":
    main()

