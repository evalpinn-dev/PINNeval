# PINNEval

PINNEval is a JAX/Flax library for evaluating physics-informed neural network
(PINN) training choices across a collection of PDE benchmarks. The codebase
contains reusable training components under `pinneval/` and benchmark scripts
under `benchmarks/`.

This repository is prepared for anonymous double-blind review. Please do not
try to infer author identities from repository metadata, package ownership,
or external experiment dashboards. If you find identifying material in this
artifact, please report it through the review process rather than using it for
de-anonymization.

## What Is Included

- PINN base abstractions and training loops.
- Flax architectures: `Mlp`, `ModifiedMlp`, and `PirateNet`.
- Optimizer support for Adam, SOAP, and SOAP/Broyden
  hybrid experiments used in the benchmark scripts.
- Domain importance/adaptive samplers: uniform, grid, RAD, LAS, R3, and vRBA.
- Loss-balancing methods: fixed/equal weighting, GradNorm, learning-rate
  annealing, ReLoBRaLo, SoftAdapt, and GradNorm-MTL.
- Benchmark implementations for Allen-Cahn, Burgers, heat, KdV,
  Kuramoto-Sivashinsky, Poisson, Rayleigh-Taylor, and wave
  problems.

## Repository Layout

```text
pinneval/
  pinneval/                 # Reusable library code
    archs.py                # Flax network architectures
    models.py               # Base PINN state/model utilities
    train.py                # Training and evaluation loops
    domain_importance_adapt.py
    loss_balancing.py
    broyden.py
    evaluator.py
    log_utils.py
  benchmarks/               # Reproduction scripts and benchmark definitions
    <problem>/
      configs/              # ml_collections experiment configs
      data/                 # Reference data used by utils.py, when needed
      models.py             # Problem-specific PINN and evaluator
      *_sweep.py            # Benchmark entrypoints
  setup.py
```

The `benchmarks/` directory is intended to be run from the individual benchmark
directories. It is not currently installed as an importable Python package.

## Installation

Create a fresh environment, then install the package in editable mode:

```bash
cd pinneval
python -m pip install -r requirements.txt
python -m pip install -e .
```

The package depends on JAX, Flax, Optax, SciPy, NumPy, Matplotlib,
`ml_collections`, W&B, `tabulate`, and SOAP for JAX.

For full benchmark reproduction, use the accelerator/JAX installation appropriate for
your CUDA or TPU environment.

SOAP support is provided by the upstream `soap-jax` package, pinned in
`requirements.txt` for reproducibility. If installing from a bundled local copy
instead, run:

```bash
python -m pip install -e ../soap_jax


```bash
python -m pip install -e ../soap_jax
```

## Basic Import Check

After installation, run:

```bash
python -c "import pinneval; print('pinneval import ok')"
```

If this fails with a missing dependency, install the dependency reported by the
exception and rerun the check.

## Running Benchmarks

Benchmark scripts are organized by problem. Run commands from the corresponding
benchmark directory so local imports such as `from configs import ...` and
`import models` resolve correctly.

Example:

```bash
cd benchmarks/wave
python optimizer_sweep_soapf32.py
```

Each benchmark directory contains the same five experiment entrypoints:

```text
loss_balancing_sweep.py
nn_sweep.py
samplersweep.py
optimizer_sweep_soapf32.py
optimizer_sweep_broyden.py
```

The canonical command for a particular table or figure should be taken from the
submission's reproduction instructions. Full sweeps can run for a long time and
often execute multiple seeds.

## Logging

The benchmark scripts use Weights & Biases for experiment logging. For anonymous
review, do not connect runs to an identifying user, team, or project. Suitable
options include:

```bash
export WANDB_MODE=offline
```

or:

```bash
export WANDB_DISABLED=true
```

Some lower-level training functions accept `log=False`, but the benchmark
entrypoints generally initialize and finish W&B runs directly. If you need fully
offline operation, set the environment variables above before launching a run.

## Data

Several benchmarks include the reference data used by their `utils.py` loaders
in `benchmarks/*/data` as `.mat` or `.npy` files. Data-generation scripts and
unreferenced auxiliary datasets are intentionally omitted from this anonymous
review artifact. The largest included datasets are Ginzburg-Landau and
Rayleigh-Taylor.

For anonymous review, use the included data only for reproducing the submitted
experiments. Redistribution terms and final archival packaging should be checked
before any non-review release.

## License

The software in this repository is licensed under the Apache License, Version
2.0. See `LICENSE` and `NOTICE`.

For double-blind review, copyright attribution is currently listed as
"Anonymous Authors". This is intentional and may be replaced with the appropriate
copyright holder information after de-anonymization is allowed.

Benchmark data may have separate provenance or redistribution terms. Treat the
included data as review/reproduction material unless a data-specific license is
provided.

## Reproducibility Notes

- Experiment configs are `ml_collections.ConfigDict` files under each
  benchmark's `configs/` directory.
- Seeds are set in the benchmark configs or sweep scripts.
- Full sweeps can be expensive; many scripts run multiple seeds and long
  training schedules.
- Checkpoint writing is controlled by `config.saving`. Several default configs
  set `save_every_steps = None` and `num_keep_ckpts = 0`.
- Output directories are created relative to the current working directory and
  the configured W&B run name when checkpointing is enabled.

## Anonymous Review Constraints

This artifact is intended for double-blind review only.

- Do not add author names, institutional identifiers, personal accounts, or
  non-anonymous W&B project links to issues, forks, commits, or derived copies
  used during review.
- Do not inspect commit history or repository ownership as a way to identify
  the authors.
- Do not cite this artifact with author-identifying metadata until the review
  process permits de-anonymization.
- If a local modification is needed to run an experiment, describe the change
  without introducing identifying paths or usernames.

## Known Limitations Of This Review Artifact

- The benchmark directory is not packaged as an installable module.
- The benchmark scripts are intentionally minimal reproduction entrypoints and
  are meant to be run from their respective benchmark directories.
- The current package metadata is intentionally minimal and may be revised for a
  public, non-anonymous release.
- Dependency versions are not yet locked in this artifact.
