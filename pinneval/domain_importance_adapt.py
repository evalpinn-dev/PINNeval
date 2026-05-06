from functools import partial
import jax.numpy as jnp
from jax import lax, jit, grad, random

class BaseSampler:
    def __init__(self, batch_size, seed):
        self.batch_size = batch_size
        self.key = random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        #keys = random.split(subkey)
        return self.data_generation(self.key)

    def __getitem__(self, index):
        return next(self)

    def data_generation(self, key):
        raise NotImplementedError("Subclasses should implement this!")

    def resample(self, *args, **kwargs):
        return None

class UniformSampler(BaseSampler):
    def __init__(self, dom, batch_size, seed):
        super().__init__(batch_size, seed)
        self.dom = dom
        self.dim = dom.shape[0]

    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )
        return batch

class IC_Axis_Sampler(BaseSampler):
    def __init__(
        self,
        init_dict: dict,        # e.g. {"u": u0, "v": v0}
        axes_dict: dict,        # must contain "coords"
        batch_size: int,
        mode: str = "random",   # "random" or "full"
        seed: int = 1234,
    ):
        super().__init__(batch_size, seed)
        self.mode = mode

        self.coords = axes_dict["coords"]

        self.fields = init_dict   # dict[str, array]

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == "full":
            return self._full_batch()
        else:
            return self._random_batch()

    # -------------------------
    # Random batch
    # -------------------------
    def _random_batch(self, key):
        idx = random.choice(
            key,
            self.coords.shape[0],
            shape=(self.batch_size,),
        )

        batch = {
            "coords": self.coords[idx, :]
        }

        for name, field in self.fields.items():
            batch[name] = field[idx]

        return batch

    def _random_batch(self):
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, self.num_devices)
        return self._random_batch(keys)

    def _full_batch(self):
        batch = {"coords": self.coords}
        for name, field in self.fields.items():
            batch[name] = field
        return batch
     
class GridSampler(BaseSampler):
    def __init__(self, dom, n_points):
        """
        Initialize GridSampler for creating uniform grids over domain.
        
        Args:
            dom: Domain boundaries, shape (dim, 2) where dom[:, 0] is lower bounds
                 and dom[:, 1] is upper bounds
            n_points: List or array of number of points per dimension
            seed: Random seed for grid generation
        """
        self.dom = dom
        self.dim = dom.shape[0]
        self.n_points = jnp.array(n_points)
        
        # Calculate total batch size from grid dimensions
        batch_size = int(jnp.prod(self.n_points))
        super().__init__(batch_size, 1234)
        
        # Pre-generate the grid (immutable in JAX philosophy)
        self._grid_points = self._generate_grid()
    
    def _generate_grid(self):
        """Generate the complete grid of points."""
        # Create 1D arrays for each dimension
        grids_1d = [
            jnp.linspace(self.dom[i, 0], self.dom[i, 1], self.n_points[i])
            for i in range(self.dim)
        ]
        
        # Create meshgrid - JAX uses 'ij' indexing by default in meshgrid
        grids = jnp.meshgrid(*grids_1d, indexing='ij')
        
        # Stack and reshape to (n_total_points, dim)
        points = jnp.stack(grids, axis=-1).reshape(-1, self.dim)
        
        return points
    
    def __iter__(self):
        """Make the sampler iterable."""
        return self
    
    def __next__(self):
        return self._grid_points
    
    def data_generation(self, key):
        return self._grid_points

@partial(jit, static_argnums=(0,))
def _las_resample_core(
    model, params, weights, samples, key, beta, tau, lower_bound, upper_bound
):
    def total_sampler_loss(coords):
        return jnp.sum(model.r_losses_pp(params, coords, weights))

    gradient_coords = grad(total_sampler_loss)(samples)

    scaler = jnp.sqrt(
        jnp.sum((gradient_coords + 1e-16) ** 2, axis=-1, keepdims=True)
    )
    gradient_coords = gradient_coords / scaler

    key, subkey = random.split(key)
    noise = random.normal(subkey, shape=samples.shape)

    samples = (
        samples
        + tau * gradient_coords
        + beta * jnp.sqrt(2.0 * tau) * noise
    )

    samples = jnp.clip(samples, lower_bound, upper_bound)

    return samples, key


@partial(jit, static_argnums=(0,))
def _r3_resample_core(model, params, weights, samples, fresh, key):
    losses = model.r_losses_pp(params, samples, weights)

    mask = losses > jnp.mean(losses)
    keep_count = jnp.sum(mask)

    n = samples.shape[0]
    idx = jnp.arange(n)

    # Put kept samples first, preserving their original order.
    order = jnp.argsort(jnp.where(mask, idx, n + idx))
    kept_first = jnp.take(samples, order, axis=0)

    fresh_idx = jnp.maximum(idx - keep_count, 0)
    refill = jnp.take(fresh, fresh_idx, axis=0)
    keep_selector = (idx < keep_count).reshape((n,) + (1,) * (samples.ndim - 1))
    samples = jnp.where(keep_selector, kept_first, refill)

    key, _ = random.split(key)
    return samples, key
      
class LASSampler:
    def __init__(
        self,
        base_sampler,
        beta=0.2,
        tau=0.002,
        seed: int = 1234,
    ):
        """
        JAX version of the PyTorch LAS sampler.

        Args:
            base_sample: Initial coordinates, shape (N, D)
            lower_bound: Lower domain bound, shape (D,) or scalar
            upper_bound: Upper domain bound, shape (D,) or scalar
            beta: Langevin noise coefficient
            tau: Step size
            L_iter: Number of LAS iterations
            seed: Random seed
        """
        self.base_sampler = base_sampler
        self.base_sample = next(iter(base_sampler))  # (n_devices, batch_per_device, dim)
        self.beta = beta
        self.tau = tau
        self.key = random.PRNGKey(seed)
        self.iter = 0

    
    def __iter__(self):
        """Make the sampler iterable."""
        return self
    
    def __next__(self):
        """
        Returns current sample and increments counter.
        Automatically triggers resampling when counter hits interval.
        
        Returns:
            Current sample points, shape (n_devices, batch_per_device, dim)
        """        
        return self.base_sample

    def resample(self, model, params, weights=None):
        self.base_sample, self.key = _las_resample_core(
            model,
            params,
            weights,
            self.base_sample,
            self.key,
            self.beta,
            self.tau,
            self.base_sampler.dom[:, 0],
            self.base_sampler.dom[:, 1],
        )

        self.iter += 1
        return self.base_sample

class R3Sampler:
    def __init__(
        self,
        base_sampler,
        seed: int = 1234,
    ):
        """
        R3 sampler matching the PyTorch implementation.

        Args:
            base_sample: Initial sample, shape (1, N, D)
            base_sampler: Sampler that can generate random points in-domain
            seed: Random seed
        """
        self.base_sample = next(base_sampler)  # (1, N, D)
        self.base_sampler = base_sampler
        self.key = random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        return self.base_sample

    def resample(self, model, params, weights=None):
        """
        Keep points with loss > mean(loss), refill the rest with fresh random points.

        Assumes:
            model.sampler_loss(params, batch, weights) returns shape (N,)
            for batch shape (1, N, D).

        Returns:
            Updated sample of shape (1, N, D)
        """
        samples = self.base_sample
        fresh = next(self.base_sampler)

        if fresh.shape[0] < samples.shape[0]:
            raise ValueError(
                f"base_sampler returned only {fresh.shape[0]} fresh points, "
                f"but {samples.shape[0]} may be needed"
            )

        self.base_sample, self.key = _r3_resample_core(
            model,
            params,
            weights,
            samples,
            fresh[: samples.shape[0], ...],
            self.key,
        )
        return self.base_sample

class vRBA:
    def __init__(
        self,
        eval_sampler,
        batch_size,
        max_iter,
        k=0.0,
        c=1.0,
        eta=0.01,
        gamma=0.999,
        max_RBA0=None,
        cap_RBA=20.0,
        seed: int = 1234,
    ):
        self.eval_sample = next(eval_sampler)
        self.batch_size = batch_size
        self.k = float(k)
        self.c = float(c)
        self.eta = float(eta)
        self.key = random.PRNGKey(seed)
        self.N_total = self.eval_sample.shape[0]

        if max_RBA0 is None:
            max_RBA0 = eta / (1.0 - gamma)
        self.max_RBA0 = float(max_RBA0)

        denom = 50_000.0 if max_iter > 50_000 else max_iter/2
        self.step_RBA = float(
            (cap_RBA - self.max_RBA0) / (max_iter / denom - 1.0)
        )
        self.step = 0

        self.lambdas = jnp.zeros((self.N_total,))

        # Placeholders. prime() or resample() will set these before the first
        # model.step ever sees them.
        self.train_idx = jnp.arange(self.batch_size)
        self.sampled = self.eval_sample[self.train_idx]
        self.lambdas_sampled = self.lambdas[self.train_idx]

    def __iter__(self):
        return self

    def __next__(self):
        return self.sampled

    def get_lambdas(self):
        return self.lambdas_sampled

    def prime(self, model, params, weights=None):
        """Call once before the training loop.
        After this, next(sampler) and get_lambdas() are valid for the first step."""
        self._draw_and_update(model, params, weights)

    def resample(self, model, params, weights=None):
        """Call after each model.step.
        After this, next(sampler) and get_lambdas() are valid for the next step."""
        self._draw_and_update(model, params, weights)
        return self.sampled

    # --- internals --------------------------------------------------------

    def _draw_and_update(self, model, params, weights):
        # (1) Draw the batch we're about to serve from current lambdas.
        self.train_idx, self.key = self._draw_indices(self.lambdas, self.key)
        self.sampled = self.eval_sample[self.train_idx]

        # (2) Compute residuals on that batch at current params.
        residuals = model.r_losses_pp(params, self.sampled, weights)

        # (3) Update lambdas at those indices; cache fresh values for the loss.
        self.lambdas, self.lambdas_sampled, self.step = self._apply_lambda_update(
            self.lambdas, self.train_idx, residuals, self.step
        )

    @partial(jit, static_argnums=(0,))
    def _apply_lambda_update(self, lambdas, train_idx, residuals, step):
        # residuals is the weighted squared-norm per point, shape (B,)
        r_mag = jnp.sqrt(lax.stop_gradient(residuals.reshape(-1)) + 1e-12)

        max_rba = self.max_RBA0 + (self.step_RBA * step) // 50000.0
        gamma = 1.0 - self.eta / max_rba

        q_it = r_mag / (jnp.max(r_mag) + 1e-12)

        lambdas_old = lambdas[train_idx]
        new_lambdas = gamma * lambdas_old + self.eta * q_it
        lambdas = lambdas.at[train_idx].set(new_lambdas)

        return lambdas, new_lambdas, step + 1

    @partial(jit, static_argnums=(0,))
    def _draw_indices(self, lambdas, key):
        lambdas_k = jnp.power(lambdas, self.k)
        lambdas_k = lambdas_k / (jnp.mean(lambdas_k) + 1e-12) + self.c
        probs = lambdas_k / (jnp.sum(lambdas_k) + 1e-12)

        key, subkey = random.split(key)
        train_idx = random.choice(
            subkey,
            a=self.N_total,
            shape=(self.batch_size,),
            replace=True,
            p=probs,
        )
        return train_idx, key
        
class RADSampler:
    """
    Residual-based Adaptive Distribution (RAD) sampler.

    Expected behavior:
    - next(sampler) returns the current sampled batch
    - sampler.resample(model, params, weights) updates the sampled batch

    Assumptions:
    - eval_sampler yields candidate samples with shape (1, N, d) or (N, d)
    - self.sampled is stored with shape (1, K, d) or (K, d)
    - model.sampler_loss(params, candidate_batch, weights) returns shape (N,)
      where candidate_batch has shape (N, d)
    """

    def __init__(
        self,
        eval_sampler,
        sampled,
        seed: int = 1234,
        k=1.0,
        c=0.0,
    ):
        self.eval_sampler = eval_sampler
        self.sampled = sampled
        self.key = random.PRNGKey(seed)
        self.k = k
        self.c = c

        if sampled.ndim == 3:
            self.sample_size = sampled.shape[1]
            self._return_with_leading_axis = True
        elif sampled.ndim == 2:
            self.sample_size = sampled.shape[0]
            self._return_with_leading_axis = False
        else:
            raise ValueError(
                f"`sampled` must have shape (1, K, d) or (K, d), got shape {sampled.shape}"
            )

    def __iter__(self):
        return self

    def __next__(self):
        return self.sampled

    @partial(jit, static_argnums=(0, 1))
    def _resample_core(self, model, params, candidate_samples, weights, key):
        """
        Heavy resampling kernel.

        Args:
            model: training model object; must provide sampler_loss(...)
            params: current model params
            candidate_samples: shape (1, N, d) or (N, d)
            weights: passed through to model.sampler_loss
            key: JAX PRNG key

        Returns:
            new_sampled: shape matched to self.sampled layout
            new_key
        """
        had_leading_axis = (candidate_samples.ndim == 3)
        cand = candidate_samples[0] if had_leading_axis else candidate_samples  # (N, d)

        # Residual-based pointwise losses on the candidate pool.
        losses = model.r_losses_pp(params, cand, weights)

        # Error weighting.
        loss_pow = jnp.power(losses, self.k)
        err_eq = loss_pow / (jnp.mean(loss_pow) + 1e-12) + self.c
        probs = err_eq / (jnp.sum(err_eq) + 1e-12)
        # Sanitize probabilities.
        n = probs.shape[0]
        probs = jnp.nan_to_num(
            probs,
            nan=1.0 / n,
            posinf=0.0,
            neginf=0.0,
        )
        probs = probs / (jnp.sum(probs) + 1e-12)

        # Sample indices.
        new_key, subkey = random.split(key)
        idx = random.choice(
            subkey,
            a=n,
            shape=(self.sample_size,),
            replace=True,
            p=probs,
        )

        # Gather selected points.
        new_sampled = jnp.take(cand, idx, axis=0)  # (K, d)

        # Restore expected output shape.
        if self._return_with_leading_axis:
            new_sampled = new_sampled[None, :, :]  # (1, K, d)

        return new_sampled, new_key

    def resample(self, model, params, weights=None):
        """
        Refresh the internal sampled batch using RAD.

        This fetches a fresh candidate pool from eval_sampler, scores it with
        model.sampler_loss(...), builds sampling probabilities, and resamples
        self.sample_size points with replacement.
        """
        candidate_samples = next(self.eval_sampler)

        new_sampled, new_key = self._resample_core(
            model,
            params,
            candidate_samples,
            weights,
            self.key,
        )

        self.sampled = new_sampled
        self.key = new_key
        return self.sampled
    


def get_sampler(config, dom):
    """Construct the DIA sampler specified by config.dia.type."""
    if config.dia.type == "uniform":
        return iter(UniformSampler(dom, config.dia.batch_size, config.seed))

    if config.dia.type == "grid":
        return GridSampler(dom, config.dia.batch_sizes)

    if config.dia.type == "rad":
        base_sample  = next(iter(UniformSampler(dom, config.dia.batch_size, config.seed)))
        eval_sampler = iter(UniformSampler(dom, config.dia.batch_size_eval, config.seed))
        return RADSampler(sampled=base_sample, eval_sampler=eval_sampler, seed=config.seed)

    if config.dia.type == "las":
        base_sampler = UniformSampler(dom, config.dia.batch_size, seed=config.seed)
        return LASSampler(base_sampler=base_sampler, beta=config.dia.beta, tau=config.dia.tau, seed=config.seed)

    if config.dia.type == "R3":
        base_sampler = iter(UniformSampler(dom, config.dia.batch_size, seed=config.seed))
        return R3Sampler(base_sampler=base_sampler, seed=config.seed)

    if config.dia.type == "vRBA":
        eval_sampler = iter(UniformSampler(dom, config.dia.batch_size_eval, seed=config.seed))
        first_order_steps = config.optim.first_order_steps
        if isinstance(first_order_steps, (list, tuple)):
            first_order_steps = first_order_steps[0]
        return vRBA(eval_sampler=eval_sampler, batch_size=config.dia.batch_size, max_iter=first_order_steps, seed=config.seed)

    raise ValueError(f"Unknown DIA type: {config.dia.type}")
