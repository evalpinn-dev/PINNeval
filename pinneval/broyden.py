
from typing import Callable, Optional
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
from typing import NamedTuple, Optional

Array = jnp.ndarray



import jax
import jax.numpy as jnp
from jax import lax
from jax._src.scipy.optimize.line_search import line_search
from jax._src.scipy.optimize.line_search import line_search as _jax_single_wolfe_line_search
from jax._src.scipy.optimize.line_search import _LineSearchResults # Ensure correct import

from functools import partial

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)


def line_search_jax_with_fallback( # This is the 2-stage fallback discussed before
    fun: Callable, xk: jax.Array, pk: jax.Array,
    old_fval: jax.Array  = None, old_old_fval: jax.Array  = None, gfk: jax.Array  = None,
    c1_try1: float = 1e-4, c2_try1: float = 0.8, maxiter_try1: int = 10,
    c1_try2: float = 1e-4, c2_try2: float = 0.5, maxiter_try2: int = 20
) -> _LineSearchResults:
    results_try1 = _jax_single_wolfe_line_search(fun, xk, pk, old_fval, old_old_fval, gfk,
                                                 c1_try1, c2_try1, maxiter_try1)
    def true_branch_fb_internal_failed(_):
        results_try2 = _jax_single_wolfe_line_search(fun, xk, pk, old_fval, old_old_fval, gfk,
                                                     c1_try2, c2_try2, maxiter_try2)
        return results_try2
    def false_branch_fb_internal_succeeded(r1):
        return r1
    return jax.lax.cond(results_try1.failed, true_branch_fb_internal_failed, false_branch_fb_internal_succeeded, results_try1)

def line_search_normal_then_fallback(
    fun: Callable,
    xk: jax.Array,
    pk: jax.Array,
    old_fval: jax.Array  = None,
    old_old_fval: jax.Array  = None,
    gfk: jax.Array  = None,
    # Parameters for the "normal" initial attempt
    normal_c1: float = 1e-4,
    normal_c2: float = 0.9,
    normal_maxiter: int = 15,
    # Parameters to pass to line_search_jax_with_fallback (which has its own try1 & try2)
    fallback_c1_try1: float = 1e-4,
    fallback_c2_try1: float = 0.8,
    fallback_maxiter_try1: int = 10,
    fallback_c1_try2: float = 1e-4,
    fallback_c2_try2: float = 0.5,
    fallback_maxiter_try2: int = 20
) -> _LineSearchResults:

    # 1. Attempt "normal" line search
    results_normal = _jax_single_wolfe_line_search(
        fun, xk, pk, old_fval, old_old_fval, gfk,
        c1=normal_c1, c2=normal_c2, maxiter=normal_maxiter
    )

    # This function is called if results_normal.failed is True
    def true_branch_normal_failed(_results_normal_ignored):
        # Normal search failed, now use the line_search_jax_with_fallback
        results_full_fallback = line_search_jax_with_fallback(
            fun, xk, pk, old_fval, old_old_fval, gfk,
            c1_try1=fallback_c1_try1, c2_try1=fallback_c2_try1, maxiter_try1=fallback_maxiter_try1,
            c1_try2=fallback_c1_try2, c2_try2=fallback_c2_try2, maxiter_try2=fallback_maxiter_try2
        )
        return results_full_fallback

    # This function is called if results_normal.failed is False
    def false_branch_normal_succeeded(res_normal):
        return res_normal

    final_results = jax.lax.cond(
        results_normal.failed,
        true_branch_normal_failed,
        false_branch_normal_succeeded,
        results_normal
    )
    return final_results

class BroydenState(NamedTuple):
    count: Array
    Hk: Array          # (n, n)
    failed: Array       # bool

def _ensure_tuple_value_grads(value_fn: Callable, params):
    out = value_fn(params)
    if isinstance(out, tuple) and len(out) == 2:
        loss, grads_tree = out
        return loss, grads_tree
    loss = out
    grads_tree = jax.grad(lambda p: value_fn(p))(params)
    return loss, grads_tree

def _flatten_grads(grads_tree, unravel=None) -> Array:
    if isinstance(grads_tree, jnp.ndarray):
        if grads_tree.ndim == 0:
            raise TypeError("Received a 0-d array as gradients. "
                            "Ensure value_fn returns (loss, grads) with grads matching params, "
                            "or pass `grad` to opt.update that matches the params pytree.")
        return grads_tree.reshape(-1)
    g_flat, _ = ravel_pytree(grads_tree)
    return g_flat

def _ss_theta(name: str, ak: Array, bk: Array, rhokm: Array, thetakm: Array, thetakp: Array, ratio: Array, eps: float):
    if name == "SSBroyden1":
        thetakSR1 = 1.0 / (1.0 - bk + eps)
        thetakp_extra = jnp.maximum(jnp.array(1.0, thetakp.dtype), bk * (1.0 + ratio))
        theta = jnp.where(
            jnp.isclose(bk, 1.0),
            thetakm,
            jnp.where(
                bk > 1.0,
                jnp.maximum(thetakm, thetakSR1),
                jnp.minimum(thetakp_extra, thetakSR1),
            ),
        )
    elif name == "SSBroyden2":
        theta = jnp.maximum(thetakm, jnp.minimum(thetakp, (1.0 - bk) / (bk + eps)))
    elif name == "SSBroyden3":
        theta = thetakm
    else:
        raise ValueError(f"Unknown SSBroyden variant: {name}")
    return theta


def _direction(Hk: Array, gk: Array) -> Array:
    return -(Hk @ gk)



def broyden(name: str = "SSBroyden1",
            c1: float = 1e-4, c2: float = 0.9, alpha0: float = 1.0, tau: float = 0.5,
            eps: float = 1e-12) -> optax.GradientTransformationExtraArgs:
    """Self-Scaled Broyden with Optax LBFGS-like signature.
    Call:
        updates, state = opt.update(grads, state, params, value=loss, grad=grads, value_fn=value_fn)
    where value_fn(params) -> (loss, grads) *or* -> loss (grads are auto-diffed).
    """
    variant = name

    def init_fn(params):
        x0, _ = ravel_pytree(params)
        n = x0.size
        return BroydenState(
            count=jnp.zeros([], jnp.int32),
            Hk=jnp.eye(n, dtype=x0.dtype),
            failed=jnp.array(False)
        )

    def update_fn(grads, state: BroydenState, params=None, *, value=None, grad=None, value_fn: Optional[Callable]=None):
        if params is None:
            raise ValueError("broyden requires current params.")
        if value_fn is None:
            raise ValueError("broyden requires value_fn(params)->(loss, grads) or -> loss.")
        # Current loss/grad
        if value is None or grad is None:
            fk, g_tree = _ensure_tuple_value_grads(value_fn, params)
            gk = _flatten_grads(g_tree)
        else:
            fk = value
            gk = _flatten_grads(grad)
        xk, unravel = ravel_pytree(params)

        Hk = state.Hk
        # here new
        p_k = -_dot(Hk, gk)

        def fun_flat(x_flat):
            return value_fn(unravel(x_flat))   # returns scalar loss



        line_search_results = line_search_normal_then_fallback(
            fun_flat, 
            xk,
            p_k,
            old_fval=fk,
            old_old_fval=None,
            gfk=gk,
            # Normal attempt parameters
            normal_c1=c1,
            normal_c2=c2,
            normal_maxiter=25,
            # Fallback (which contains two tries) parameters
            fallback_c1_try1=c1,
            fallback_c2_try1=0.8,
            fallback_maxiter_try1=25,
            fallback_c1_try2=1e-4,
            fallback_c2_try2=0.5,
            fallback_maxiter_try2=25
        )
        
        s_k = line_search_results.a_k * p_k
        x_kp1 = xk + s_k
        f_kp1 = line_search_results.f_k
        g_kp1 = line_search_results.g_k
        y_k = g_kp1 - gk
        # --- MODIFIED rho_k CALCULATION ---
        yk_dot_sk = _dot(y_k, s_k)
        # Define a threshold for considering the denominator effectively zero
        small_denominator_threshold = 1e-32 

        # Condition: is yk_dot_sk too close to zero?
        pred_yk_dot_sk_is_problematic = jnp.abs(yk_dot_sk) < small_denominator_threshold

        def calculate_rho_k_normally(operand_yds):
            # Normal calculation: 1.0 / (y_k^T s_k)
            # Add a very small epsilon here too, just in case yds passed the threshold
            # but is still problematic for reciprocal on some hardware/dtype,
            # or make the threshold slightly larger.
            return 1.0 / (operand_yds + jnp.sign(operand_yds) * 1e-20)


        def handle_problematic_rho_k_denominator(operand_yds):
            # Fallback value, mimicking SciPy's behavior for BFGS
            return jnp.array(1000.0, dtype=operand_yds.dtype)

        # This is the new rho_k, calculated conditionally
        rho_k = jax.lax.cond(
            pred_yk_dot_sk_is_problematic,
            handle_problematic_rho_k_denominator,
            calculate_rho_k_normally,
            yk_dot_sk # Operand passed to the branches
        )
        # --- END OF MODIFIED rho_k CALCULATION ---

        # Store current state variables for convenience
        Hk = Hk
        yk = y_k
        sk = s_k
        rhok = rho_k
        alpha_k_linesearch = line_search_results.a_k # Step length from line search
        gfk_old = gk # Gradient at x_k (before line search)
        d = xk.shape[0] # Dimension

        # --- Conditional Hessian Update ---

        eps_div = 1e-32 

        Hkyk = _dot(Hk, yk)
        ykHkyk = _dot(yk, Hkyk)
        
        # hk and bk calculations
        hk = ykHkyk * rhok
        bk = -alpha_k_linesearch * rhok * _dot(sk, gfk_old)
        ak = bk * hk - 1.0

        # Intermediate terms for thetak
        abs_ak = jnp.abs(ak)
        # Add epsilon to denominator for sqrt_term
        sqrt_term = jnp.sqrt(abs_ak / (1.0 + abs_ak + eps_div)) 
        rhokm_val = hk * (1.0 - sqrt_term)
        # If hk is large, rhokm_val could be > 1, ensure it's capped by 1
        # Or, if hk is negative (ykHkyk and rhok different signs), rhokm_val could be negative.
        # The original `min(1, ...)` suggests rhokm should not exceed 1.
        # Let's ensure it's also non-negative if that's implied by theory, though original min(1,val) doesn't enforce it.
        # For now, directly translate:
        rhokm = jnp.minimum(1.0, rhokm_val) 

        thetakm_den = ak + jnp.sign(ak) * eps_div # Avoid division by exact zero
        thetakm = (rhokm - 1.0) / thetakm_den
        
        thetakp_den = rhokm + eps_div # Avoid division by zero if rhokm is zero
        thetakp = 1.0 / thetakp_den

        bk_safe_den = bk + jnp.sign(bk) * eps_div # Avoid division by exact zero for term_for_thetak
        term_for_thetak = (1.0 - bk) / bk_safe_den
        thetak = jnp.maximum(thetakm, jnp.minimum(thetakp, term_for_thetak))

        # Tauk calculation (depends on initial_scale and current iteration)
        is_initial_step_and_scale_cond = False & (state.count == 0) & \
                                        jnp.allclose(Hk, jnp.eye(d, dtype=Hk.dtype), atol=1e-6) # Check if Hk is identity

        tauk_A_den = (1.0 + ak * thetak + eps_div)
        tauk_A = hk / tauk_A_den # tauk for initial scaling case

        rhokk_den = bk + jnp.sign(bk) * eps_div
        rhokk = jnp.minimum(1.0, 1.0 / rhokk_den) # if bk is very small, 1/bk is large
        
        sigmak = 1.0 + thetak * ak
        
        # Handle d=1 case for sigmaknm1_exp to avoid division by zero
        sigmaknm1_exp = lax.cond(d == 1,
                                lambda _: 0.0, # Effectively makes sigmaknm1 = 1 if d=1 (abs(sigmak)**0)
                                lambda _: 1.0 / (1.0 - d),
                                None)
        sigmaknm1 = jnp.abs(sigmak) ** sigmaknm1_exp

        tauk_B_cond = thetak <= 0.0
        tauk_B_true = jnp.minimum(rhokk * sigmaknm1, sigmak)
        tauk_B_false_den = thetak + eps_div
        tauk_B_false = rhokk * jnp.minimum(sigmaknm1, 1.0 / tauk_B_false_den)
        tauk_B = jnp.where(tauk_B_cond, tauk_B_true, tauk_B_false)

        tauk = jnp.where(is_initial_step_and_scale_cond, tauk_A, tauk_B)
        tauk_safe = tauk + jnp.sign(tauk) * eps_div # Ensure tauk is not zero for division

        # vk and phik calculations
        ykHkyk_safe = ykHkyk + jnp.sign(ykHkyk) * eps_div
        vk = sk * rhok - Hkyk / ykHkyk_safe
        
        phik_den = (1.0 + ak * thetak + eps_div)
        phik = (1.0 - thetak) / phik_den

        # SSBroyden2 Hessian update
        term1_H_update = Hk
        term2_H_update_num = _einsum('i,j->ij', Hkyk, Hkyk) # Outer product: Hkyk @ Hkyk.T
        term2_H_update = term2_H_update_num / ykHkyk_safe
        
        term3_H_update_vk_outer = _einsum('i,j->ij', vk, vk) # Outer product: vk @ vk.T
        term3_H_update = phik * ykHkyk * term3_H_update_vk_outer
        
        H_numerator = term1_H_update - term2_H_update + term3_H_update
        H_scaled = H_numerator / tauk_safe # Divide by tauk
        
        term4_H_update_sk_outer = _einsum('i,j->ij', sk, sk) # Outer product: sk @ sk.T
        term4_H_update = term4_H_update_sk_outer * rhok
        
        H_kp1 = H_scaled + term4_H_update


        # Safeguard against non-finite values (already in JAX's bfgs.py)
        H_kp1 = jnp.where(jnp.isfinite(rho_k), H_kp1, state.Hk)
        H_kp1 = jnp.where(jnp.isfinite(rho_k), H_kp1, state.Hk)
        converged = jnp.linalg.norm(g_kp1, ord=jnp.inf) < 1e-5

        

        updates = unravel(sk)
        new_state = BroydenState(
            count=optax.safe_int32_increment(state.count),
            Hk=H_kp1,
            failed=False
        )
        return updates, new_state


    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
