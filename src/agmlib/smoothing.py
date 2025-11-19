from __future__ import annotations

from typing import Any, Dict, Tuple, TYPE_CHECKING, Optional

import math
import numpy as np

from .config import AGMConfigEntity

if TYPE_CHECKING:  # Avoid hard import at module load
    import torch


def _require_torch() -> "torch":
    import importlib

    torch = importlib.import_module("torch")
    assert torch is not None, "torch import failed"
    return torch


def compute_knn(z: Any, k: int, block_size: Optional[int] = None) -> Any:
    """Compute batch-local kNN indices using torch.cdist + topk.

    Args:
        z: Latent tensor of shape (batch, dim).
        k: Number of neighbors to select per row (excluding self).
        block_size: Optional row-block size for memory-efficient computation.

    Returns:
        Tensor of indices with shape (batch, k), dtype long.
    """
    torch = _require_torch()
    assert z is not None, "z required"
    assert isinstance(k, int) and k >= 1, "k must be >= 1"
    z_t = z
    assert z_t.ndim == 2, f"expected 2D (batch, dim), got {z_t.shape}"
    batch_size = z_t.shape[0]
    assert batch_size >= k + 1, "batch_size must be >= k+1 to exclude self"

    # If no block size provided, compute full distance matrix
    if block_size is None or block_size >= batch_size:
        dists = torch.cdist(z_t, z_t, p=2)
        inf = torch.tensor(float("inf"), device=z_t.device, dtype=dists.dtype)
        dists.fill_diagonal_(inf)
        knn_dists, knn_idx = torch.topk(dists, k=k, largest=False, dim=1)
        assert torch.isfinite(knn_dists).all(), "non-finite distances in kNN"
        return knn_idx

    # Blockwise over rows for memory efficiency
    assert isinstance(block_size, int) and block_size >= 1, "block_size must be >= 1"
    device = z_t.device
    dtype = z_t.dtype
    inf = torch.tensor(float("inf"), device=device, dtype=dtype)
    knn_idx_out = torch.empty((batch_size, k), dtype=torch.long, device=device)

    for start in range(0, batch_size, block_size):
        end = min(start + block_size, batch_size)
        rows = z_t[start:end]  # (b, dim)
        # Distances to all points
        dists_block = torch.cdist(rows, z_t, p=2)  # (b, N)
        # Mask self distances in this block
        bsz = end - start
        if bsz > 0:
            idx_block = torch.arange(start, end, device=device)
            row_idx = torch.arange(0, bsz, device=device)
            dists_block[row_idx, idx_block] = inf
        assert torch.isfinite(dists_block).all(), "non-finite distances in kNN (block)"
        _, idxs = torch.topk(dists_block, k=k, largest=False, dim=1)
        knn_idx_out[start:end] = idxs

    return knn_idx_out


def adaptive_sigmas(z: Any, knn: Any, config: AGMConfigEntity) -> Any:
    """Compute adaptive per-row sigma from kth neighbor distance with clamping.

    Returns tensor shape (batch, 1).
    """
    torch = _require_torch()
    assert z is not None and knn is not None and config is not None, "inputs required"
    z_t = z
    assert z_t.ndim == 2, f"expected (batch, dim), got {z_t.shape}"
    batch, _ = z_t.shape
    assert knn.shape == (batch, config.kernel_smoothing.k), "knn shape mismatch"

    # Compute kth neighbor distance per row
    # Gather vectors of kth neighbors
    idx_k = knn[:, -1]
    z_k = z_t[idx_k]
    distances = (z_t - z_k).pow(2).sum(dim=1).sqrt()  # (batch,)
    sigmas = distances.clamp(min=config.kernel_smoothing.sigma_min, max=config.kernel_smoothing.sigma_max)
    sigmas = sigmas.unsqueeze(1)  # (batch, 1)
    assert torch.isfinite(sigmas).all(), "non-finite sigma values"
    return sigmas


def compute_kernel_weights(z: Any, knn: Any, sigmas: Any) -> Any:
    """Gaussian weights per row normalized over neighbors.

    Returns tensor shape (batch, k).
    """
    torch = _require_torch()
    z_t = z
    batch, dim = z_t.shape
    k = knn.shape[1]
    # Gather neighbors for each row
    neighbors = z_t[knn]  # (batch, k, dim)
    diffs = neighbors - z_t.unsqueeze(1)  # (batch, k, dim)
    sq_dist = (diffs * diffs).sum(dim=2)  # (batch, k)
    denom = 2.0 * (sigmas ** 2).squeeze(1)  # (batch,)
    denom = torch.clamp(denom, min=1e-12)
    logits = -sq_dist / denom.unsqueeze(1)  # (batch, k)
    w = torch.exp(logits)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    w = w / w_sum
    assert torch.isfinite(w).all(), "non-finite kernel weights"
    # Rows should sum to ~1
    row_sums = w.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "weight rows not normalized"
    return w


def kernel_consensus(h: Any, z: Any, knn: Any, sigmas: Any) -> Any:
    """Apply kernel consensus operator G[h]. Supports h shape (batch,) or (batch, D)."""
    torch = _require_torch()
    z_t = z
    batch, _ = z_t.shape
    w = compute_kernel_weights(z_t, knn, sigmas)  # (batch, k)
    # Gather h neighbors
    h_t = h
    if h_t.ndim == 1:
        h_neighbors = h_t[knn]  # (batch, k)
        G = (w * h_neighbors).sum(dim=1)  # (batch,)
    elif h_t.ndim == 2:
        h_neighbors = h_t[knn]  # (batch, k, D)
        w_exp = w.unsqueeze(2)  # (batch, k, 1)
        G = (w_exp * h_neighbors).sum(dim=1)  # (batch, D)
    else:
        raise AssertionError(f"Unsupported h ndim: {h_t.ndim}")
    assert torch.isfinite(G).all(), "non-finite consensus output"
    return G


def trust_region_update(
    h: Any,
    G: Any,
    *,
    alpha_step: float,
    tau: float,
    dim_exponent: float,
    noise_std: float = 0.0,
) -> Any:
    """Compute Δh with trust-region clipping and dimension-aware step.

    Δh = α_step * d^(−dim_exponent) * (G − h); clip per-row ||Δh|| ≤ τ.
    """
    torch = _require_torch()
    assert alpha_step >= 0.0, "alpha_step must be ≥ 0"
    assert tau >= 0.0, "tau must be ≥ 0"
    assert dim_exponent >= 0.0, "dim_exponent must be ≥ 0"

    h_t, G_t = h, G
    assert h_t.shape == G_t.shape, "h and G must have same shape"

    if h_t.ndim == 1:
        d = 1
        delta = alpha_step * (d ** (-dim_exponent)) * (G_t - h_t)
        if noise_std > 0.0:
            delta = delta + torch.randn_like(delta) * float(noise_std)
        norm = delta.abs()
        scale = torch.clamp(tau / (norm + 1e-12), max=1.0)
        delta_scaled = delta * scale
        updated = h_t + delta_scaled
        assert torch.isfinite(updated).all(), "non-finite updated h"
        return updated

    if h_t.ndim == 2:
        d = h_t.shape[1]
        delta = alpha_step * (float(d) ** (-dim_exponent)) * (G_t - h_t)
        if noise_std > 0.0:
            delta = delta + torch.randn_like(delta) * float(noise_std)
        norms = torch.linalg.vector_norm(delta, ord=2, dim=1, keepdim=True)
        scales = torch.clamp(tau / (norms + 1e-12), max=1.0)
        delta_scaled = delta * scales
        updated = h_t + delta_scaled
        assert torch.isfinite(updated).all(), "non-finite updated h"
        # Verify clipping invariant
        clipped_norms = torch.linalg.vector_norm(delta_scaled, ord=2, dim=1)
        assert torch.all(clipped_norms <= tau + 1e-6), "Δh clipping invariant violated"
        return updated

    raise AssertionError(f"Unsupported h ndim: {h_t.ndim}")


def should_apply_smoothing(td_error_stats: Dict[str, float], local_uncertainty: Any, *, config: AGMConfigEntity) -> bool:
    """Gate smoothing by uncertainty/variance using config thresholds only.

    This is a placeholder policy. Replace thresholds with config-driven values
    as they are introduced. No hard-coded hidden constants beyond simple guards.
    """
    assert td_error_stats is not None and config is not None, "inputs required"
    td_var = float(td_error_stats.get("var", 0.0))
    if not config.kernel_smoothing.enabled:
        return False
    # Apply when TD variance exceeds configured minimum
    return td_var >= config.kernel_smoothing.td_var_min


def schedule_alpha_step(step: int, *, config: AGMConfigEntity) -> float:
    """Schedule alpha_step according to config (cosine/constant)."""
    assert step >= 0, "step must be non-negative"
    init = config.kernel_smoothing.alpha_step_init
    if config.kernel_smoothing.alpha_step_schedule == "constant":
        return init
    # cosine decay from init to ~0 over decay_steps
    decay_steps = max(1, config.kernel_smoothing.alpha_step_decay_steps)
    ratio = min(1.0, step / float(decay_steps))
    # Cosine from 1 -> 0
    weight = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return float(init * weight)


def featurewise_scale_preservation(before: Any, after: Any) -> Any:
    """Rescale features to preserve per-dimension std of `before`.

    Computes scale factors σ_orig/σ_new and rescales `after` accordingly.
    """
    torch = _require_torch()
    assert before.shape == after.shape, "before/after shape mismatch"
    if after.ndim == 1:
        # Scalar case: nothing to do
        return after
    # Compute per-dimension std with eps guard
    eps = 1e-12
    orig_std = torch.std(before, dim=0, unbiased=False).clamp(min=eps)
    new_std = torch.std(after, dim=0, unbiased=False).clamp(min=eps)
    factors = orig_std / new_std
    return after * factors


def average_disagreement(h: Any, G: Any) -> float:
    """Compute mean L2 disagreement ||G - h|| per row as a gating signal."""
    torch = _require_torch()
    diff = G - h
    if diff.ndim == 1:
        val = diff.abs().mean()
        return float(val.item())
    norms = torch.linalg.vector_norm(diff, ord=2, dim=1)
    return float(norms.mean().item())


def should_apply_smoothing_decision(*, td_var: float, disagreement: float, config: AGMConfigEntity) -> bool:
    """Gate smoothing using configured thresholds for TD variance and disagreement.

    Applies iff both thresholds are satisfied (AND logic) to be conservative.
    """
    assert config is not None, "config required"
    if not config.kernel_smoothing.enabled:
        return False
    if td_var < config.kernel_smoothing.td_var_min:
        return False
    if disagreement < config.kernel_smoothing.disagreement_min:
        return False
    return True


def free_energy(h: Any, delta: Any, *, z: Any, knn: Any, sigmas: Any, temperature: float = 1.0) -> float:
    """Compute F[H] = U[H] - T·S[H] as a monitoring metric.

    U = 1/2 * ||h - delta||^2
    S = -sum_i sum_j p(j|i) log p(j|i), with p(j|i) derived from Gaussian weights.
    """
    torch = _require_torch()
    # Energy term
    diff = h - delta
    if diff.ndim == 1:
        U = 0.5 * torch.sum(diff * diff)
    else:
        U = 0.5 * torch.sum(torch.linalg.vector_norm(diff, ord=2, dim=1) ** 2)
    # Entropy term from normalized kernel weights over neighbors
    w = compute_kernel_weights(z, knn, sigmas)  # (batch, k)
    eps = 1e-12
    S = -torch.sum(w * torch.log(w + eps))
    F = U - float(temperature) * S
    return float(F.item())


def apply_batch_smoothing(
    *,
    h: Any,
    z: Any,
    step: int,
    td_var: float,
    config: AGMConfigEntity,
) -> Tuple[Any, Dict[str, Any]]:
    """End-to-end batch smoothing flow with diagnostics.

    Returns updated h (same shape as input) and diagnostics dict with keys:
    - gate: bool
    - avg_disagreement: float
    - sigma_stats: {min, mean, max}
    - delta_h_stats: {mean, max}
    - sigmas: Tensor (for optional histogramming)
    - delta_h: Tensor (for optional histogramming)
    """
    torch = _require_torch()
    assert h is not None and z is not None and config is not None, "inputs required"
    knn = compute_knn(z, config.kernel_smoothing.k, block_size=getattr(config.kernel_smoothing, "knn_block_size", None))
    sigmas = adaptive_sigmas(z, knn, config)
    G = kernel_consensus(h, z, knn, sigmas)
    avg_dis = average_disagreement(h, G)
    gate = should_apply_smoothing_decision(td_var=float(td_var), disagreement=avg_dis, config=config)
    alpha = schedule_alpha_step(step, config=config)
    if gate:
        h_updated = trust_region_update(
            h,
            G,
            alpha_step=alpha,
            tau=config.kernel_smoothing.trust_region_tau,
            dim_exponent=config.kernel_smoothing.dim_exponent,
            noise_std=float(getattr(config.kernel_smoothing, "noise_std", 0.0)),
        )
    else:
        h_updated = h

    # Optional feature-wise scale preservation
    if config.kernel_smoothing.scale_preservation:
        h_updated = featurewise_scale_preservation(h, h_updated)

    # Diagnostics
    sigma_stats = {
        "min": float(sigmas.min().item()),
        "mean": float(sigmas.mean().item()),
        "max": float(sigmas.max().item()),
    }
    delta = h_updated - h
    if delta.ndim == 1:
        delta_norm = delta.abs()
    else:
        delta_norm = torch.linalg.vector_norm(delta, ord=2, dim=1)
    delta_h_stats = {
        "mean": float(delta_norm.mean().item()),
        "max": float(delta_norm.max().item()),
    }

    diag: Dict[str, Any] = {
        "gate": gate,
        "avg_disagreement": avg_dis,
        "sigma_stats": sigma_stats,
        "delta_h_stats": delta_h_stats,
        "sigmas": sigmas,
        "delta_h": delta_norm,
    }
    return h_updated, diag

