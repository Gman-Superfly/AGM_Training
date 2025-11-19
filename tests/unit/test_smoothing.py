import importlib
import math
import pytest


torch_spec = importlib.util.find_spec("torch")
torch = None if torch_spec is None else importlib.import_module("torch")


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_kernel_weights_sum_to_one():
    from agmlib.smoothing import compute_knn, adaptive_sigmas, compute_kernel_weights
    from agmlib.config import AGMConfigEntity

    import torch as T

    z = T.randn(32, 8)
    cfg = AGMConfigEntity(
        replay={},
        kernel_smoothing={"k": 4},
        early_stopping={},
        distributed={},
    )
    knn = compute_knn(z, cfg.kernel_smoothing.k)
    sigmas = adaptive_sigmas(z, knn, cfg)
    w = compute_kernel_weights(z, knn, sigmas)
    row_sums = w.sum(dim=1)
    assert T.allclose(row_sums, T.ones_like(row_sums), atol=1e-5)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_trust_region_clip():
    from agmlib.smoothing import trust_region_update
    import torch as T

    h = T.zeros(10, 4)
    G = T.ones(10, 4)
    updated = trust_region_update(h, G, alpha_step=1.0, tau=0.1, dim_exponent=0.0)
    deltas = updated - h
    norms = T.linalg.vector_norm(deltas, ord=2, dim=1)
    assert T.all(norms <= 0.100001)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_alpha_schedule_and_gating_and_sigma_bounds_and_scale_preservation():
    from agmlib.smoothing import (
        schedule_alpha_step,
        compute_knn,
        adaptive_sigmas,
        kernel_consensus,
        trust_region_update,
        featurewise_scale_preservation,
        average_disagreement,
        should_apply_smoothing_decision,
        apply_batch_smoothing,
    )
    from agmlib.config import AGMConfigEntity
    import torch as T

    # Config with thresholds and scale preservation on
    cfg = AGMConfigEntity(
        replay={},
        kernel_smoothing={
            "k": 4,
            "sigma_min": 1e-3,
            "sigma_max": 1.0,
            "alpha_step_init": 0.2,
            "alpha_step_schedule": "cosine",
            "alpha_step_decay_steps": 10,
            "trust_region_tau": 0.05,
            "dim_exponent": 0.0,
            "scale_preservation": True,
            "td_var_min": 0.0,
            "disagreement_min": 0.0,
        },
        early_stopping={},
        distributed={},
    )

    T.manual_seed(0)
    z = T.randn(32, 6)
    h = T.randn(32, 3)

    # Alpha schedule monotone non-increasing over decay window
    a0 = schedule_alpha_step(0, config=cfg)
    a5 = schedule_alpha_step(5, config=cfg)
    a10 = schedule_alpha_step(10, config=cfg)
    assert a0 >= a5 >= a10

    # Sigma bounds
    knn = compute_knn(z, cfg.kernel_smoothing.k)
    sigmas = adaptive_sigmas(z, knn, cfg)
    assert T.all(sigmas >= cfg.kernel_smoothing.sigma_min - 1e-9)
    assert T.all(sigmas <= cfg.kernel_smoothing.sigma_max + 1e-9)

    # Gating decision should allow when thresholds zero
    G = kernel_consensus(h, z, knn, sigmas)
    dis = average_disagreement(h, G)
    assert should_apply_smoothing_decision(td_var=0.0, disagreement=dis, config=cfg)

    # End-to-end batch smoothing with diagnostics and scale preservation
    h_upd, diag = apply_batch_smoothing(h=h, z=z, step=0, td_var=0.0, config=cfg)
    assert set(["gate", "avg_disagreement", "sigma_stats", "delta_h_stats", "sigmas", "delta_h"]).issubset(diag.keys())
    # Scale preservation: per-dimension std should be close to original
    orig_std = T.std(h, dim=0, unbiased=False)
    new_std = T.std(h_upd, dim=0, unbiased=False)
    assert T.allclose(orig_std, new_std, atol=1e-5, rtol=1e-3)
