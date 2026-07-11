"""Harness lock: validate caching + patching on every model used in the study.

Run BEFORE any confirmatory experiment (see PREREGISTRATION.md). Checks, per model:

  0. reshape unit test (no model): the [L, H*D] <-> [L, H, D] head splice touches
     exactly the intended head slice — the core patch_head operation.
  1. cache shapes match the config; attention rows sum to 1.
  2. identity patch: splicing every head of every layer with clean == corrupted
     is a numerical no-op (validates the hook plumbing changes nothing by itself).
  3. FULL RECOVERY (the trivial case): replacing the entire encoder output of the
     corrupted run with the clean run's must reproduce the clean first-step
     logits exactly and give patching effect ~1.0.
  4. all-head patch: splice all layers x heads on a real minimal pair; reported
     for context (this does NOT need to be 1.0 — the residual stream still
     carries corrupted token embeddings past every attention layer).
  5. forecast sanity: the clean-run forecast is closer to the clean continuation
     than to the corrupted one.

    python verify_harness.py --device mps
    python verify_harness.py --device mps --models amazon/chronos-t5-mini ...
"""

from __future__ import annotations

import argparse
import time

import torch

from chronos_harness import (get_inner, load_pipeline, patch_encoder_output,
                             patch_heads, predict_value, run_with_cache)
from synthetic import period_pair

STUDY_MODELS = [
    "amazon/chronos-t5-mini",
    "amazon/chronos-t5-small",
    "amazon/chronos-t5-base",
    "amazon/chronos-t5-large",
]


def check_reshape_unit() -> None:
    """Pure-torch check of the head-splice reshape, independent of any model."""
    H, D, L = 8, 64, 10
    rng = torch.Generator().manual_seed(0)
    corr = torch.randn(1, L, H * D, generator=rng)
    clean = torch.randn(1, L, H * D, generator=rng)
    head = 3
    out4 = corr.view(1, L, H, D).clone()
    out4[:, :, head, :] = clean.view(1, L, H, D)[:, :, head, :]
    spliced = out4.view(1, L, H * D)
    for h in range(H):
        s, c = spliced[0, :, h * D:(h + 1) * D], corr[0, :, h * D:(h + 1) * D]
        cl = clean[0, :, h * D:(h + 1) * D]
        if h == head:
            assert torch.equal(s, cl), "patched head must equal the clean slice"
        else:
            assert torch.equal(s, c), f"head {h} must be untouched"
    print("  [0] reshape unit test: head splice touches exactly one head  OK")


def verify_model(model_id: str, device: str | None) -> dict:
    t0 = time.time()
    pipe = load_pipeline(model_id, device)
    cfg = get_inner(pipe).config
    print(f"\n{model_id}: {cfg.num_layers} layers x {cfg.num_heads} heads, "
          f"d_model={cfg.d_model}, d_kv={cfg.d_kv}  "
          f"(loaded in {time.time() - t0:.0f}s)")

    pair = period_pair(7, 12, length=140, noise=0.05, seed=0)
    all_heads = [(l, h) for l in range(cfg.num_layers)
                 for h in range(cfg.num_heads)]

    # [1] cache shapes + attention normalization
    cache = run_with_cache(pipe, pair.clean.values)
    L = len(cache.tokens)
    assert len(cache.attn) == cfg.num_layers and len(cache.resid) == cfg.num_layers
    assert cache.attn[0].shape == (cfg.num_heads, L, L)
    assert cache.resid[0].shape == (L, cfg.d_model)
    rowsums = cache.attn[0].sum(-1)
    assert torch.allclose(rowsums, torch.ones_like(rowsums), atol=1e-3)
    print(f"  [1] cache: {L} tokens, shapes + attention row-sums  OK")

    # [2] identity patch is a numerical no-op
    r_id = patch_heads(pipe, pair.corrupted.values, pair.corrupted.values,
                       all_heads)
    assert abs(r_id["yhat_patched"] - r_id["yhat_corr"]) < 1e-4, r_id
    print(f"  [2] identity patch (clean==corrupted, all heads): "
          f"|delta yhat| = {abs(r_id['yhat_patched'] - r_id['yhat_corr']):.2e}  OK")

    # [3] full recovery: encoder-output patch must hit effect ~1.0
    r_full = patch_encoder_output(pipe, pair.clean.values, pair.corrupted.values)
    assert r_full["logits_match"], "patched logits must equal clean logits"
    assert abs(r_full["effect"] - 1.0) < 0.02, r_full
    print(f"  [3] full recovery (encoder-output patch): effect = "
          f"{r_full['effect']:.4f}, logits match  OK")

    # [4] all-head patch on the real pair (context, not a pass/fail gate)
    r_all = patch_heads(pipe, pair.clean.values, pair.corrupted.values, all_heads)
    print(f"  [4] all {len(all_heads)} heads patched: effect = "
          f"{r_all['effect']:+.3f}  (embeddings stay corrupted; <1.0 expected)")

    # [4b] all heads + token embeddings = every encoder quantity is clean by
    # induction -> MUST fully recover. Sine pair: both periods complete integer
    # cycles at this length, so clean/corrupted scales match and the effect is
    # not scale-confounded.
    spair = period_pair(7, 12, length=168, kind="sine", noise=0.05, seed=0)
    r_full_heads = patch_heads(pipe, spair.clean.values, spair.corrupted.values,
                               all_heads, patch_embed=True)
    assert abs(r_full_heads["effect"] - 1.0) < 0.02, r_full_heads
    print(f"  [4b] all heads + embeddings patched: effect = "
          f"{r_full_heads['effect']:.4f}  (full recovery)  OK")

    # [5] forecast sanity
    yc = predict_value(pipe, pair.clean.values)
    d_clean = abs(yc - pair.clean.future[0])
    d_corr = abs(yc - pair.corrupted.future[0])
    assert d_clean < d_corr, (yc, pair.clean.future[0], pair.corrupted.future[0])
    print(f"  [5] forecast sanity: clean yhat {yc:+.3f} is closer to clean truth "
          f"({d_clean:.3f} < {d_corr:.3f})  OK")

    return dict(model=model_id, layers=cfg.num_layers, heads=cfg.num_heads,
                all_head_effect=float(r_all["effect"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--models", nargs="+", default=STUDY_MODELS)
    args = ap.parse_args()

    check_reshape_unit()
    results = [verify_model(m, args.device) for m in args.models]

    print("\nharness verified on:")
    for r in results:
        print(f"  {r['model']}: {r['layers']}L x {r['heads']}H, "
              f"all-head patch effect {r['all_head_effect']:+.3f}")
    print("verify_harness.py: all checks passed")


if __name__ == "__main__":
    main()
