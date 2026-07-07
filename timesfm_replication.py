"""Cross-family replication (prereg: secondary, best-effort): the H1 pipeline
on TimesFM 2.5 200M (torch) — a genuinely different family from Chronos-T5:
decoder-only, causal attention over 32-step PATCH tokens, continuous output
head (no value-bin vocabulary).

EXPLORATORY seeds (<100) only. Adaptations fixed before running:
  * tokens are 32-step patches -> periods {64, 96} = patch-lags {2, 3};
    context 1024 = 32 tokens; pattern seasonality repeats exactly in token
    space (periods are patch multiples).
  * CAUSAL uniform null: for query i the null lag-window mass is
    (2*tol+1)/(i+1); ratio = mean(mass) / mean(null), tol=1 in token units.
  * candidate rule as Stage 1: > 3x null at BOTH lags AND > 2x the
    phase-scrambled control at those lags.
  * causal: group patch (pre-`out` head splice, all positions) on
    period-64<->96 pairs; recovery of the t+1 point forecast; 10 seeds;
    size-matched random control heads.

Run with the timesfm venv (NOT the chronos venv):
    ~/.venvs/lag-llama/bin/python timesfm_replication.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from attention_analysis import bootstrap_ci, patching_effect
from synthetic import period_pair, seasonal

CONTEXT = 1024
PATCH = 32
PERIODS = (64, 96)          # -> patch lags 2, 3
NOISE = 0.05
DESC_SEEDS = range(5)
CAUSAL_SEEDS = range(10)
TOL = 1


def load():
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", torch_compile=False)
    model.compile(timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=128,
                                         normalize_inputs=True))
    inner = model.model
    inner.eval()
    attns = [m for _, m in inner.named_modules()
             if type(m).__name__ == "MultiHeadAttention"]
    return model, inner, attns


def stash_wrapper(store, layer):
    def fn(query, key, value, mask=None):
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        scores = q @ k.transpose(-1, -2)          # model uses scale=1.0
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask
        probs = scores.softmax(-1)
        if layer not in store or probs.shape[-1] > store[layer].shape[-1]:
            store[layer] = probs.detach()[0].float()   # [H, T, T]
        return (probs @ v).permute(0, 2, 1, 3)
    return fn


def forecast(model, values) -> float:
    point, _ = model.forecast(horizon=1, inputs=[np.asarray(values,
                                                            np.float32)])
    return float(point[0, 0])


def cached_attention(model, attns, values) -> dict:
    store: dict = {}
    originals = [a.attention_fn for a in attns]
    for i, a in enumerate(attns):
        a.attention_fn = stash_wrapper(store, i)
    try:
        forecast(model, values)
    finally:
        for a, o in zip(attns, originals):
            a.attention_fn = o
    return store


def causal_ratio(attn: torch.Tensor, lag: int, tol: int = TOL) -> np.ndarray:
    """attn: [H, T, T] causal. Ratio of lag-window mass to the causal uniform
    null, per head."""
    H, T, _ = attn.shape
    masses, nulls = [], []
    for i in range(lag + tol, T):
        lo, hi = i - lag - tol, i - lag + tol + 1
        masses.append(attn[:, i, lo:hi].sum(-1).numpy())
        nulls.append((2 * tol + 1) / (i + 1))
    return np.mean(masses, axis=0) / np.mean(nulls)


def patch_group(model, attns, clean_values, corr_values, heads) -> dict:
    """Splice head group clean->corrupted at each attention's pre-`out` input
    (heads still separate there), all positions. Value-space recovery of the
    t+1 point forecast."""
    n_heads = attns[0].num_heads
    d_head = attns[0].in_features // n_heads
    by_layer: dict = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)

    y_clean = forecast(model, clean_values)
    y_corr = forecast(model, corr_values)

    captured: dict = {}
    handles = [attns[l].out.register_forward_pre_hook(
        (lambda mod, args, _l=l:
         captured.__setitem__(_l, args[0].detach().clone())
         if _l not in captured or args[0].shape[1] >= captured[_l].shape[1]
         else None))
        for l in by_layer]
    try:
        forecast(model, clean_values)
    finally:
        for h in handles:
            h.remove()

    def make_splice(clean_pre, hs):
        def hook(mod, args):
            out = args[0]
            if out.shape != clean_pre.shape:
                return None
            b, T, _ = out.shape
            out4 = out.view(b, T, n_heads, d_head).clone()
            c4 = clean_pre.view(b, T, n_heads, d_head)
            for h in hs:
                out4[:, :, h, :] = c4[:, :, h, :]
            return (out4.view(b, T, n_heads * d_head),)
        return hook

    handles = [attns[l].out.register_forward_pre_hook(
        make_splice(captured[l], tuple(hs))) for l, hs in by_layer.items()]
    try:
        y_patched = forecast(model, corr_values)
    finally:
        for h in handles:
            h.remove()
    return dict(y_clean=y_clean, y_corr=y_corr, y_patched=y_patched,
                effect=patching_effect(y_clean, y_corr, y_patched))


def main() -> None:
    model, inner, attns = load()
    n_layers, n_heads = len(attns), attns[0].num_heads
    print(f"TimesFM 2.5 200M torch: {n_layers}L x {n_heads}H, patch={PATCH}, "
          f"context={CONTEXT} ({CONTEXT // PATCH} tokens)")

    # sanity: the stash wrapper must not change the forecast
    s0 = seasonal(period=64, length=CONTEXT, kind="pattern", noise=NOISE,
                  seed=0)
    y_plain = forecast(model, s0.values)
    _ = cached_attention(model, attns, s0.values)
    y_wrapped = forecast(model, s0.values)
    assert abs(y_plain - y_wrapped) < 1e-4, (y_plain, y_wrapped)
    print(f"wrapper sanity: forecast unchanged ({y_plain:+.4f})  OK")

    # ---- descriptive: ratios vs causal null and scrambled control ----
    real, scram = {}, {}
    for period in PERIODS:
        lag = period // PATCH
        rs, ss = [], []
        for seed in DESC_SEEDS:
            s = seasonal(period=period, length=CONTEXT, kind="pattern",
                         noise=NOISE, seed=seed)
            attn = cached_attention(model, attns, s.values)
            rs.append(np.stack([causal_ratio(attn[l], lag)
                                for l in range(n_layers)]))
            perm = np.random.default_rng(seed + 50_000).permutation(s.values)
            attn = cached_attention(model, attns, perm)
            ss.append(np.stack([causal_ratio(attn[l], lag)
                                for l in range(n_layers)]))
        real[period] = np.mean(rs, axis=0)
        scram[period] = np.mean(ss, axis=0)

    candidates = [(l, h) for l in range(n_layers) for h in range(n_heads)
                  if all(real[p][l, h] > 3.0
                         and real[p][l, h] > 2.0 * scram[p][l, h]
                         for p in PERIODS)]
    gm = np.exp(np.mean([np.log(np.maximum(real[p], 1e-9)) for p in PERIODS],
                        axis=0))
    order = np.argsort(gm, axis=None)[::-1][:10]
    print("\ntop heads: ratio vs CAUSAL null @ P64(lag2) | P96(lag3) "
          "(scrambled in parens)")
    for f in order:
        l, h = np.unravel_index(f, gm.shape)
        mark = "  CANDIDATE" if (int(l), int(h)) in candidates else ""
        print(f"  L{l}H{h:<3} "
              f"{real[64][l, h]:6.2f} ({scram[64][l, h]:5.2f}) | "
              f"{real[96][l, h]:6.2f} ({scram[96][l, h]:5.2f})  "
              f"gm={gm[l, h]:5.2f}{mark}")
    print(f"{len(candidates)} candidate(s): "
          f"{['L%dH%d' % c for c in candidates]}")

    # ---- causal: group patch on period-64<->96 pairs ----
    blob = dict(model="google/timesfm-2.5-200m-pytorch", layers=n_layers,
                heads=n_heads, periods=list(PERIODS), context=CONTEXT,
                real={str(p): real[p].tolist() for p in PERIODS},
                scrambled={str(p): scram[p].tolist() for p in PERIODS},
                candidates=[list(c) for c in candidates])
    if candidates:
        rng = np.random.default_rng(7)
        pool = [(l, h) for l in range(n_layers) for h in range(n_heads)
                if (l, h) not in candidates]
        control = [pool[i] for i in rng.choice(len(pool), len(candidates),
                                               replace=False)]
        g_eff, c_eff = [], []
        for seed in CAUSAL_SEEDS:
            pair = period_pair(64, 96, length=CONTEXT, noise=NOISE, seed=seed)
            g_eff.append(patch_group(model, attns, pair.clean.values,
                                     pair.corrupted.values,
                                     candidates)["effect"])
            c_eff.append(patch_group(model, attns, pair.clean.values,
                                     pair.corrupted.values, control)["effect"])
        gm_, glo, ghi = bootstrap_ci(g_eff)
        cm_, clo, chi = bootstrap_ci(c_eff)
        print(f"\ncausal group patch (t+1 point forecast recovery, "
              f"{len(list(CAUSAL_SEEDS))} seeds):")
        print(f"  candidates ({len(candidates)}): {gm_:+.3f} "
              f"[{glo:+.3f},{ghi:+.3f}]")
        print(f"  control    ({len(control)}): {cm_:+.3f} "
              f"[{clo:+.3f},{chi:+.3f}]")
        blob.update(group_effects=g_eff, control_effects=c_eff,
                    control=[list(c) for c in control])
    else:
        print("\nno candidates -> descriptive replication FAILS on TimesFM "
              "(report as-is)")

    out = Path("results/timesfm-replication.json")
    out.write_text(json.dumps(blob, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
