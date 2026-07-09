"""Stage 5a — characterize the algorithm (chronos-t5-small, the confirmed
circuit; EXPLORATORY seeds).

(1) Is the period a fixed learned lag or detected from the input?
    Feed one sequence whose period changes mid-way (P=7 for the first half,
    P=12 after). For each confirmed head, track attention mass at lag-7 vs
    lag-12 as a function of query position. A fixed-lag head keeps its lag; an
    input-detecting head re-locks after the change — and the crossing point
    measures the re-lock latency in steps.

(2) Is the forecast slope a linear readout of the trend direction?
    Steer the H2-procedure direction by +-k over a grid and measure forecast
    displacement. (H2's confirmatory null predicts NO clean linear response —
    this quantifies it.)

    python mechanism.py --device mps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from attention_analysis import bootstrap_ci
from chronos_harness import (_first_step_logits, _expected_value,
                             add_direction_hooks, load_pipeline,
                             run_with_cache, tokenize)
from confirmatory_h1 import select_heads
from confirmatory_h2 import features, series_set, SELECTION_SEEDS
from synthetic import seasonal, trend

MODEL = "amazon/chronos-t5-small"
LENGTH = 168
CP = 84                    # period switches 7 -> 12 here
NOISE = 0.05
SEEDS = range(10)


def period_switch_series(seed: int) -> np.ndarray:
    a = seasonal(period=7, length=CP, kind="pattern", noise=NOISE, seed=seed)
    b = seasonal(period=12, length=LENGTH - CP, kind="pattern", noise=NOISE,
                 seed=seed + 500)
    return np.concatenate([a.values, b.values])


def lag_mass_profile(attn: torch.Tensor, lag: int, tol: int = 1) -> np.ndarray:
    """Per-query attention mass in the lag window, one [L] row per head."""
    H, L, _ = attn.shape
    prof = np.full((H, L), np.nan)
    for i in range(lag + tol, L):
        prof[:, i] = attn[:, i, i - lag - tol:i - lag + tol + 1].sum(-1).numpy()
    return prof


@torch.no_grad()
def yhat(pipe, values, hooks=None):
    try:
        ids, mask, scale = tokenize(pipe, values)
        s = float(scale[0])
        return _expected_value(pipe, _first_step_logits(pipe, ids, mask), s), s
    finally:
        for h in (hooks or []):
            h.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    pipe = load_pipeline(MODEL, args.device)
    heads, _ = select_heads(pipe)   # confirmed H1 group: L5H7, L4H1, L4H7
    print(f"{MODEL}: confirmed heads {['L%dH%d' % h for h in heads]}")

    # ---- (1) period re-lock ----
    profiles7, profiles12 = [], []
    for seed in SEEDS:
        cache = run_with_cache(pipe, period_switch_series(seed))
        p7 = np.mean([lag_mass_profile(cache.attn[l], 7)[h] for l, h in heads],
                     axis=0)
        p12 = np.mean([lag_mass_profile(cache.attn[l], 12)[h]
                       for l, h in heads], axis=0)
        profiles7.append(p7)
        profiles12.append(p12)
    p7 = np.nanmean(profiles7, axis=0)
    p12 = np.nanmean(profiles12, axis=0)

    early = slice(30, CP)                    # inside the P=7 regime
    late = slice(CP + 24, LENGTH)            # well after the switch
    e7, e12 = np.nanmean(p7[early]), np.nanmean(p12[early])
    l7, l12 = np.nanmean(p7[late]), np.nanmean(p12[late])
    print(f"\n(1) period switch 7->12 at t={CP} (mean over confirmed heads, "
          f"{len(list(SEEDS))} seeds):")
    print(f"    queries before switch: mass@lag7 {e7:.3f} vs mass@lag12 "
          f"{e12:.3f}  ({e7 / e12:.1f}x lag-7)")
    print(f"    queries after switch:  mass@lag7 {l7:.3f} vs mass@lag12 "
          f"{l12:.3f}  ({l12 / l7:.1f}x lag-12)")
    diff = p12 - p7
    post = np.arange(CP, LENGTH)
    crossing = next((int(i) for i in post
                     if not np.isnan(diff[i]) and diff[i] > 0
                     and np.all(diff[i:i + 5] > 0)), None)
    latency = None if crossing is None else crossing - CP
    print(f"    re-lock latency: lag-12 mass overtakes lag-7 at t={crossing} "
          f"({latency} steps = {None if latency is None else latency / 12:.1f} "
          f"new-period cycles after the switch)")
    verdict1 = (e7 > 2 * e12) and (l12 > 2 * l7)
    print(f"    -> period is {'DETECTED FROM INPUT (re-locks in-sequence)' if verdict1 else 'NOT clearly input-detected'}")

    # ---- (2) linearity of trend steering ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sel = series_set(SELECTION_SEEDS)
    Xs = np.stack([features(pipe, v) for v, _ in sel])
    ys = np.array([s for _, s in sel])
    layer = 4  # magnitude-R2 peak layer from the inventory (see results/)
    sc = StandardScaler().fit(Xs[:, layer])
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                             max_iter=2000).fit(sc.transform(Xs[:, layer]), ys)
    d = torch.tensor(clf.coef_[0] / sc.scale_).float()
    d = d / d.norm()
    base_norm = float(np.mean([np.linalg.norm(Xs[i, layer])
                               for i in range(len(sel))]))

    ks = [-4, -2, -1, -0.5, 0.5, 1, 2, 4]
    moves = {k: [] for k in ks}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        s = trend(slope=0.005 * rng.choice([-1, 1]), length=LENGTH,
                  level=rng.uniform(3, 5), noise=NOISE, seed=seed)
        y0, sc_ = yhat(pipe, s.values)
        for k in ks:
            y1, _ = yhat(pipe, s.values,
                         add_direction_hooks(pipe, layer, d,
                                             k * 0.5 * base_norm))
            moves[k].append((y1 - y0) / sc_)
    mean_moves = {k: float(np.mean(v)) for k, v in moves.items()}
    kk = np.array(ks)
    mm = np.array([mean_moves[k] for k in ks])
    r = float(np.corrcoef(kk, mm)[0, 1])
    print(f"\n(2) trend-direction steering, layer {layer}, k x 0.5*norm:")
    for k in ks:
        m, lo, hi = bootstrap_ci(moves[k])
        print(f"    k={k:+.1f}: dyhat/scale {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print(f"    correlation(k, response) = {r:+.2f} -> "
          f"{'clean linear readout' if abs(r) > 0.95 else 'NOT a linear readout (as the H2 null predicted)'}")

    Path("results/mechanism.json").write_text(json.dumps(dict(
        heads=[list(h) for h in heads], cp=CP,
        lag7_profile=p7.tolist(), lag12_profile=p12.tolist(),
        relock_latency_steps=latency, input_detected=bool(verdict1),
        steering_k=ks, steering_moves={str(k): v for k, v in moves.items()},
        steering_linearity_r=r), indent=1))
    print("wrote results/mechanism.json")


if __name__ == "__main__":
    main()
