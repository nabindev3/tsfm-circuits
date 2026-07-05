"""First-signal script: score every Chronos encoder head for seasonal attention.

Generates seasonal series at two different periods, caches encoder attention, and
ranks every head by how much attention it puts at the seasonal lag relative to a
uniform null. A head that scores high at BOTH periods is tracking "one cycle ago"
as a relation, not a fixed offset — that's the induction-head signature, and those
are the candidates for causal patching (patch_all_heads in chronos_harness).

    python demo.py --device mps
"""

from __future__ import annotations

import argparse

import numpy as np

from attention_analysis import head_scores, rank_heads
from chronos_harness import DEFAULT_MODEL, get_inner, load_pipeline, run_with_cache
from synthetic import seasonal


def score_period(pipe, period: int, length: int, kind: str, noise: float,
                 tol: int, seeds: list[int]) -> np.ndarray:
    """Mean seasonal-attention ratio per head, averaged over seeds: [layers, heads]."""
    ratios = []
    for seed in seeds:
        s = seasonal(period=period, length=length, kind=kind, noise=noise, seed=seed)
        cache = run_with_cache(pipe, s.values)
        ratios.append(head_scores(cache.attn, period, tol=tol,
                                  valid_len=cache.valid_len))
    return np.mean(ratios, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None, help="mps | cuda | cpu (default: auto)")
    ap.add_argument("--period", type=int, default=7)
    ap.add_argument("--period2", type=int, default=12,
                    help="second period; heads high on both generalize the lag")
    ap.add_argument("--length", type=int, default=140)
    ap.add_argument("--kind", default="pattern", choices=["pattern", "sine"])
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--tol", type=int, default=1,
                    help="attention window half-width around the lag")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--top", type=int, default=None,
                    help="only print the top-K heads (default: all)")
    args = ap.parse_args()

    pipe = load_pipeline(args.model, args.device)
    cfg = get_inner(pipe).config
    seeds = list(range(args.n_seeds))
    print(f"{args.model}: {cfg.num_layers} layers x {cfg.num_heads} heads | "
          f"kind={args.kind} noise={args.noise} length={args.length} "
          f"seeds={len(seeds)} tol={args.tol}")

    r1 = score_period(pipe, args.period, args.length, args.kind, args.noise,
                      args.tol, seeds)
    r2 = score_period(pipe, args.period2, args.length, args.kind, args.noise,
                      args.tol, seeds)

    ranked = rank_heads(r1)
    if args.top:
        ranked = ranked[: args.top]
    print(f"\nratio = attention mass at the seasonal lag / uniform null "
          f"(1.0 = no preference)")
    print(f"{'head':>8} {'P=' + str(args.period):>8} {'P=' + str(args.period2):>8}"
          f"  both>3x")
    for layer, head, ratio in ranked:
        both = "  <-- lag-tracking" if ratio > 3 and r2[layer, head] > 3 else ""
        print(f"  L{layer}H{head:<4} {ratio:8.2f} {r2[layer, head]:8.2f}{both}")

    generalizing = [(l, h) for l in range(cfg.num_layers)
                    for h in range(cfg.num_heads)
                    if r1[l, h] > 3 and r2[l, h] > 3]
    print(f"\n{len(generalizing)} head(s) exceed 3x the null at BOTH periods: "
          f"{['L%dH%d' % lh for lh in generalizing]}")
    print("these are the candidate seasonal induction heads — test them causally "
          "with chronos_harness.patch_all_heads on a period_pair.")


if __name__ == "__main__":
    main()
