"""RQ1 causal: per-head activation patching on period-7 <-> period-12 minimal pairs.

For each encoder head, splice the clean run's head output into the corrupted run
and measure how far the first-step forecast moves toward the clean prediction
(normalized: 0 = no effect, 1 = full recovery). Averaged over seeds. Cross-marks
the heads that demo.py flagged as lag-tracking, so you can see whether the
attention story and the causal story agree.

    python patch_demo.py --device mps
"""

from __future__ import annotations

import argparse

import numpy as np

from attention_analysis import bootstrap_ci, head_scores
from chronos_harness import (DEFAULT_MODEL, load_pipeline, patch_all_heads,
                             run_with_cache)
from synthetic import period_pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--p-clean", type=int, default=7)
    ap.add_argument("--p-corr", type=int, default=12)
    ap.add_argument("--length", type=int, default=140)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--n-seeds", type=int, default=3)
    args = ap.parse_args()

    pipe = load_pipeline(args.model, args.device)

    effects, attn_ratios = [], []
    for seed in range(args.n_seeds):
        pair = period_pair(args.p_clean, args.p_corr, length=args.length,
                           noise=args.noise, seed=seed)
        r = patch_all_heads(pipe, pair)
        effects.append(r["effects"])
        cache = run_with_cache(pipe, pair.clean.values)
        attn_ratios.append(head_scores(cache.attn, args.p_clean,
                                       valid_len=cache.valid_len))
        print(f"seed {seed}: yhat clean {r['yhat_clean']:+.3f} vs corrupted "
              f"{r['yhat_corr']:+.3f} (gap {r['yhat_clean'] - r['yhat_corr']:+.3f})")

    mean_eff = np.mean(effects, axis=0)
    mean_attn = np.mean(attn_ratios, axis=0)

    print(f"\nper-head patching effect (clean head -> corrupted run, "
          f"P={args.p_clean} vs P={args.p_corr}, {args.n_seeds} seeds)")
    print(f"{'head':>8} {'effect':>8} {'attn x null':>12}")
    order = np.argsort(mean_eff, axis=None)[::-1]
    for flat in order:
        layer, head = np.unravel_index(flat, mean_eff.shape)
        flag = "  <-- lag-tracking candidate" if mean_attn[layer, head] > 3 else ""
        print(f"  L{layer}H{head:<4} {mean_eff[layer, head]:+8.3f} "
              f"{mean_attn[layer, head]:12.2f}{flag}")

    eff_stack = np.stack(effects)  # [seeds, layers, heads]
    print("\ntop-5 causal heads (mean [95% bootstrap CI over seeds]):")
    for flat in order[:5]:
        layer, head = np.unravel_index(flat, mean_eff.shape)
        m, lo, hi = bootstrap_ci(eff_stack[:, layer, head])
        print(f"  L{layer}H{head}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
