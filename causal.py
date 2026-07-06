"""Stage 2 — causal validation: turn inventory candidates into causally
confirmed components.

EXPLORATORY ONLY (seeds < 100). The pre-registered confirmatory H1 run on seeds
100-119 is a separate, deliberate step — this file never touches that range.

Per model, using the seasonal candidates from results/inventory-<model>.json:
  * activation patching (clean -> corrupted) of the candidate GROUP on four
    minimal-pair types: period(7<->12), phase, trend-on/off, changepoint-location
    — the off-diagonal pair types are the selectivity controls;
  * single- vs ALL-position patching, both reported (single-position typically
    undershoots per-component effects);
  * size-matched random-head control group + mismatched-clean control (clean
    store from a different seed: effects must require the matched pair);
  * per-head effects for candidates vs a sample of non-candidates, with a
    label-permutation p-value (B=10,000) and paired-bootstrap CIs (B=10,000)
    over seeds on everything;
  * path patching: direct-vs-total effect for the top head (downstream encoder
    frozen), and sender->decoder-cross-attention receiver decomposition;
  * the top changepoint-collapse heads are additionally patched on their own
    pair type (changepoint-location) and on period pairs (reverse selectivity).

Metrics: normalized recovery of the logit difference between the clean-correct
and corrupted-correct value bins (primary), and of the expected-value forecast
(secondary).

    python causal.py --device mps                      # all four scales
    python causal.py --device mps --models amazon/chronos-t5-small
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from attention_analysis import bootstrap_ci, patching_effect, permutation_test
from chronos_harness import (_capture_pre_o, cross_attention_receivers,
                             get_inner, load_pipeline, logit_diff,
                             patch_head_direct, patch_heads, tokenize)
from synthetic import (changepoint_location_pair, period_pair, phase_pair,
                       trend_onoff_pair)
from verify_harness import STUDY_MODELS

LENGTH = 168
NOISE = 0.05
SEEDS = range(10)          # exploratory (< 100)
N_NULL_HEADS = 30          # non-candidate sample for the permutation null

PAIR_TYPES = {
    "period": lambda seed: period_pair(7, 12, length=LENGTH, noise=NOISE,
                                       seed=seed),
    "phase": lambda seed: phase_pair(7, length=LENGTH, noise=NOISE, seed=seed),
    "trend_onoff": lambda seed: trend_onoff_pair(7, 0.03, length=LENGTH,
                                                 noise=NOISE, seed=seed),
    "cp_location": lambda seed: changepoint_location_pair(
        7, length=LENGTH, noise=NOISE, seed=seed),
}


def both_effects(pipe, r, pair) -> tuple[float, float]:
    """(logit-diff recovery, value-space recovery) from a patch_heads result."""
    y_cl, y_co = float(pair.clean.future[0]), float(pair.corrupted.future[0])
    m_clean = logit_diff(pipe, r["logits_clean"], y_cl, y_co, r["scale_clean"])
    m_corr = logit_diff(pipe, r["logits_corr"], y_cl, y_co, r["scale_corr"])
    m_patch = logit_diff(pipe, r["logits_patched"], y_cl, y_co, r["scale_corr"])
    return patching_effect(m_clean, m_corr, m_patch), r["effect"]


def ci_str(values) -> str:
    m, lo, hi = bootstrap_ci(values)
    return f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]"


def run_model(model_id: str, device: str | None, out_dir: Path) -> dict:
    t0 = time.time()
    pipe = load_pipeline(model_id, device)
    cfg = get_inner(pipe).config
    inv = json.loads(
        (out_dir / f"inventory-{model_id.split('/')[-1]}.json").read_text())
    candidates = [tuple(c) for c in inv["seasonal"]["candidates"]]
    collapse = np.array(inv["changepoint"]["collapse"])
    cp_heads = [tuple(map(int, np.unravel_index(f, collapse.shape)))
                for f in np.argsort(collapse, axis=None)[::-1][:5]]

    all_heads = [(l, h) for l in range(cfg.num_layers)
                 for h in range(cfg.num_heads)]
    non_cand = [x for x in all_heads if x not in candidates]
    rng = np.random.default_rng(7)
    control_group = [non_cand[i] for i in
                     rng.choice(len(non_cand), len(candidates), replace=False)]
    null_heads = [non_cand[i] for i in
                  rng.choice(len(non_cand), min(N_NULL_HEADS, len(non_cand)),
                             replace=False)]
    last_pos = [LENGTH - 1]    # final context token (EOS is LENGTH)

    print(f"\n=== {model_id} ({cfg.num_layers}L x {cfg.num_heads}H) — "
          f"{len(candidates)} seasonal candidates ===")

    res: dict = {k: {"group": [], "single": [], "control": []}
                 for k in PAIR_TYPES}
    res["mismatched"] = []
    res["cp_group_cp"] = []
    res["cp_group_period"] = []
    per_head_cand: dict = {c: [] for c in candidates}
    per_head_null: dict = {c: [] for c in null_heads}

    for seed in SEEDS:
        pairs = {k: f(seed) for k, f in PAIR_TYPES.items()}
        for ptype, pair in pairs.items():
            ids_cl, mask_cl, _ = tokenize(pipe, pair.clean.values)
            store = _capture_pre_o(pipe, ids_cl, mask_cl,
                                   list(range(cfg.num_layers)))
            def eff(heads, positions=None, st=store, pr=pair):
                r = patch_heads(pipe, pr.clean.values, pr.corrupted.values,
                                heads, _clean_store=st, positions=positions)
                return both_effects(pipe, r, pr)

            res[ptype]["group"].append(eff(candidates))
            res[ptype]["single"].append(eff(candidates, positions=last_pos))
            res[ptype]["control"].append(eff(control_group))

            if ptype == "period":
                for c in candidates:
                    per_head_cand[c].append(eff([c]))
                for c in null_heads:
                    per_head_null[c].append(eff([c]))
                # mismatched-clean control: clean run from a different seed
                other = PAIR_TYPES["period"](seed + 50)
                r = patch_heads(pipe, other.clean.values, pair.corrupted.values,
                                candidates)
                res["mismatched"].append(both_effects(pipe, r, pair))
            if ptype == "cp_location":
                res["cp_group_cp"].append(eff(cp_heads))
            if ptype == "period":
                res["cp_group_period"].append(eff(cp_heads))

    # ------- report: group effects by pair type (logit metric primary) -------
    print(f"  group patch of seasonal candidates, logit-diff recovery "
          f"(95% paired-bootstrap CI over {len(list(SEEDS))} seeds):")
    print(f"    {'pair type':<12} {'all-position':>24} {'single-pos':>24} "
          f"{'control group':>24}")
    for ptype in PAIR_TYPES:
        row = [ci_str([e[0] for e in res[ptype][k]])
               for k in ("group", "single", "control")]
        print(f"    {ptype:<12} {row[0]:>24} {row[1]:>24} {row[2]:>24}")
    print(f"    mismatched-clean control (period): "
          f"{ci_str([e[0] for e in res['mismatched']])}")

    # ------- permutation null over per-head effects (period pairs) -------
    cand_means = [np.mean([e[0] for e in v]) for v in per_head_cand.values()]
    null_means = [np.mean([e[0] for e in v]) for v in per_head_null.values()]
    p = permutation_test(cand_means, null_means)
    print(f"  per-head (period): candidates mean {np.mean(cand_means):+.4f} vs "
          f"{len(null_means)} non-candidates {np.mean(null_means):+.4f}, "
          f"permutation p = {p:.4f}")

    # ------- changepoint-head selectivity -------
    print(f"  top-5 collapse heads {['L%dH%d' % c for c in cp_heads]}: "
          f"cp_location {ci_str([e[0] for e in res['cp_group_cp']])} | "
          f"period {ci_str([e[0] for e in res['cp_group_period']])}")

    # ------- path patching (period pairs) -------
    top_head = max(per_head_cand, key=lambda c: np.mean(
        [e[0] for e in per_head_cand[c]]))
    direct, total = [], []
    for seed in range(3):
        pair = PAIR_TYPES["period"](seed)
        rt = patch_heads(pipe, pair.clean.values, pair.corrupted.values,
                         [top_head])
        rd = patch_head_direct(pipe, pair.clean.values, pair.corrupted.values,
                               *top_head)
        total.append(both_effects(pipe, rt, pair)[0])
        direct.append(both_effects(pipe, rd, pair)[0])
    share = np.mean(direct) / np.mean(total) if np.mean(total) else float("nan")
    print(f"  path (top head L{top_head[0]}H{top_head[1]}): total "
          f"{np.mean(total):+.4f}, direct-only {np.mean(direct):+.4f} "
          f"({share:.0%} direct to encoder output)")

    pair = PAIR_TYPES["period"](0)
    rec = cross_attention_receivers(pipe, pair.clean.values,
                                    pair.corrupted.values, candidates)
    eff = rec["receiver_effects"]
    order = np.argsort(eff, axis=None)[::-1][:3]
    tops = []
    for f in order:
        dl, dh = np.unravel_index(f, eff.shape)
        tops.append(f"D{dl}H{dh} {eff[dl, dh] / rec['sender_effect']:.0%}")
    print(f"  receivers (sender = candidate group, effect "
          f"{rec['sender_effect']:+.3f}): top cross-attn heads "
          f"{' | '.join(tops)}; top-3 sum "
          f"{eff.flatten()[order].sum() / rec['sender_effect']:.0%} of sender")
    print(f"  ({time.time() - t0:.0f}s)")

    blob = dict(
        model=model_id, candidates=[list(c) for c in candidates],
        cp_heads=[list(c) for c in cp_heads],
        control_group=[list(c) for c in control_group],
        seeds=list(SEEDS), length=LENGTH, noise=NOISE,
        effects={pt: {k: res[pt][k] for k in ("group", "single", "control")}
                 for pt in PAIR_TYPES},
        mismatched=res["mismatched"],
        cp_group=dict(cp_location=res["cp_group_cp"],
                      period=res["cp_group_period"]),
        per_head_candidates={f"L{l}H{h}": v for (l, h), v
                             in per_head_cand.items()},
        per_head_null={f"L{l}H{h}": v for (l, h), v in per_head_null.items()},
        permutation_p=float(p),
        path=dict(top_head=list(top_head), total=total, direct=direct),
        receivers=dict(effects=eff.tolist(),
                       sender_effect=rec["sender_effect"]),
    )
    out = out_dir / f"causal-{model_id.split('/')[-1]}.json"
    out.write_text(json.dumps(blob, indent=1))
    print(f"  wrote {out}")
    return blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--models", nargs="+", default=STUDY_MODELS)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out_dir = Path(args.out)

    blobs = [run_model(m, args.device, out_dir) for m in args.models]

    if len(blobs) > 1:
        print("\n=== cross-scale causal summary (exploratory, logit metric) ===")
        for b in blobs:
            g = np.mean([e[0] for e in b["effects"]["period"]["group"]])
            c = np.mean([e[0] for e in b["effects"]["period"]["control"]])
            s = np.mean([e[0] for e in b["effects"]["period"]["single"]])
            print(f"  {b['model'].split('/')[-1]:8s} period group {g:+.3f} "
                  f"(control {c:+.3f}, single-pos {s:+.3f}), perm p = "
                  f"{b['permutation_p']:.4f}")


if __name__ == "__main__":
    main()
