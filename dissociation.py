"""Stage 3 — the double dissociation: ablate {seasonal heads, trend direction,
changepoint heads} x test on {seasonal, trend, changepoint, mixed} families.

EXPLORATORY ONLY (seeds < 100).

Ablations:
  * seasonal heads   — mean-ablate the inventory's seasonal candidate group
                       (position-specific info destroyed, average kept);
  * trend direction  — zero-project the up-vs-down mass-mean direction, fit on
                       held-out series at the layer where the inventory's slope
                       R^2 peaks, applied at every position;
  * changepoint heads— mean-ablate the top-5 post-CP attention-collapse heads.
                       NOTE: this group overlaps the seasonal candidates at
                       most scales (reported per model) — the overlap IS the
                       shared-circuit finding from Stages 1-2.

Degradation metric: increase in scale-normalized CRPS of the first forecast
step vs the intact model (secondary: scale-normalized absolute error), with
95% paired-bootstrap CIs over seeds.

    python dissociation.py --device mps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from attention_analysis import bootstrap_ci
from chronos_harness import (forecast_scores, get_inner, load_pipeline,
                             mean_ablate_hooks, project_out_hooks,
                             run_with_cache)
from synthetic import season_plus_changepoint, seasonal, seasonal_trend, trend
from verify_harness import STUDY_MODELS

LENGTH = 168
NOISE = 0.05
SEEDS = range(10)          # test seeds (exploratory)
FIT_SEEDS = range(30, 70)  # trend-direction fitting (disjoint, still < 100)


def make_family(name: str, seed: int):
    rng = np.random.default_rng(seed + 90_000)
    if name == "seasonal":
        return seasonal(period=7, length=LENGTH, kind="pattern", noise=NOISE,
                        seed=seed)
    if name == "trend":
        mag = 10 ** rng.uniform(np.log10(0.002), np.log10(0.012))
        return trend(slope=mag * rng.choice([-1, 1]), length=LENGTH,
                     level=rng.uniform(3, 5), noise=NOISE, seed=seed)
    if name == "changepoint":
        return season_plus_changepoint(period=7, length=LENGTH,
                                       cp_kind="pattern", noise=NOISE,
                                       seed=seed)
    if name == "mixed":
        return seasonal_trend(period=7, slope=0.02, length=LENGTH, noise=NOISE,
                              seed=seed)
    raise ValueError(name)


FAMILIES = ("seasonal", "trend", "changepoint", "mixed")


def fit_trend_direction(pipe, layer: int) -> torch.Tensor:
    """Mass-mean direction (mean up-trend features minus mean down-trend
    features) on mean-pooled residuals at `layer`."""
    feats, signs = [], []
    for i in FIT_SEEDS:
        rng = np.random.default_rng(i)
        sign = 1 if i % 2 == 0 else -1
        mag = 10 ** rng.uniform(np.log10(0.002), np.log10(0.012))
        s = trend(slope=sign * mag, length=LENGTH, level=rng.uniform(3, 5),
                  noise=NOISE, seed=i)
        cache = run_with_cache(pipe, s.values)
        feats.append(cache.resid[layer][:cache.valid_len].mean(0))
        signs.append(sign)
    feats = torch.stack(feats)
    signs = torch.tensor(signs)
    d = feats[signs > 0].mean(0) - feats[signs < 0].mean(0)
    return d / d.norm()


def run_model(model_id: str, device: str | None, out_dir: Path) -> dict:
    t0 = time.time()
    pipe = load_pipeline(model_id, device)
    inv = json.loads(
        (out_dir / f"inventory-{model_id.split('/')[-1]}.json").read_text())
    seasonal_heads = [tuple(c) for c in inv["seasonal"]["candidates"]]
    collapse = np.array(inv["changepoint"]["collapse"])
    cp_heads = [tuple(map(int, np.unravel_index(f, collapse.shape)))
                for f in np.argsort(collapse, axis=None)[::-1][:5]]
    overlap = sorted(set(seasonal_heads) & set(cp_heads))
    probe_layer = inv["trend"]["peak_r2_layer"]
    direction = fit_trend_direction(pipe, probe_layer)

    print(f"\n=== {model_id} ===")
    print(f"  seasonal group: {len(seasonal_heads)} heads | cp group: "
          f"{['L%dH%d' % c for c in cp_heads]} | overlap: "
          f"{['L%dH%d' % c for c in overlap] or 'none'} | trend direction @ "
          f"layer {probe_layer}")

    ablations = {
        "none": lambda: [],
        "seasonal_heads": lambda: mean_ablate_hooks(pipe, seasonal_heads),
        "trend_direction": lambda: project_out_hooks(pipe, probe_layer,
                                                     direction),
        "cp_heads": lambda: mean_ablate_hooks(pipe, cp_heads),
    }

    # scores[ablation][family] = list over seeds of dict(abs_err, crps)
    scores = {a: {f: [] for f in FAMILIES} for a in ablations}
    for seed in SEEDS:
        for fam in FAMILIES:
            s = make_family(fam, seed)
            for abl, make_hooks in ablations.items():
                scores[abl][fam].append(
                    forecast_scores(pipe, s.values, float(s.future[0]),
                                    hooks=make_hooks()))

    # degradation = ablated - intact, paired per seed
    table = {}
    print(f"  dissociation table: delta scale-normalized CRPS vs intact "
          f"(95% paired-bootstrap CI over {len(list(SEEDS))} seeds)")
    header = "ablate \\ test"
    print(f"    {header:<16}" + "".join(f"{f:>26}" for f in FAMILIES))
    for abl in ("seasonal_heads", "trend_direction", "cp_heads"):
        cells = {}
        row = f"    {abl:<16}"
        for fam in FAMILIES:
            d = [scores[abl][fam][i]["crps"] - scores["none"][fam][i]["crps"]
                 for i in range(len(list(SEEDS)))]
            da = [scores[abl][fam][i]["abs_err"]
                  - scores["none"][fam][i]["abs_err"]
                  for i in range(len(list(SEEDS)))]
            m, lo, hi = bootstrap_ci(d)
            cells[fam] = dict(dcrps=d, dabs=da, mean=m, ci=[lo, hi])
            row += f"  {m:+.3f} [{lo:+.3f},{hi:+.3f}]"
        table[abl] = cells
        print(row)
    intact = {f: float(np.mean([x["crps"] for x in scores["none"][f]]))
              for f in FAMILIES}
    print("    intact CRPS/scale: "
          + " ".join(f"{f}={intact[f]:.3f}" for f in FAMILIES))
    print(f"  ({time.time() - t0:.0f}s)")

    blob = dict(model=model_id, seasonal_heads=[list(c) for c in seasonal_heads],
                cp_heads=[list(c) for c in cp_heads],
                overlap=[list(c) for c in overlap], probe_layer=probe_layer,
                seeds=list(SEEDS), intact_crps=intact, table=table)
    out = out_dir / f"dissociation-{model_id.split('/')[-1]}.json"
    out.write_text(json.dumps(blob, indent=1))
    print(f"  wrote {out}")
    return blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--models", nargs="+", default=STUDY_MODELS)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    blobs = [run_model(m, args.device, Path(args.out)) for m in args.models]

    if len(blobs) > 1:
        print("\n=== cross-scale dissociation summary (delta CRPS/scale) ===")
        for b in blobs:
            t = b["table"]
            print(f"  {b['model'].split('/')[-1]:8s} "
                  f"S-heads: seas {t['seasonal_heads']['seasonal']['mean']:+.3f} "
                  f"trend {t['seasonal_heads']['trend']['mean']:+.3f} | "
                  f"T-dir: seas {t['trend_direction']['seasonal']['mean']:+.3f} "
                  f"trend {t['trend_direction']['trend']['mean']:+.3f} | "
                  f"CP-heads: cp {t['cp_heads']['changepoint']['mean']:+.3f} "
                  f"trend {t['cp_heads']['trend']['mean']:+.3f}")


if __name__ == "__main__":
    main()
