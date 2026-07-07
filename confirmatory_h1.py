"""CONFIRMATORY H1 — executes the pre-registered protocol verbatim.

Frozen protocol (PREREGISTRATION.md, commit 4b8f0ac):
  Data: pattern seasonality, length 140, noise 0.05, amplitude 1.0.
  (a) descriptive: >= 1 head with seasonal attention ratio > 3x the uniform
      bidirectional null (tol=1, EOS excluded) at BOTH P=7 and P=12, averaged
      over SELECTION seeds 100-104.
  (b) causal: the group of all such heads (capped at 10% of heads, keeping the
      highest min-ratio ones), patched clean->corrupted on period-7<->12 pairs,
      EVALUATED on held-out seeds 105-119: mean value-space recovery >= 0.15
      with 95% bootstrap CI lower bound > 0.05; size-matched random control
      group mean < 0.05. Metric is the deterministic first-step forecast
      (m2/m3 as frozen), all positions.
  H1-general: holds in >= 3 of 4 models, incl. at least one of {base, large}.

Interpretation note (fixed before running): the prereg's control clause
"size-matched random control group (same layers excluded)" is implemented as
sampling control heads only from layers containing NO selected head, with a
fixed rng seed hardcoded below.

THIS SCRIPT UNBLINDS SEEDS 100-119. Results are reported as-is; thresholds do
not move; failures are nulls.

    python confirmatory_h1.py --device mps
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from attention_analysis import bootstrap_ci, head_scores
from chronos_harness import (get_inner, load_pipeline, patch_heads,
                             run_with_cache)
from synthetic import period_pair, seasonal
from verify_harness import STUDY_MODELS

LENGTH = 140
NOISE = 0.05
PERIODS = (7, 12)
SELECTION_SEEDS = range(100, 105)
EVAL_SEEDS = range(105, 120)
CONTROL_RNG_SEED = 12345


def select_heads(pipe) -> tuple[list, dict]:
    ratios = {}
    for period in PERIODS:
        per_seed = []
        for seed in SELECTION_SEEDS:
            s = seasonal(period=period, length=LENGTH, kind="pattern",
                         noise=NOISE, seed=seed)
            cache = run_with_cache(pipe, s.values)
            per_seed.append(head_scores(cache.attn, period, tol=1,
                                        valid_len=cache.valid_len))
        ratios[period] = np.mean(per_seed, axis=0)

    n_layers, n_heads = ratios[PERIODS[0]].shape
    passing = [(l, h) for l in range(n_layers) for h in range(n_heads)
               if all(ratios[p][l, h] > 3.0 for p in PERIODS)]
    cap = max(1, int(0.10 * n_layers * n_heads))
    passing.sort(key=lambda lh: -min(ratios[p][lh] for p in PERIODS))
    return passing[:cap], ratios


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).isoformat()
    print(f"CONFIRMATORY H1 — unblinding seeds 100-119 at {stamp}")

    results, confirmed_models = {}, []
    for model_id in STUDY_MODELS:
        pipe = load_pipeline(model_id, args.device)
        cfg = get_inner(pipe).config
        name = model_id.split("/")[-1]

        heads, ratios = select_heads(pipe)
        h1a = len(heads) >= 1
        print(f"\n=== {name} ({cfg.num_layers}L x {cfg.num_heads}H) ===")
        print(f"  (a) selection (seeds 100-104): {len(heads)} head(s) > 3x "
              f"null at both periods: {['L%dH%d' % lh for lh in heads]}"
              f"  -> {'PASS' if h1a else 'FAIL'}")

        if not h1a:
            results[name] = dict(h1a=False, heads=[], confirmed=False)
            continue

        cand_layers = {l for l, _ in heads}
        pool = [(l, h) for l in range(cfg.num_layers)
                if l not in cand_layers for h in range(cfg.num_heads)]
        if len(pool) < len(heads):
            # DEVIATIONS.md D1: layer-exclusion pool empty/too small (selected
            # heads span all layers) -> fall back to head-exclusion
            pool = [(l, h) for l in range(cfg.num_layers)
                    for h in range(cfg.num_heads) if (l, h) not in heads]
            print(f"      [DEVIATIONS.md D1: control pool = non-selected heads]")
        rng = np.random.default_rng(CONTROL_RNG_SEED)
        control = [pool[i] for i in
                   rng.choice(len(pool), len(heads), replace=False)]

        group_eff, control_eff = [], []
        for seed in EVAL_SEEDS:
            pair = period_pair(7, 12, length=LENGTH, noise=NOISE, seed=seed)
            r = patch_heads(pipe, pair.clean.values, pair.corrupted.values,
                            heads)
            group_eff.append(r["effect"])
            r = patch_heads(pipe, pair.clean.values, pair.corrupted.values,
                            control)
            control_eff.append(r["effect"])

        gm, glo, ghi = bootstrap_ci(group_eff)
        cm = float(np.mean(control_eff))
        h1b = gm >= 0.15 and glo > 0.05 and cm < 0.05
        confirmed = h1a and h1b
        if confirmed:
            confirmed_models.append(name)

        print(f"  (b) causal (seeds 105-119, value-space recovery): group "
              f"{gm:+.3f} [{glo:+.3f},{ghi:+.3f}] vs control {cm:+.3f}")
        print(f"      thresholds: mean>=0.15 {'PASS' if gm >= 0.15 else 'FAIL'}"
              f" | CI-lb>0.05 {'PASS' if glo > 0.05 else 'FAIL'}"
              f" | control<0.05 {'PASS' if cm < 0.05 else 'FAIL'}")
        print(f"  H1 per-model verdict: "
              f"{'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")

        results[name] = dict(
            h1a=h1a, heads=[list(x) for x in heads],
            ratios={str(p): ratios[p].tolist() for p in PERIODS},
            control=[list(x) for x in control],
            group_effects=group_eff, control_effects=control_eff,
            group=dict(mean=gm, ci=[glo, ghi]), control_mean=cm,
            h1b=bool(h1b), confirmed=bool(confirmed))

    general = (len(confirmed_models) >= 3
               and any(m.endswith(("base", "large")) for m in confirmed_models))
    print(f"\nH1-GENERAL: {len(confirmed_models)}/4 models confirmed "
          f"({confirmed_models}) -> "
          f"{'CONFIRMED' if general else 'NOT CONFIRMED'} "
          f"(needs >=3 incl. base or large)")

    blob = dict(timestamp=stamp, protocol="PREREGISTRATION.md",
                confirmed_models=confirmed_models,
                h1_general=bool(general), models=results)
    out = Path(args.out) / "confirmatory-h1.json"
    out.write_text(json.dumps(blob, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
