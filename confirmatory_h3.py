"""CONFIRMATORY H3 — double dissociation (pre-registered protocol).

Frozen protocol (PREREGISTRATION.md): D = increase in scale-normalized
first-step absolute error on held-out seeds. (i) Mean-ablating the H1 head
group must give D(seasonal task) >= 2x D(trend task); AND (ii) zero-projecting
the H2 direction must give D(trend task) >= 2x D(seasonal task); each ratio
with bootstrap CI lower bound > 1.

Operationalization fixed before running:
  * H1 head group = the frozen H1 selection rule on seeds 100-104 (same code
    path as confirmatory_h1.select_heads). mini returned zero heads there, so
    clause (i) is not evaluable for mini and H3 fails there by definition.
  * H2 direction = the frozen H2 procedure on seeds 100-104 (same layer
    choice + L1-logistic direction as confirmatory_h2).
  * Tasks on eval seeds 105-119: seasonal = pattern P=7 length-140 series
    (one per seed); trend = both members of trend_pair(slope 0.02), errors
    averaged per seed. Data params as frozen (length 140, noise 0.05).
  * Ratio CIs: percentile bootstrap (B=10,000) over the 15 eval seeds,
    resampling numerator and denominator jointly (paired by seed).
  * The prereg defines no explicit H3-general rule; per-model verdicts are
    reported with a count.

    python confirmatory_h3.py --device mps
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from chronos_harness import (forecast_scores, get_inner, load_pipeline,
                             mean_ablate_hooks, project_out_hooks)
from confirmatory_h1 import select_heads
from confirmatory_h2 import SELECTION_SEEDS, features, series_set
from synthetic import seasonal, trend_pair
from verify_harness import STUDY_MODELS

LENGTH = 140
NOISE = 0.05
EVAL_SEEDS = range(105, 120)


def fit_h2_direction(pipe):
    """Identical procedure to confirmatory_h2: LOO layer choice + L1-logistic
    direction on the 10 selection series."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cfg = get_inner(pipe).config
    sel = series_set(SELECTION_SEEDS)
    Xs = np.stack([features(pipe, v) for v, _ in sel])
    ys = np.array([s for _, s in sel])
    loo = []
    for layer in range(cfg.num_layers):
        hits = 0
        for i in range(len(sel)):
            tr = [j for j in range(len(sel)) if j != i]
            sc = StandardScaler().fit(Xs[tr, layer])
            clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                     max_iter=2000)
            clf.fit(sc.transform(Xs[tr, layer]), ys[tr])
            hits += clf.predict(sc.transform(Xs[[i], layer]))[0] == ys[i]
        loo.append(hits / len(sel))
    layer = int(np.argmax(loo))
    sc = StandardScaler().fit(Xs[:, layer])
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                             max_iter=2000).fit(sc.transform(Xs[:, layer]), ys)
    d = torch.tensor(clf.coef_[0] / sc.scale_).float()
    return layer, d / d.norm()


def ratio_ci(num, den, n_boot: int = 10_000, seed: int = 0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return (float(num.mean() / den.mean()),
            float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).isoformat()
    print(f"CONFIRMATORY H3 — double dissociation, eval seeds 105-119, "
          f"at {stamp}")

    results, confirmed_models = {}, []
    for model_id in STUDY_MODELS:
        pipe = load_pipeline(model_id, args.device)
        name = model_id.split("/")[-1]
        heads, _ = select_heads(pipe)
        layer, direction = fit_h2_direction(pipe)
        print(f"\n=== {name}: H1 group = {len(heads)} head(s), H2 direction @ "
              f"layer {layer} ===")

        conds = {"intact": lambda: [],
                 "head_ablate": (lambda: mean_ablate_hooks(pipe, heads))
                 if heads else None,
                 "proj": lambda: project_out_hooks(pipe, layer, direction)}
        # err[cond][task] = list over seeds
        err = {c: {"seasonal": [], "trend": []} for c in conds if conds[c]}
        for seed in EVAL_SEEDS:
            s = seasonal(period=7, length=LENGTH, kind="pattern", noise=NOISE,
                         seed=seed)
            p = trend_pair(slope=0.02, length=LENGTH, noise=NOISE, seed=seed)
            for cond, mk in conds.items():
                if mk is None:
                    continue
                err[cond]["seasonal"].append(
                    forecast_scores(pipe, s.values, float(s.future[0]),
                                    hooks=mk())["abs_err"])
                t = [forecast_scores(pipe, sr.values, float(sr.future[0]),
                                     hooks=mk())["abs_err"]
                     for sr in (p.clean, p.corrupted)]
                err[cond]["trend"].append(float(np.mean(t)))

        model_res = dict(heads=[list(h) for h in heads], h2_layer=layer)
        halves = {}
        if heads:
            d_seas = [a - b for a, b in zip(err["head_ablate"]["seasonal"],
                                            err["intact"]["seasonal"])]
            d_trend = [a - b for a, b in zip(err["head_ablate"]["trend"],
                                             err["intact"]["trend"])]
            m, lo, hi = ratio_ci(d_seas, d_trend)
            ok1 = m >= 2.0 and lo > 1.0
            print(f"  (i) head-ablation: D(seas) {np.mean(d_seas):+.4f}, "
                  f"D(trend) {np.mean(d_trend):+.4f}, ratio {m:+.2f} "
                  f"[{lo:+.2f},{hi:+.2f}] -> {'PASS' if ok1 else 'FAIL'}")
            halves["head_ablation"] = dict(d_seasonal=d_seas, d_trend=d_trend,
                                           ratio=[m, lo, hi], passed=bool(ok1))
        else:
            ok1 = False
            print("  (i) head-ablation: NOT EVALUABLE (H1 selection returned "
                  "0 heads) -> FAIL")
            halves["head_ablation"] = dict(passed=False,
                                           reason="no H1 heads selected")

        d_trend = [a - b for a, b in zip(err["proj"]["trend"],
                                         err["intact"]["trend"])]
        d_seas = [a - b for a, b in zip(err["proj"]["seasonal"],
                                        err["intact"]["seasonal"])]
        m, lo, hi = ratio_ci(d_trend, d_seas)
        ok2 = m >= 2.0 and lo > 1.0
        print(f"  (ii) direction-projection: D(trend) {np.mean(d_trend):+.4f},"
              f" D(seas) {np.mean(d_seas):+.4f}, ratio {m:+.2f} "
              f"[{lo:+.2f},{hi:+.2f}] -> {'PASS' if ok2 else 'FAIL'}")
        halves["projection"] = dict(d_trend=d_trend, d_seasonal=d_seas,
                                    ratio=[m, lo, hi], passed=bool(ok2))

        confirmed = ok1 and ok2
        if confirmed:
            confirmed_models.append(name)
        print(f"  H3 per-model verdict: "
              f"{'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")
        model_res.update(halves=halves, confirmed=bool(confirmed))
        results[name] = model_res

    print(f"\nH3: {len(confirmed_models)}/4 models confirmed "
          f"({confirmed_models}); prereg defines no general rule — reported "
          f"as-is")
    blob = dict(timestamp=stamp, confirmed_models=confirmed_models,
                models=results)
    out = Path(args.out) / "confirmatory-h3.json"
    out.write_text(json.dumps(blob, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
