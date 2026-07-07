"""CONFIRMATORY H2 — trend is a steerable direction (pre-registered protocol),
plus a labeled-EXPLORATORY redesign (stronger / multi-layer steering).

Frozen protocol (PREREGISTRATION.md): per model, a logistic probe on
mean-pooled cache.resid — layer and steering coefficient alpha chosen on seeds
100-104 — must (i) separate up/down trend_pairs with >= 90% accuracy on seeds
105-119, and (ii) adding +-alpha*direction must move the first-step forecast in
the predicted direction for >= 80% of held-out series with median
|delta yhat|/scale >= 0.05. H2-general: >= 3 of 4 models.

Operationalization fixed before running (prereg left these open, all chosen on
selection data only): layer = best leave-one-out accuracy on the 10 selection
series (ties -> lower layer); direction = L1-logistic coefficients mapped back
to raw residual space, unit-normalized; alpha = f * (mean per-position residual
norm at the layer), f the smallest of {0.05, 0.1, 0.2, 0.5, 1, 2} meeting both
steering criteria on the selection series (if none, the f with the best median
movement); each held-out series contributes two trials (+alpha must move yhat
up, -alpha down).

    python confirmatory_h2.py --device mps
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from chronos_harness import (_first_step_logits, _expected_value,
                             add_direction_hooks, get_inner, load_pipeline,
                             run_with_cache, tokenize)
from synthetic import trend_pair
from verify_harness import STUDY_MODELS

LENGTH = 140
NOISE = 0.05
SLOPE = 0.02
SELECTION_SEEDS = range(100, 105)
EVAL_SEEDS = range(105, 120)
FRACS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


def series_set(seeds):
    out = []
    for seed in seeds:
        p = trend_pair(slope=SLOPE, length=LENGTH, noise=NOISE, seed=seed)
        out.append((p.clean.values, +1))
        out.append((p.corrupted.values, -1))
    return out


def features(pipe, values):
    cache = run_with_cache(pipe, values)
    return (torch.stack([r[:cache.valid_len].mean(0) for r in cache.resid])
            .numpy())  # [layers, d]


@torch.no_grad()
def yhat(pipe, values, hooks=None) -> tuple[float, float]:
    try:
        ids, mask, scale = tokenize(pipe, values)
        s = float(scale[0])
        return _expected_value(pipe, _first_step_logits(pipe, ids, mask), s), s
    finally:
        for h in (hooks or []):
            h.remove()


def steer_trials(pipe, series, layer, direction, alpha):
    """[(correct: bool, |dyhat|/scale)] — two trials per series (+ up, - down)."""
    trials = []
    for values, _sign in series:
        y0, s = yhat(pipe, values)
        for sgn in (+1, -1):
            y1, _ = yhat(pipe, values,
                         add_direction_hooks(pipe, layer, direction,
                                             sgn * alpha))
            move = (y1 - y0) / s
            trials.append((move * sgn > 0, abs(move)))
    return trials


def crit(trials) -> tuple[float, float, bool]:
    frac_ok = float(np.mean([t[0] for t in trials]))
    med = float(np.median([t[1] for t in trials]))
    return frac_ok, med, frac_ok >= 0.8 and med >= 0.05


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).isoformat()
    print(f"CONFIRMATORY H2 — steering, seeds 100-119, at {stamp}")

    results, confirmed_models = {}, []
    for model_id in STUDY_MODELS:
        pipe = load_pipeline(model_id, args.device)
        cfg = get_inner(pipe).config
        name = model_id.split("/")[-1]

        sel = series_set(SELECTION_SEEDS)
        Xs = np.stack([features(pipe, v) for v, _ in sel])   # [10, layers, d]
        ys = np.array([s for _, s in sel])

        # layer: leave-one-out accuracy on the 10 selection series
        loo = []
        for layer in range(cfg.num_layers):
            hits = 0
            for i in range(len(sel)):
                tr = [j for j in range(len(sel)) if j != i]
                sc = StandardScaler().fit(Xs[tr, layer])
                clf = LogisticRegression(penalty="l1", solver="liblinear",
                                         C=1.0, max_iter=2000)
                clf.fit(sc.transform(Xs[tr, layer]), ys[tr])
                hits += clf.predict(sc.transform(Xs[[i], layer]))[0] == ys[i]
            loo.append(hits / len(sel))
        layer = int(np.argmax(loo))

        sc = StandardScaler().fit(Xs[:, layer])
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                 max_iter=2000).fit(sc.transform(Xs[:, layer]),
                                                    ys)
        d_raw = torch.tensor(clf.coef_[0] / sc.scale_).float()
        d_raw = d_raw / d_raw.norm()
        base_norm = float(np.mean([np.linalg.norm(Xs[i, layer])
                                   for i in range(len(sel))]))

        # alpha: smallest f meeting both criteria on SELECTION series
        chosen_f, sel_stats = None, {}
        for f in FRACS:
            fo, med, ok = crit(steer_trials(pipe, sel, layer, d_raw,
                                            f * base_norm))
            sel_stats[f] = (fo, med)
            if ok and chosen_f is None:
                chosen_f = f
        if chosen_f is None:
            chosen_f = max(FRACS, key=lambda f: sel_stats[f][1]
                           * (sel_stats[f][0] >= 0.5))
        alpha = chosen_f * base_norm

        # ---------- held-out evaluation ----------
        ev = series_set(EVAL_SEEDS)
        Xe = np.stack([features(pipe, v) for v, _ in ev])
        ye = np.array([s for _, s in ev])
        acc = float(np.mean(clf.predict(sc.transform(Xe[:, layer])) == ye))

        fo, med, steer_ok = crit(steer_trials(pipe, ev, layer, d_raw, alpha))
        confirmed = acc >= 0.90 and steer_ok
        if confirmed:
            confirmed_models.append(name)

        print(f"\n=== {name}: layer {layer} (LOO {loo[layer]:.2f}), "
              f"f={chosen_f} (alpha={alpha:.1f}) ===")
        print(f"  probe accuracy on held-out: {acc:.3f} "
              f"({'PASS' if acc >= 0.9 else 'FAIL'} >=0.90)")
        print(f"  steering on held-out: direction-correct {fo:.2f} "
              f"({'PASS' if fo >= 0.8 else 'FAIL'} >=0.80), median "
              f"|dyhat|/scale {med:.4f} ({'PASS' if med >= 0.05 else 'FAIL'} "
              f">=0.05)")
        print(f"  H2 per-model verdict: "
              f"{'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")

        # ---------- EXPLORATORY redesign (labeled) ----------
        expl = {}
        fo4, med4, _ = crit(steer_trials(pipe, ev, layer, d_raw, 4 * alpha))
        expl["4x_alpha"] = (fo4, med4)
        multi = []
        for values, _s in ev:
            y0, s = yhat(pipe, values)
            for sgn in (+1, -1):
                hooks = []
                for l in range(cfg.num_layers):
                    hooks += add_direction_hooks(pipe, l, d_raw,
                                                 sgn * alpha
                                                 / cfg.num_layers ** 0.5)
                y1, _ = yhat(pipe, values, hooks)
                move = (y1 - y0) / s
                multi.append((move * sgn > 0, abs(move)))
        fom, medm, _ = crit(multi)
        expl["all_layers"] = (fom, medm)
        print(f"  [EXPLORATORY] 4x alpha: correct {fo4:.2f}, median {med4:.4f} "
              f"| all-layers: correct {fom:.2f}, median {medm:.4f}")

        results[name] = dict(
            layer=layer, loo=loo, chosen_f=chosen_f, alpha=alpha,
            selection_stats={str(k): v for k, v in sel_stats.items()},
            probe_acc=acc, steer_correct=fo, steer_median=med,
            confirmed=bool(confirmed), exploratory=expl)

    general = len(confirmed_models) >= 3
    print(f"\nH2-GENERAL: {len(confirmed_models)}/4 confirmed "
          f"({confirmed_models}) -> "
          f"{'CONFIRMED' if general else 'NOT CONFIRMED'} (needs >=3)")

    blob = dict(timestamp=stamp, confirmed_models=confirmed_models,
                h2_general=bool(general), models=results)
    out = Path(args.out) / "confirmatory-h2.json"
    out.write_text(json.dumps(blob, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
