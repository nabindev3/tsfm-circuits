"""Descriptive circuit inventory (Weeks 2-3): ranked candidate lists, per
phenomenon, per model — the leads to test causally in Stage 3.

EXPLORATORY ONLY. All series seeds are < 100 as required by PREREGISTRATION.md;
nothing here touches the confirmatory seed range 100-119.

Per model:
  1. SEASONAL HEADS — seasonal_attention_score on period-{7,12,24} series,
     reported vs the uniform null AND vs a phase-scrambled control (the same
     values in shuffled order: identical tokens/scale, no temporal structure —
     kills spurious "periodicity" a head could fake from value statistics).
     Candidate rule: >= 3x null at >= 2 periods AND >= 2x its scrambled score.
  2. TREND DIRECTION — L1 logistic probe on mean-pooled cache.resid[layer] for
     slope sign + ridge R^2 for signed slope. Series carry a random positive
     level offset so the tokenizer's mean-scaling makes the pooled token value
     ~identical for both classes — the probe can't cheat on the marginal mean.
     The accuracy-peak layer is the candidate trend subspace.
  3. CHANGEPOINT COMPONENTS — on stationary vs season_plus_changepoint pairs:
     heads whose attention mass collapses onto post-changepoint keys (paired
     difference vs the stationary control), and the per-layer residual-stream
     delta-norm spike at the changepoint.

    python inventory.py --device mps                  # all four scales
    python inventory.py --device mps --models amazon/chronos-t5-small

Writes results/inventory-<model>.json and prints ranked candidates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from attention_analysis import head_scores
from chronos_harness import get_inner, load_pipeline, run_with_cache
from synthetic import changepoint_pair, seasonal, trend
from verify_harness import STUDY_MODELS

LENGTH = 168          # integer multiple of 7, 12, and 24
NOISE = 0.05
SEEDS = range(5)      # exploratory (< 100)
PERIODS = (7, 12, 24)


# ---------------------------------------------------------------------------
# 1. seasonal heads
# ---------------------------------------------------------------------------


def seasonal_inventory(pipe) -> dict:
    real, scrambled = {}, {}
    for period in PERIODS:
        rs, ss = [], []
        for seed in SEEDS:
            s = seasonal(period=period, length=LENGTH, kind="pattern",
                         noise=NOISE, seed=seed)
            cache = run_with_cache(pipe, s.values)
            rs.append(head_scores(cache.attn, period, valid_len=cache.valid_len))
            perm = np.random.default_rng(seed + 50_000).permutation(s.values)
            cache_s = run_with_cache(pipe, perm)
            ss.append(head_scores(cache_s.attn, period,
                                  valid_len=cache_s.valid_len))
        real[period] = np.mean(rs, axis=0)
        scrambled[period] = np.mean(ss, axis=0)

    gm = np.exp(np.mean([np.log(np.maximum(real[p], 1e-9)) for p in PERIODS],
                        axis=0))
    n_layers, n_heads = gm.shape
    candidates = []
    for l in range(n_layers):
        for h in range(n_heads):
            strong = [p for p in PERIODS
                      if real[p][l, h] > 3.0
                      and real[p][l, h] > 2.0 * scrambled[p][l, h]]
            if len(strong) >= 2:
                candidates.append((l, h))
    return dict(real=real, scrambled=scrambled, gm=gm, candidates=candidates)


def report_seasonal(res: dict, top: int = 10) -> None:
    order = np.argsort(res["gm"], axis=None)[::-1][:top]
    print("  [seasonal] head: ratio-vs-null @ P7 | P12 | P24 "
          "(scrambled control in parens), gm = geometric mean")
    for flat in order:
        l, h = np.unravel_index(flat, res["gm"].shape)
        cells = " | ".join(
            f"{res['real'][p][l, h]:5.2f} ({res['scrambled'][p][l, h]:4.2f})"
            for p in PERIODS)
        mark = "  CANDIDATE" if (int(l), int(h)) in res["candidates"] else ""
        print(f"    L{l}H{h:<3} {cells}  gm={res['gm'][l, h]:5.2f}{mark}")
    print(f"  candidates (>=3x null & >=2x scrambled at >=2 periods): "
          f"{['L%dH%d' % c for c in res['candidates']]}")


# ---------------------------------------------------------------------------
# 2. trend direction probe
# ---------------------------------------------------------------------------


def trend_probe(pipe, n_per_class: int = 40) -> dict:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats, signs, slopes = [], [], []
    for i in range(2 * n_per_class):           # series seeds 0..79, all < 100
        rng = np.random.default_rng(i)
        sign = 1 if i < n_per_class else -1
        mag = 10 ** rng.uniform(np.log10(0.002), np.log10(0.012))
        level = rng.uniform(3.0, 5.0)          # keeps values positive: the
        slope = sign * mag                     # mean-scaled pooled token is ~1
        s = trend(slope=slope, length=LENGTH, level=level, noise=NOISE, seed=i)
        cache = run_with_cache(pipe, s.values)
        feats.append(np.stack([r[:cache.valid_len].mean(0).numpy()
                               for r in cache.resid]))  # [layers, d_model]
        signs.append(sign)
        slopes.append(slope)
    X = np.stack(feats)                        # [n, layers, d]
    y_sign, y_slope = np.array(signs), np.array(slopes)

    acc, r2 = [], []
    for layer in range(X.shape[1]):
        Xl = X[:, layer, :]
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0, max_iter=2000))
        acc.append(float(cross_val_score(clf, Xl, y_sign, cv=5).mean()))
        reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        r2.append(float(cross_val_score(reg, Xl, y_slope, cv=5,
                                        scoring="r2").mean()))
    return dict(acc=acc, r2=r2, peak_layer=int(np.argmax(acc)),
                peak_acc=float(max(acc)), peak_r2_layer=int(np.argmax(r2)),
                peak_r2=float(max(r2)))


def report_trend(res: dict) -> None:
    print("  [trend] L1-logistic sign accuracy by layer (5-fold CV):")
    print("    " + " ".join(f"L{i}:{a:.2f}" for i, a in enumerate(res["acc"])))
    print("    ridge R^2 (signed slope): "
          + " ".join(f"L{i}:{r:+.2f}" for i, r in enumerate(res["r2"])))
    print(f"    candidate trend subspace: layer {res['peak_layer']} "
          f"(acc {res['peak_acc']:.2f}); magnitude peaks at layer "
          f"{res['peak_r2_layer']} (R^2 {res['peak_r2']:+.2f})")


# ---------------------------------------------------------------------------
# 3. changepoint components
# ---------------------------------------------------------------------------


def changepoint_inventory(pipe, period: int = 7) -> dict:
    collapse_list, spike_list, peaks = [], [], []
    cp_at = None
    for seed in SEEDS:
        pair = changepoint_pair(period=period, length=LENGTH, cp_kind="pattern",
                                noise=NOISE, seed=seed)
        cp_at = pair.corrupted.meta["cp_at"]
        c_st = run_with_cache(pipe, pair.clean.values)      # stationary
        c_cp = run_with_cache(pipe, pair.corrupted.values)  # with changepoint

        # heads whose post-CP queries collapse onto post-CP keys, minus the
        # same statistic on the stationary control
        v = c_cp.valid_len
        q = slice(cp_at + 2, v)
        per_head = []
        for a_cp, a_st in zip(c_cp.attn, c_st.attn):
            m_cp = a_cp[:, q, cp_at:].sum(-1).mean(-1)  # [heads]
            m_st = a_st[:, q, cp_at:].sum(-1).mean(-1)
            per_head.append((m_cp - m_st).numpy())
        collapse_list.append(np.stack(per_head))            # [layers, heads]

        # residual-stream delta-norm profile per layer; spike near the CP
        spikes, pk = [], []
        for r_cp, r_st in zip(c_cp.resid, c_st.resid):
            d = (r_cp - r_st).norm(dim=-1).numpy()          # [L]
            background = np.median(d[5:cp_at - 5])
            window = d[cp_at - 2:cp_at + period + 1]
            spikes.append(float(window.max() / (background + 1e-9)))
            pk.append(int(np.argmax(d)))
        spike_list.append(spikes)
        peaks.append(pk)

    return dict(collapse=np.mean(collapse_list, axis=0),
                spike=np.mean(spike_list, axis=0),
                peak_pos=np.median(peaks, axis=0).astype(int),
                cp_at=cp_at)


def report_changepoint(res: dict, top: int = 5) -> None:
    order = np.argsort(res["collapse"], axis=None)[::-1][:top]
    heads = []
    for flat in order:
        l, h = np.unravel_index(flat, res["collapse"].shape)
        heads.append(f"L{l}H{h} {res['collapse'][l, h]:+.3f}")
    print(f"  [changepoint] cp_at={res['cp_at']}; top post-CP attention-collapse "
          f"heads (delta mass vs stationary control):")
    print("    " + " | ".join(heads))
    best = int(np.argmax(res["spike"]))
    print("    resid delta-norm spike ratio by layer: "
          + " ".join(f"L{i}:{s:.1f}" for i, s in enumerate(res["spike"])))
    print(f"    strongest spike: layer {best} ({res['spike'][best]:.1f}x "
          f"background, peak at position {res['peak_pos'][best]})")


# ---------------------------------------------------------------------------


def run_model(model_id: str, device: str | None, out_dir: Path) -> dict:
    t0 = time.time()
    pipe = load_pipeline(model_id, device)
    cfg = get_inner(pipe).config
    print(f"\n=== {model_id} ({cfg.num_layers}L x {cfg.num_heads}H) ===")

    sea = seasonal_inventory(pipe)
    report_seasonal(sea)
    tre = trend_probe(pipe)
    report_trend(tre)
    cpt = changepoint_inventory(pipe)
    report_changepoint(cpt)
    print(f"  ({time.time() - t0:.0f}s)")

    blob = dict(
        model=model_id, layers=cfg.num_layers, heads=cfg.num_heads,
        length=LENGTH, noise=NOISE, seeds=list(SEEDS), periods=list(PERIODS),
        seasonal=dict(
            real={str(p): sea["real"][p].tolist() for p in PERIODS},
            scrambled={str(p): sea["scrambled"][p].tolist() for p in PERIODS},
            gm=sea["gm"].tolist(),
            candidates=[list(c) for c in sea["candidates"]]),
        trend=tre,
        changepoint=dict(collapse=cpt["collapse"].tolist(),
                         spike=cpt["spike"].tolist(),
                         peak_pos=cpt["peak_pos"].tolist(),
                         cp_at=cpt["cp_at"]),
    )
    out = out_dir / f"inventory-{model_id.split('/')[-1]}.json"
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
    out_dir.mkdir(exist_ok=True)

    blobs = [run_model(m, args.device, out_dir) for m in args.models]

    if len(blobs) > 1:
        print("\n=== cross-scale summary (exploratory) ===")
        for b in blobs:
            cands = ["L%dH%d" % tuple(c) for c in b["seasonal"]["candidates"]]
            col = np.array(b["changepoint"]["collapse"])
            l, h = np.unravel_index(int(col.argmax()), col.shape)
            print(f"  {b['model'].split('/')[-1]:8s} seasonal: "
                  f"{', '.join(cands) if cands else 'NONE'} | trend peak "
                  f"L{b['trend']['peak_layer']} ({b['trend']['peak_acc']:.2f}) | "
                  f"CP head L{l}H{h} ({col.max():+.3f}), spike "
                  f"{max(b['changepoint']['spike']):.1f}x")


if __name__ == "__main__":
    main()
