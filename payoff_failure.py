"""Stage 5b — the payoff: circuit-informed failure prediction on REAL data.

Claim under test: low seasonal-head activation predicts seasonality
mis-forecasting. On ETTh1 (hourly; daily period 24; the same dataset and MASE
convention as the fm-difficulty pipeline), for each context window we compute
  * activation: mean attention ratio at lag 24 of the CONFIRMED chronos-t5-small
    head group (L5H7/L4H1/L4H7 from the pre-registered H1 selection), and
  * error: seasonal MASE of the median forecast over a 24-step horizon
    (num_samples=20, fixed torch seed).
Then: Spearman rank correlation, selective prediction (mean MASE kept at
decreasing coverage, ranked by activation), and AUC for flagging worst-quartile
windows from low activation.

    python payoff_failure.py --device mps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from attention_analysis import seasonal_attention_score
from chronos_harness import load_pipeline, run_with_cache
from confirmatory_h1 import select_heads

URL = ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
       "ETTh1.csv")
MODEL = "amazon/chronos-t5-small"
CONTEXT = 168            # 7 days of hourly data
HORIZON = 24             # forecast one day
PERIOD = 24
N_WINDOWS = 150
NUM_SAMPLES = 20


def load_series() -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(URL)
    return df["OT"].values.astype(np.float64)


def mase(forecast: np.ndarray, truth: np.ndarray, context: np.ndarray,
         season: int = PERIOD) -> float:
    mae = float(np.mean(np.abs(forecast - truth)))
    naive = np.abs(context[season:] - context[:-season])
    denom = float(np.mean(naive))
    return mae / max(denom, 1e-5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ts = load_series()
    pipe = load_pipeline(MODEL, args.device)
    heads, _ = select_heads(pipe)
    print(f"{MODEL} confirmed heads {['L%dH%d' % h for h in heads]}; "
          f"ETTh1 OT: {len(ts)} hours, {N_WINDOWS} windows "
          f"(context {CONTEXT}, horizon {HORIZON}, period {PERIOD})")

    starts = np.linspace(0, len(ts) - CONTEXT - HORIZON - 1, N_WINDOWS,
                         dtype=int)
    acts, errs = [], []
    torch.manual_seed(0)
    for w, s0 in enumerate(starts):
        context = ts[s0:s0 + CONTEXT]
        truth = ts[s0 + CONTEXT:s0 + CONTEXT + HORIZON]

        cache = run_with_cache(pipe, context)
        act = float(np.mean([
            seasonal_attention_score(cache.attn[l][h], PERIOD, tol=1,
                                     valid_len=cache.valid_len).ratio
            for l, h in heads]))

        ctx_t = torch.as_tensor(context, dtype=torch.float32).unsqueeze(0)
        samples = pipe.predict(ctx_t, prediction_length=HORIZON,
                               num_samples=NUM_SAMPLES)
        forecast = samples[0].float().median(dim=0).values.numpy()

        acts.append(act)
        errs.append(mase(forecast, truth, context))

    acts, errs = np.array(acts), np.array(errs)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(acts, errs)
    print(f"\nSpearman(activation, MASE) = {rho:+.3f} (p = {pval:.2g}) — "
          f"negative = low activation predicts high error")

    order = np.argsort(-acts)   # keep highest-activation windows first
    print("selective prediction (keep highest-activation windows):")
    coverage_rows = {}
    for cov in (1.0, 0.75, 0.5, 0.25):
        keep = order[: int(cov * len(errs))]
        coverage_rows[cov] = float(errs[keep].mean())
        print(f"  coverage {cov:4.0%}: mean MASE {errs[keep].mean():.3f}")
    improvement = 1 - coverage_rows[0.25] / coverage_rows[1.0]
    print(f"  -> filtering to the top-25% activation windows cuts MASE by "
          f"{improvement:.0%}")

    worst = errs >= np.quantile(errs, 0.75)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(worst, -acts))
    print(f"AUC (flag worst-quartile MASE from LOW activation) = {auc:.3f}")

    Path("results/payoff-failure.json").write_text(json.dumps(dict(
        model=MODEL, heads=[list(h) for h in heads], dataset="ETTh1/OT",
        context=CONTEXT, horizon=HORIZON, period=PERIOD,
        activations=acts.tolist(), mase=errs.tolist(),
        spearman=[float(rho), float(pval)], auc=auc,
        coverage_mase={str(k): v for k, v in coverage_rows.items()}),
        indent=1))
    print("wrote results/payoff-failure.json")


if __name__ == "__main__":
    main()
