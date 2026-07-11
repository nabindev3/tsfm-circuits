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
    acts, mases, maes_normalized, naive_denoms_normalized = [], [], [], []
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

        # Calculate MAE
        mae = float(np.mean(np.abs(forecast - truth)))

        # Calculate MASE denominator (seasonal-naive error)
        naive = np.abs(context[PERIOD:] - context[:-PERIOD])
        denom = float(np.mean(naive))

        acts.append(act)
        mases.append(mae / max(denom, 1e-5))
        maes_normalized.append(mae / cache.scale)
        naive_denoms_normalized.append(denom / cache.scale)

    acts = np.array(acts)
    mases = np.array(mases)
    maes_normalized = np.array(maes_normalized)
    naive_denoms_normalized = np.array(naive_denoms_normalized)

    from scipy.stats import spearmanr
    rho_mase, pval_mase = spearmanr(acts, mases)
    rho_mae, pval_mae = spearmanr(acts, maes_normalized)
    rho_naive, pval_naive = spearmanr(acts, naive_denoms_normalized)

    print(f"\nSpearman(activation, MASE) = {rho_mase:+.3f} (p = {pval_mase:.2g}) — "
          f"positive correlation reverses expectation due to denominator scaling")
    print(f"Spearman(activation, MAE/scale) = {rho_mae:+.3f} (p = {pval_mae:.2g}) — "
          f"negative correlation means low activation predicts high error")
    print(f"Spearman(activation, seasonal-naive predictability) = {rho_naive:+.3f} (p = {pval_naive:.2g}) — "
          f"strong seasonality meter")

    order = np.argsort(-acts)   # keep highest-activation windows first
    print("\nselective prediction (keep highest-activation windows, metric=MAE/scale):")
    coverage_rows_mae = {}
    for cov in (1.0, 0.75, 0.5, 0.25):
        keep = order[: int(cov * len(maes_normalized))]
        coverage_rows_mae[cov] = float(maes_normalized[keep].mean())
        print(f"  coverage {cov:4.0%}: mean MAE/scale {maes_normalized[keep].mean():.3f}")
    improvement = 1 - coverage_rows_mae[0.25] / coverage_rows_mae[1.0]
    print(f"  -> filtering to the top-25% activation windows cuts MAE/scale by "
          f"{improvement:.0%}")

    worst_mae = maes_normalized >= np.quantile(maes_normalized, 0.75)
    from sklearn.metrics import roc_auc_score
    auc_mae = float(roc_auc_score(worst_mae, -acts))
    print(f"AUC (flag worst-quartile MAE/scale from LOW activation) = {auc_mae:.3f}")

    Path("results/payoff-failure.json").write_text(json.dumps(dict(
        model=MODEL, heads=[list(h) for h in heads], dataset="ETTh1/OT",
        context=CONTEXT, horizon=HORIZON, period=PERIOD,
        activations=acts.tolist(), mase=mases.tolist(),
        maes_normalized=maes_normalized.tolist(),
        naive_denoms_normalized=naive_denoms_normalized.tolist(),
        spearman_mase=[float(rho_mase), float(pval_mase)],
        spearman_mae_normalized=[float(rho_mae), float(pval_mae)],
        spearman_naive=[float(rho_naive), float(pval_naive)],
        auc_mae=auc_mae,
        coverage_mae_normalized={str(k): v for k, v in coverage_rows_mae.items()}),
        indent=1))
    print("wrote results/payoff-failure.json")


if __name__ == "__main__":
    main()

