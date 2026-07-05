"""Seasonal-attention scoring (find induction-style heads) + patching-effect math.

The seasonal attention score is the forecasting analog of the induction-head score:
for every query position i, how much attention mass lands on key position i - P
(the same phase one cycle back)? A head implementing "copy the token one period
ago" concentrates there; the null is uniform attention over all keys (the Chronos
encoder is bidirectional, so the null is (2*tol+1)/L, not the causal 1/(i+1)).

Pure math on arrays — no model dependency. Run directly to self-test:
    python attention_analysis.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeasonalScore:
    score: float  # mean attention mass in the lag window
    null: float   # expected mass under uniform attention
    ratio: float  # score / null  (the headline number)


def _to_numpy(attn) -> np.ndarray:
    if hasattr(attn, "detach"):
        attn = attn.detach().cpu().float().numpy()
    return np.asarray(attn, dtype=np.float64)


def seasonal_attention_score(attn, period: int, tol: int = 1,
                             valid_len: int | None = None) -> SeasonalScore:
    """attn: [L, L] attention weights (rows = queries, sum to 1 over keys).

    For each query i in [period+tol, valid_len), sum the mass on keys within
    [i-period-tol, i-period+tol], then average over queries. Queries start at
    period+tol so the window never clips at key 0 (which would bias the score
    below the null). valid_len lets the caller exclude trailing special tokens
    (Chronos appends EOS)."""
    a = _to_numpy(attn)
    assert a.ndim == 2 and a.shape[0] == a.shape[1], "expected a single [L, L] head"
    L = a.shape[0]
    if valid_len is None:
        valid_len = L
    queries = range(period + tol, valid_len)
    masses = []
    for i in queries:
        masses.append(a[i, i - period - tol: i - period + tol + 1].sum())
    if not masses:
        raise ValueError("no valid queries: increase series length or lower period")
    score = float(np.mean(masses))
    null = (2 * tol + 1) / L
    return SeasonalScore(score=score, null=null, ratio=score / null)


def head_scores(attn_per_layer, period: int, tol: int = 1,
                valid_len: int | None = None) -> np.ndarray:
    """attn_per_layer: list over layers of [heads, L, L] (Cache.attn from the
    harness). Returns ratios as [layers, heads]."""
    out = []
    for layer_attn in attn_per_layer:
        a = _to_numpy(layer_attn)
        assert a.ndim == 3, "expected [heads, L, L] per layer"
        out.append([seasonal_attention_score(a[h], period, tol, valid_len).ratio
                    for h in range(a.shape[0])])
    return np.array(out)


def rank_heads(ratios: np.ndarray):
    """[(layer, head, ratio)] sorted by descending ratio."""
    order = np.argsort(ratios, axis=None)[::-1]
    ls, hs = np.unravel_index(order, ratios.shape)
    return [(int(l), int(h), float(ratios[l, h])) for l, h in zip(ls, hs)]


def patching_effect(clean: float, corrupted: float, patched: float,
                    eps: float = 1e-9) -> float:
    """Normalized recovery: 0 = patch did nothing, 1 = patch fully restored the
    clean behavior. Standard denominator guard for degenerate pairs."""
    denom = clean - corrupted
    if abs(denom) < eps:
        return float("nan")
    return (patched - corrupted) / denom


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile-bootstrap CI for the mean over independent runs (seeds/series).
    Returns (mean, lo, hi). This is the CI reported for all patching effects."""
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2:
        return float(x.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(x.mean()), float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


# ---------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------


def _perfect_head(L: int, period: int) -> np.ndarray:
    a = np.full((L, L), 1.0 / L)
    for i in range(period, L):
        a[i] = 0.0
        a[i, i - period] = 1.0
    return a


def _self_test() -> None:
    L, P = 64, 7

    perfect = seasonal_attention_score(_perfect_head(L, P), P, tol=1)
    assert perfect.score > 0.999, perfect
    assert perfect.ratio > 20, perfect

    uniform = seasonal_attention_score(np.full((L, L), 1.0 / L), P, tol=1)
    assert abs(uniform.score - uniform.null) < 1e-12
    assert abs(uniform.ratio - 1.0) < 1e-12

    # random softmax attention should hover around the null
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(L, L))
    rand = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    r = seasonal_attention_score(rand, P, tol=1)
    assert 0.5 < r.ratio < 2.0, r

    # tol=0 tightens the window
    exact = seasonal_attention_score(_perfect_head(L, P), P, tol=0)
    assert exact.score > 0.999 and abs(exact.null - 1 / L) < 1e-12

    # valid_len excludes trailing rows (e.g. the EOS token)
    a = _perfect_head(L, P)
    a[-1] = 1.0 / L  # a junk EOS row
    with_eos = seasonal_attention_score(a, P, tol=1, valid_len=L)
    without = seasonal_attention_score(a, P, tol=1, valid_len=L - 1)
    assert without.score > with_eos.score

    # multi-head ranking
    layer = np.stack([rand, _perfect_head(L, P), np.full((L, L), 1.0 / L)])
    ratios = head_scores([layer, layer], P)
    assert ratios.shape == (2, 3)
    top = rank_heads(ratios)[0]
    assert top[1] == 1, top  # the perfect head wins

    # patching effect algebra
    assert patching_effect(10.0, 2.0, 10.0) == 1.0
    assert patching_effect(10.0, 2.0, 2.0) == 0.0
    assert patching_effect(10.0, 2.0, 6.0) == 0.5
    assert np.isnan(patching_effect(5.0, 5.0, 5.0))

    # bootstrap CI: covers the true mean, degenerate on constants, sane coverage
    m, lo, hi = bootstrap_ci([3.0, 3.0, 3.0, 3.0])
    assert m == lo == hi == 3.0
    sample = rng.normal(loc=1.0, scale=1.0, size=50)
    m, lo, hi = bootstrap_ci(sample)
    assert lo < m < hi and lo < 1.0 < hi
    covered = 0
    for trial in range(200):
        s = np.random.default_rng(trial).normal(0.0, 1.0, size=30)
        _, lo, hi = bootstrap_ci(s, n_boot=1000, seed=trial)
        covered += lo <= 0.0 <= hi
    assert covered >= 175, f"bootstrap coverage too low: {covered}/200"

    print("attention_analysis.py: all self-tests passed")
    print(f"  perfect head: score={perfect.score:.3f} ratio={perfect.ratio:.1f}x | "
          f"uniform null: score={uniform.score:.3f} ratio={uniform.ratio:.2f}x")


if __name__ == "__main__":
    _self_test()
