# Pre-registration: Do Forecasting Models Reinvent Induction Heads?

**Frozen 2026-07-05.** This file is confirmatory-analysis law: after the commit that
introduces it, it is never edited. Deviations go in `DEVIATIONS.md` with dates, and
anything not specified here is exploratory and will be labeled as such.

## Disclosure of prior exploration

RQ1 was explored on `chronos-t5-small` before this document (seeds 0–2, periods 7 vs 12,
length 140, pattern seasonality): lag-tracking heads L4H1/L4H7/L5H7 were found, and
group-patching them recovered ~22% of the forecast gap. **Those numbers are exploratory
and will not be reported as confirmatory.** All seeds < 100 are exploratory forever.
Confirmatory analyses use seeds 100–119, untouched as of freezing.

## Fixed apparatus

Models: `amazon/chronos-t5-{mini,small,base,large}` — all four passed
`verify_harness.py` (exact-recovery patch effect 1.0000, identity-patch no-op, reshape
unit test) before freezing. Data: `synthetic.py` generators, pattern seasonality,
length 140, noise 0.05, amplitude 1.0. Metrics: (m1) seasonal attention ratio vs
uniform bidirectional null, tol=1, EOS excluded; (m2) deterministic first-step forecast
= softmax(value-token logits) · bin centers · scale; (m3) patching effect = normalized
recovery (patched − corrupted)/(clean − corrupted); (m4) 95% percentile-bootstrap CIs
over seeds, 10,000 resamples. Periods: {7, 12} confirmatory; {5, 10} robustness
(exploratory).

## Hypotheses, thresholds, decision rules

**H1 — seasonal induction heads exist and are causal.** Per model:
(a) *descriptive:* ≥ 1 head has attention ratio > 3× null at BOTH periods, averaged over
selection seeds 100–104; (b) *causal:* the group of all such heads (capped at 10% of
heads, keeping the highest-min-ratio ones) patched clean→corrupted on period-7↔12 pairs,
evaluated on held-out seeds 105–119, gives mean recovery ≥ 0.15 with CI lower bound
> 0.05, while a size-matched random control group (same layers excluded) gives mean
< 0.05. Head selection uses seeds 100–104 only; evaluation uses 105–119 only.
**H1-general:** holds in ≥ 3 of 4 models, including at least one of {base, large}.

**H2 — trend is a steerable direction.** Per model: a logistic probe on mean-pooled
`cache.resid` (layer and steering coefficient α chosen on seeds 100–104) separates
up/down `trend_pair`s with ≥ 90% accuracy on seeds 105–119; adding ±α·direction moves
the first-step forecast in the predicted direction for ≥ 80% of held-out series with
median |Δŷ|/scale ≥ 0.05. **H2-general:** ≥ 3 of 4 models.

**H3 — double dissociation.** Degradation D = increase in scale-normalized first-step
absolute error on held-out seeds. Mean-ablating the H1 head group must give
D(seasonal task) ≥ 2× D(trend task), AND zero-projecting the H2 direction must give
D(trend task) ≥ 2× D(seasonal task), each ratio with bootstrap CI lower bound > 1.

## Stopping rules

Seeds, models, thresholds, and periods are fixed above and do not move after any
confirmatory result is seen. No seeds are added after unblinding. A hypothesis that
misses its threshold is reported as a null at full precision — no post-hoc threshold,
group-size, or metric changes. If a result is threshold-adjacent, it is reported as-is;
"suggestive" language is allowed only for exploratory analyses.

## Generality beyond the family & payoff

Secondary (harness not yet built, best-effort): replicate the H1 pipeline on one
non-Chronos, decoder-only forecaster (TimesFM or Lag-Llama). Payoff test if H1+H3 hold:
head-group ablation should *predict* targeted failure — seasonal degradation ≥ 2× trend
degradation on series types chosen in advance. If H1 fails at base/large, the clean null
is: "seasonal copying is not head-localized at scale," reported with all-head-patch
upper bounds from `verify_harness.py`.
