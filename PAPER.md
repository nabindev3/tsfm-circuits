# Do Forecasting Models Reinvent Induction Heads? A Pre-Registered Circuit Analysis of Chronos

**Nabin Prasad Dev**

## Abstract

Value-bin tokenization makes a forecaster a language model — and it makes the
model grow induction heads. We present a pre-registered, causally validated
circuit analysis of seasonality, trend, and changepoints in the Chronos-T5
family (20M–710M). Chronos quantizes a series into discrete value bins and
runs a T5 over the tokens, so the induction-head question from language-model
interpretability transfers directly: seasonality is structurally an induction
task ("the token one period ago predicts the next token"). We find seasonal
induction heads at every scale, confirm them causally with minimal-pair
activation patching under a frozen pre-registered protocol (46M model: group
recovery +0.28, 95% CI [+0.17, +0.43], vs +0.004 for matched controls), and
characterize their algorithm: the period is detected from the input, not
stored — on a mid-sequence period switch the heads re-lock to the new lag
within 5 steps, under half a cycle. The circuit consolidates with scale (a
single head recovers +0.40 of the behavioral gap at 200M) and then dissolves
into redundancy at 710M (best head +0.02; group intact at +0.27) — no
LLM-style phase transition. Two matched negative results triangulate the
cause: Chronos-Bolt, sharing the *identical* T5 backbone but patch-based I/O,
has zero such heads, and so does patch-based TimesFM — the motif is a property
of LM-style tokenization, not of transformer forecasters. The mechanism pays
its way on real data: low activation of the confirmed heads predicts
mis-forecasting on ETTh1 (ρ = −0.38, p = 2×10⁻⁶; filtering to top-quartile
activation cuts error 38%). We report all pre-registered verdicts as frozen,
including two informative failures: trend is linearly decodable from every
layer yet not steerable via any single direction, and MASE-based evaluation
reverses the failure-prediction sign because its seasonal-naive denominator
confounds seasonality strength.

## 1. Introduction

Time-series foundation models (TSFMs) increasingly borrow the machinery of
language models. Chronos [Ansari et al., 2024] takes the borrowing to its
logical end: it mean-scales a series, quantizes values into 4,096 discrete
bins, and trains an off-the-shelf T5 to do next-token prediction over the bin
vocabulary. This design invites a precise interpretability question. In LLMs,
*induction heads* — attention heads that find the previous occurrence of the
current token and copy what followed it — are the best-understood circuit
motif [Elhage et al., 2021; Olsson et al., 2022]. A seasonal series is exactly
an induction problem in Chronos's token space: position *t* is predicted by
position *t − P*. **Do forecasting models reinvent induction heads?**

The question has a sharp scientific payoff either way. If the heads exist and
are causal, mechanistic interpretability's core motif transfers across
modalities, and the tokenized-LM framing of forecasting acquires mechanical
content. If they do not, seasonality must be computed some other way, and the
LM analogy is cosmetic. We answer with an unusual degree of rigor for circuit
work: a frozen, timestamped pre-registration written before any confirmatory
run, exploratory/confirmatory seed separation, two null baselines for every
descriptive claim, causal validation by intervention, ablation-based
dissociation, four model scales, two contrast families, and a real-data
capability demonstration. Three of our four pre-registered hypotheses
*failed*, and each failure is informative; we report all of them at full
precision.

**Contributions.** (1) The first causally validated circuit account of
seasonality in a TSFM: seasonal induction heads exist at all four Chronos-T5
scales and are confirmed by pre-registered minimal-pair patching at 46M.
(2) A mechanism: the heads implement *input-detected* period copying —
they re-lock to a new period within 5 steps of a mid-sequence switch — and
transmit abstract periodicity rather than pattern values (patching from a
*different* series with the same period restores behavior as well as the
matched series). (3) A scaling story that is not a phase transition:
attention sharpens and causal load first consolidates into single heads
(200M) and then dissolves into redundant groups (710M). (4) A controlled
tokenization result: the identical T5 backbone with patch-based I/O
(Chronos-Bolt) has no such heads, and neither does TimesFM — the motif comes
from the value-bin LM design. (5) A working payoff: circuit activation
predicts real-data forecast failure and doubles as a seasonality meter, with
a cautionary decomposition showing MASE's seasonal-naive denominator reverses
the naive analysis. (6) Methodological findings for the field: attention-ratio
selection *without* a phase-scrambled control fails at scale (it selects
value-matching impostors that flunk causal tests — the exact failure our
frozen selection rule exhibited); single-position patching understates
per-component effects by an order of magnitude in encoder–decoder
forecasters; and ratio-based dissociation criteria are ill-posed when the
spared task's degradation is ~0.

## 2. Background

**Chronos.** Chronos-T5 [Ansari et al., 2024] maps a context window to tokens
by mean-scaling (s = mean |x|) and uniform binning over [−15, 15] in scaled
space (4,093 value bins + special tokens), then runs an encoder–decoder T5;
forecasts are sampled autoregressively from the decoder. We study the four
public scales: mini (20M, 4L×8H), small (46M, 6L×8H), base (200M, 12L×12H),
large (710M, 24L×16H). Chronos-Bolt shares the T5 backbone but replaces
tokenization with 16-step patch embeddings and the LM head with direct
multi-quantile regression. TimesFM [Das et al., 2024] is a decoder-only
patch-based forecaster. Together they let us hold architecture fixed and vary
tokenization (Bolt), and vary architecture family entirely (TimesFM).

**Induction heads and activation patching.** Induction heads implement
[A][B] … [A] → [B] via a match-and-copy attention pattern [Elhage et al.,
2021; Olsson et al., 2022]. Causal attribution uses activation patching on
minimal pairs [Vig et al., 2020; Meng et al., 2022; Wang et al., 2023]:
replace one component's activation in a corrupted run with its clean-run
value and measure recovery of clean behavior. Path patching [Wang et al.,
2023; Goldowsky-Dill et al., 2023] restricts the intervention to specific
sender→receiver routes.

## 3. Methods: one shared pipeline

All experiments share one harness (this repository; `reproduce.sh` re-runs
everything with fixed seeds and logs).

**Synthetic minimal pairs.** Controlled generators produce seasonal series
(a random per-cycle pattern tiled — the sharpest induction test, since only
copying position *t − P* predicts position *t*), trends, changepoints, and
compositions. Minimal pairs differ in exactly one causal factor (period,
phase, trend-on/off, changepoint location), share noise realizations, and are
constructed to diverge at the first forecast step (guaranteed divergence
≥ 0.5 amplitude).

**Descriptive scoring.** The *seasonal attention score* of a head is its mean
attention mass in a ±1 window at lag P, over queries where the window is
in-range. It is compared to two nulls: the uniform null ((2·tol+1)/L for the
bidirectional encoder; per-query causal null for decoder-only models) and a
*phase-scrambled control* — the same values in shuffled order, which preserves
value statistics while destroying temporal structure. Candidates must exceed
3× the uniform null at ≥2 periods *and* 2× their scrambled score. Section 5
shows the scrambled control is not optional.

**Deterministic behavioral metrics.** Sampling-free forecasts: the expected
value of the first-step decoder distribution (softmax over value-token logits
· bin centers · scale), the logit difference between the clean-correct and
corrupted-correct bins, and closed-form CRPS from the discrete CDF.
Patching effect is normalized recovery (patched − corrupted)/(clean −
corrupted). CIs are percentile bootstraps (B = 10,000) over seeds, paired by
construction; group comparisons use label-permutation tests (B = 10,000).

**Interventions.** Head patching splices at the input of each attention's
output projection, where the tensor is still [batch, L, heads × d_head];
`patch_heads` splices arbitrary (layer, head) groups, optionally restricted
to positions; mean-ablation replaces a head's per-position outputs with its
positional mean; direction interventions add or project out unit vectors in
the residual stream. Harness validation on every model: an identity patch is
a numerical no-op; patching the full encoder output reproduces clean logits
exactly (effect 1.0000); patching *all heads plus token embeddings* — after
which every encoder quantity is clean by induction — yields effect 1.000 ±
0.0002 on all four scales.

**Pre-registration.** `PREREGISTRATION.md` was frozen in a timestamped commit
before any confirmatory run, declaring: prior exploration (seeds < 100,
disclosed), confirmatory seeds 100–119 split into selection (100–104) and
held-out evaluation (105–119), hypotheses H1–H3 with numeric thresholds,
metrics, stopping rules, and a no-edit rule (deviations go to
`DEVIATIONS.md`; one occurred, documented before the affected evaluation ran).

## 4. Descriptive circuit inventory (exploratory)

Across periods {7, 12, 24} with 10 seeds and both nulls: lag-tracking
candidates exist at every scale and concentrate at ~⅔ depth — mini 1 (L2H2),
small 3 (L1H7, L4H7, L5H7), base 12 (L8–L11), large 15 (L11–L21). The
scrambled control is decisive: several heads with enormous uniform-null
ratios (base L2H11 at 29×; large's H1 column) score identically on shuffled
series — they match token *values*, not temporal lags — and would have topped
a control-free ranking. Trend-sign probes saturate at 1.00 accuracy in
*every* layer of every model (with a level-offset control blocking the
mean-token shortcut); only slope-magnitude R² has a profile (0.5–0.8 at L0
rising to ~0.95 by mid-depth). On changepoint series, the heads whose
attention collapses onto post-changepoint keys are largely the seasonal
candidates themselves (small L5H7; base L9H10/L11H1), a first hint that
"changepoint handling" is the seasonal circuit re-locking, confirmed
mechanistically in §8.

## 5. Causal validation

**Group patching on four pair types** (logit-difference recovery, 10 seeds,
paired-bootstrap CIs; matched-size random control groups). Period pairs:
mini +0.12, small +0.20, base +0.50, large +0.27, against controls of +0.04,
+0.03, +0.07, +0.00. Phase pairs pattern identically (+0.14 to +0.52).
Trend-on/off pairs: −0.001 and +0.002 at base and large — the seasonal group
carries *no* trend information at the scales where it is cleanly measurable.

**Single- vs all-position patching.** Single-position (final context token)
effects are ≈ 0 everywhere (max +0.05 vs up to +0.52 all-position). In an
encoder–decoder forecaster — bidirectional encoder, decoder cross-attending
to all positions — per-position patching understates per-component effects
even more severely than in autoregressive LMs; both numbers are reported for
every result.

**Abstract periodicity.** Patching candidates from a clean run of a
*different* random pattern with the same period recovers as much as the
matched clean run (base +0.73 vs +0.50). The heads transmit "where we are in
a P-cycle," not pattern content — induction as relation-tracking, not
copying.

**Per-head verdicts and redundancy.** With CI-excludes-0 as the criterion:
mini 1/1 confirmed (L2H2), small 1/3 (L5H7, +0.17), base 5/12 (L9H1 alone
+0.40), large 4/15 (all ≤ +0.02). At 710M no single head matters (candidates
vs 30 non-candidates: permutation p = 0.89) while the group stays causal
against a +0.000 control — redundancy, quantified in §7.

**Path patching.** Direct-path patches (downstream encoder frozen) show top
heads write 49–61% of their effect directly to the encoder output at
small/base/large (9% at mini). Receiver decomposition (sender group → each
decoder cross-attention head) concentrates in 3 cross-heads at mini (77% of
the sender effect) and diffuses with scale (base: top-3 = 4%).

**Pre-registered confirmatory verdict (H1).** Under the frozen protocol —
selection by attention ratio on seeds 100–104, evaluation on 105–119 —
**small is CONFIRMED**: +0.278 [+0.166, +0.426] vs control +0.004. Mini fails
selection; base and large pass selection but fail causally (−0.007, −0.003),
because the frozen rule lacked the scrambled control and selected the
value-matching impostor columns. H1-general (≥3 of 4 scales): **not
confirmed**. The post-hoc diagnosis (labeled exploratory) is exact: with the
scrambled control the causal groups at base/large are recovered; without it,
attention-based selection fails at scale. We report this as the null it is —
and as a selection-methodology result the field can use.

## 6. The dissociation table

Mean-ablating {seasonal heads}, zero-projecting {trend direction}, and
mean-ablating {changepoint-collapse heads}, tested on {seasonal, trend,
changepoint, mixed} families (ΔCRPS/scale, paired CIs): the table is not a
clean diagonal but a **two-block structure**. Wherever head ablation bites
(base: seasonal +0.37, changepoint +0.52), seasonal and changepoint degrade
*together* while trend is spared at exactly 0.000 at every scale — the
"didn't just break the model" defense holds. The trend row is null
everywhere it is valid: projecting out the trend direction hurts nothing at
mini/small/base — including trend itself — and at large it damages all
families indiscriminately (invalid as a targeted intervention). The
pre-registered H3 criterion (degradation *ratios* ≥ 2 with CI > 1) fails 0/4,
partly for a criterion-design reason: at small the sign pattern is precisely
the predicted dissociation (seasonal +0.037, trend −0.003), but a ratio test
explodes when the spared task's degradation is ~0. Dissociation criteria
should be difference-based, not ratio-based.

## 7. Generality: scales and families

**Across scale** (Fig. `results/emergence.png`): candidates 1→3→12→15; peak
attention ratio 4.7×→18.5×; best single-head causal effect +0.12→+0.40
(consolidation into base's L9H1) then **collapse** to +0.02 at large while
the group effect holds (+0.27 vs 0.000); redundancy index (1 −
best-head/group) 0.00→0.16→0.19→0.93. No phase transition: *sharpen,
consolidate, dissolve*.

**Across family.** TimesFM 2.5-200M (decoder-only, 32-step patches; periods
64/96 = patch-lags 2/3; causal null): zero candidates — every high-ratio head
scores identically on scrambled series (recency bias, no seasonal structure).
Chronos-Bolt-small — the *same 6L×8H T5 encoder* as our confirmed model,
differing only in patch I/O — also zero, with dead causal patches (−0.001).
**What generalizes:** within token-Chronos, everything — heads, causality,
mid-depth placement, group redundancy growth. **What doesn't:** the motif
itself, to any patch-based model, including one with an identical backbone.
Seasonal induction heads are a property of LM-style value-bin tokenization.

## 8. Mechanism and payoff

**Input-detected period.** On sequences whose period switches 7→12 mid-way,
the confirmed heads flip from 14.8× lag-7 preference (before) to 16.0×
lag-12 preference (after), with lag-12 mass overtaking lag-7 **five steps**
after the switch — under half a cycle of the new period. The period is
computed from the input, per query, not stored as a learned lag. This single
result subsumes three earlier observations: cross-period generalization,
abstract-periodicity transfer (§5), and "changepoint heads" being the
seasonal heads (§4): a changepoint is simply a re-lock event.

**Trend is not a linear readout.** Steering the probe direction by ±k gives
a flat-then-saturating, asymmetric response (correlation with k: 0.77; near
zero in ±1, −2.0 at k=−2, +0.08 at k=+4). Combined with the pre-registered
H2 failure (probes 1.00 everywhere; steering direction-correct 50–60% =
chance) and the projection nulls (§6), trend in Chronos is **linearly
decodable everywhere and causally localized nowhere** — a clean
decode≠control dissociation.

**Payoff: circuit-informed selective prediction on real data.** On ETTh1
(hourly, period 24; 150 windows, 168-step contexts), activation of the three
confirmed heads at lag 24 predicts scale-normalized forecast error:
ρ = −0.376 (p = 2×10⁻⁶); keeping the top-25% activation windows cuts mean
error by 38% (0.167→0.104). Activation also tracks seasonal-naive
predictability (ρ = −0.673): the circuit is a real-data seasonality meter.
Caution: against **MASE** the correlation reverses (+0.18), because MASE's
seasonal-naive denominator is smallest exactly where the heads fire —
MASE-based "difficulty" labels partly encode seasonality strength.

## 9. Limitations

Confirmatory support for H1 comes from one scale (46M); the base/large causal
groups are exploratory pending a v2 pre-registration with scrambled-control
selection (and a difference-based H3 criterion) on a fresh seed range.
Synthetic series are pattern-seasonal with modest noise; real-data evidence
is one dataset (ETTh1) and one payoff task. The TimesFM analysis tests
patch-lags 2–3 at one context length; deeper search could yet find structure.
Mean-ablation and single-direction interventions are coarse; trend may be
steerable via nonlinear or multi-dimensional interventions we did not find.
CRPS/expected-value metrics evaluate one forecast step. The prereg's frozen
selection rule predates the scrambled control — a disclosed design flaw that
itself became a finding.

## 10. Related work

Induction heads and in-context copying [Elhage et al., 2021; Olsson et al.,
2022]; activation/path patching and causal circuit analysis [Vig et al.,
2020; Meng et al., 2022; Wang et al., 2023; Goldowsky-Dill et al., 2023;
Heimersheim & Nanda, 2024]; linear probing and its causal gap [Alain &
Bengio, 2016; Belinkov, 2022]; activation steering [Turner et al., 2023; Li
et al., 2023]; TSFMs: Chronos [Ansari et al., 2024], Chronos-Bolt, TimesFM
[Das et al., 2024], Lag-Llama [Rasul et al., 2023], Moirai [Woo et al.,
2024]; pre-registration in empirical ML [e.g., Cockburn et al., 2020]. To our
knowledge this is the first causal circuit-level account of a time-series
foundation model and the first pre-registered mech-interp study.

## Reproducibility

Everything is in this repository: frozen `PREREGISTRATION.md` +
`DEVIATIONS.md`, per-stage scripts (`verify_harness.py`, `inventory.py`,
`causal.py`, `stage2_verdicts.py`, `dissociation.py`, `confirmatory_h{1,2,3}.py`,
`bolt_replication.py`, `timesfm_replication.py`, `emergence.py`,
`mechanism.py`, `payoff_failure.py`), raw per-seed results in `results/`, and
`reproduce.sh` (fixed seeds, logged runs). Chronos experiments run on a
single Apple-silicon machine; TimesFM in a separate venv.
