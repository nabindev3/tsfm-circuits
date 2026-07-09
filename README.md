# tsfm-circuits

Infrastructure for **"Do Forecasting Models Reinvent Induction Heads?"** — a
circuit-level analysis of seasonality, trend, and changepoints in Chronos.

The bet: Chronos quantizes a series into discrete value-bins and runs a T5 over them,
so it's a language model in disguise. That lets us ask the induction-head question of a
forecasting model, and seasonality is structurally an induction task ("the token P steps
ago predicts the next one").

## What's here

| File | Status | Purpose |
|---|---|---|
| `PAPER.md` | ✅ draft | **the manuscript** — abstract through related work, every number from `results/` |
| `PREREGISTRATION.md` | 🔒 frozen | hypotheses, thresholds, seed policy, stopping rules — **read first** |
| `DEVIATIONS.md` | ✅ | the one documented protocol deviation (D1, control sampling on large) |
| `synthetic.py` | ✅ tested | controlled series + **minimal-pair API** (the causal-patching backbone) |
| `attention_analysis.py` | ✅ tested | seasonal attention score + patching-effect math + **bootstrap CIs** |
| `chronos_harness.py` | ✅ verified on-device | tokenize → cache attention + residuals; per-head, group & encoder-output patching |
| `verify_harness.py` | ✅ 4/4 models pass | harness lock: identity patch, exact full recovery (effect=1.0000), reshape unit test |
| `demo.py` | ✅ verified on-device | RQ1 descriptive: score every head for seasonal attention |
| `patch_demo.py` | ✅ verified on-device | RQ1 causal: patch every head on period-7↔12 minimal pairs |
| `inventory.py` | ✅ run on 4 scales | **descriptive circuit inventory**: ranked candidates vs uniform null + scrambled/permutation controls, with 95% bootstrap CIs → `results/` |
| `causal.py` | ✅ run on 4 scales | **Stage 2 causal validation**: group/single-position patching on 4 pair types, controls, permutation nulls, path patching → `results/` |
| `stage2_verdicts.py` | ✅ | per-candidate CONFIRMED/rejected verdicts (CI excludes 0) + group verdict vs prereg thresholds |
| `dissociation.py` | ✅ run on 4 scales | **Stage 3 double dissociation**: ablate {seasonal heads, trend dir, CP heads} × test {seasonal, trend, changepoint, mixed}, ΔCRPS with CIs |
| `reproduce.sh` | ✅ | rerun everything with fixed seeds, logged to `logs/` |

The harness is verified on all four study models — `chronos-t5-{mini,small,base,large}`
(4L×8H / 6L×8H / 12L×12H / 24L×16H) — on this machine (MPS, interpreter
`~/.venvs/tsfm-sae-difficulty`; conda base has a broken chronos). On every model:
the encoder-output patch reproduces clean logits exactly (effect 1.0000), **all heads
+ token embeddings patched gives full recovery (effect = 1.0)** — the "patch
everything → clean behavior" trivial case — and the identity patch is a numerical
no-op. Patching all heads *without* embeddings gives effect ≈ 0.9–1.5 (correctly not
1.0: the residual stream carries corrupted embeddings past every splice; >1 means the
attention pathway overshoots the behavioral gap).

## Quick start

```bash
pip install -r requirements.txt
python synthetic.py            # self-test the data engine
python attention_analysis.py   # self-test scoring + CIs (perfect head ~1.0, null ~0.05)
python verify_harness.py --device mps    # harness lock across all 4 model scales
python chronos_harness.py --device mps   # smoke: cache + forecast metric + patching
python demo.py --device mps    # RQ1 descriptive: seasonal heads in Chronos-T5-small
python patch_demo.py --device mps        # RQ1 causal: per-head patching effects
python inventory.py --device mps         # full descriptive inventory, all 4 scales
python causal.py --device mps  # Stage 2: causal validation sweep, all 4 scales
python stage2_verdicts.py      # per-candidate confirm/reject table
python dissociation.py --device mps      # Stage 3: double-dissociation table
./reproduce.sh                 # all of the above, logged to logs/
```

## Confirmatory H1 result (seeds 100–119, unblinded 2026-07-07)

**H1-general: NOT CONFIRMED (1/4).** Under the frozen protocol: **small
CONFIRMED** (group recovery +0.278 [+0.166, +0.426] vs control +0.004); mini
fails selection (0 heads > 3× null at both periods); base and large pass
selection but fail causally (−0.007 / −0.003). Diagnosis (exploratory,
post-hoc): the frozen selection rule — attention ratio alone, no
phase-scrambled control, which was invented only after freezing — selects
token-matching heads (base's H11 column, large's H1 column) that the causal
test correctly rejects. With the scrambled control (Stage 1/2 pipeline,
exploratory seeds), the causally-effective groups at base (+0.50 vs +0.07
control) and large (+0.27 vs 0.00) are recovered. The clean null: **attention
scores alone cannot find seasonal circuits at scale; a scrambled control is
necessary.** One protocol deviation was required and is logged in
`DEVIATIONS.md` (control sampling on large). Raw data:
`results/confirmatory-h1.json`.

## Confirmatory H2 result (seeds 100–119, unblinded 2026-07-07)

**H2-general: NOT CONFIRMED (0/4) — decodable ≠ steerable.** Probes pass at
1.00 held-out accuracy in every model, but adding ±α·(probe direction) moves
the first-step forecast in the predicted direction only 50–60% of the time
(chance 50%), despite large movements. LOO layer selection saturated at 1.00
for all layers (tie-break → layer 0): there is no signal to pick a steering
locus — trend is redundantly encoded everywhere and causally localized
nowhere. Labeled-exploratory redesigns (4×α, all-layer steering) do not
rescue it. Raw data: `results/confirmatory-h2.json`.

## Confirmatory H3 result (seeds 105–119, run 2026-07-07)

**H3: NOT CONFIRMED (0/4).** The direction-projection half fails everywhere:
zero-projecting the H2 direction changes trend error by ≈ 0 (matching the H2
steering null — no causal trend locus). The head-ablation half fails
everywhere too, but for two different reasons worth separating: at base/large
the frozen H1 selection provides token-matching head groups with no seasonal
effect; at **small** the effect pattern is exactly the predicted dissociation
(seasonal error +0.037, trend error −0.003, i.e. spared) — but the frozen
ratio criterion (D-seasonal ≥ 2× D-trend, CI-lb > 1) is ill-posed when the
spared task's degradation is ~0/negative: the ratio explodes instead of
passing. A dissociation "too clean" for its own test statistic — reported
as-is per the stopping rules. Raw data: `results/confirmatory-h3.json`.

## Stage 4 — generality: what generalizes and what doesn't

**Across scale** (`emergence.py`, `results/emergence.png`): lag-tracking heads
exist at every Chronos-T5 scale and attention sharpens with scale (max ratio
4.7× → 18.5×), but there is **no LLM-style phase transition**. Instead:
*consolidate, then dissolve* — the best single head's causal effect rises
mini→base (+0.12 → +0.40, base's L9H1) then collapses at large (+0.02) while
the group effect stays strong (+0.27 vs 0.00 control); the redundancy index
(1 − best-head/group) goes 0.00 → 0.16 → 0.19 → **0.93**.

**Across family**: the motif does NOT transfer, and the two negatives triangulate
the cause. TimesFM 2.5-200M (different family: decoder-only, patch input): zero
lag-tracking heads. **Chronos-Bolt-small (same 6L×8H T5 backbone as the
confirmed model, only the I/O differs — patches + quantile head): also zero**,
with scrambled controls matching real series and dead causal patches
(`bolt_replication.py`). Same architecture, same training recipe family, no
value-bin tokenization → no induction heads. **Generalization statement:
seasonal induction heads are a property of LM-style value-bin tokenization,
not of transformer forecasters in general — Chronos reinvents induction heads
precisely because its design makes forecasting language modeling.**

## Stage 5 — mechanism + payoff (chronos-t5-small, the confirmed circuit)

**Mechanism** (`mechanism.py`): the period is **detected from the input, not a
fixed learned lag** — on a mid-sequence period switch (7→12) the confirmed
heads flip from 14.8× lag-7 preference to 16.0× lag-12 preference, re-locking
within **5 steps (< half a cycle of the new period)**. This one experiment
explains the cross-period generalization, the mismatched-clean transfer
(abstract periodicity), and the changepoint re-locking, and it is why "these
heads are induction heads" is the right description: they attend to "one
*current* period ago" as an input-dependent relation. Trend, by contrast, is
**not a linear readout**: ±k steering gives a flat-then-saturating asymmetric
response (r = 0.77), quantifying the H2 null.

**Payoff** (`payoff_failure.py`): circuit-informed selective prediction on
real data (ETTh1 hourly, period 24, 150 windows). Low activation of the three
confirmed heads predicts actual mis-forecasting: Spearman(activation,
MAE/scale) = **−0.376** (p = 2×10⁻⁶); keeping the top-25% activation windows
cuts normalized MAE by **38%** (0.167 → 0.104). The heads are also a strong
real-data seasonality meter (ρ = −0.673 against seasonal-naive
predictability). Methodological caution that falls out for free: against
**MASE** the correlation *reverses* (+0.18) because MASE's seasonal-naive
denominator is smallest exactly where the heads fire — MASE-based difficulty
labels partly encode seasonality strength, which matters for any
MASE-difficulty pipeline (including fm-difficulty).

## Exploratory first results (chronos-t5-small, pattern seasonality, seeds 0–2)

⚠️ **Exploratory** — obtained before `PREREGISTRATION.md` was frozen. Confirmatory
runs use seeds 100–119 and the thresholds fixed there; these numbers guide, they
don't count.

**Descriptive (demo.py).** Three heads put >3× the uniform-null attention mass at the
seasonal lag at *both* period 7 and period 12 — i.e. they track "one cycle ago" as a
relation, not a fixed offset (the induction-head signature):
**L4H1** (3.3× / 13.4×), **L4H7** (4.8× / 3.0×), **L5H7** (4.1× / 3.8×).
Side pattern worth chasing: the H7 column lights up at P=7, the H1 column at P=12.

**Causal (patch_demo.py + patch_heads).** Single-head patches are uniformly small
(max ≈ +0.09 recovery) — the circuit is distributed. But patching the lag-tracking
trio *as a group* recovers **21.8%** of the clean↔corrupted forecast gap vs **0.1%**
for a size-matched control trio. Growing the group to all 10 marginal heads *dilutes*
the effect (+0.12), so the trio is close to the functional core.

Metric: deterministic first-forecast-step expected value (softmax over value-token
logits · bin centers · scale); effect is normalized recovery
`(patched − corrupted) / (clean − corrupted)`.

## Chronos-version notes

Chronos internals vary across versions; two spots adapt rather than assume:

1. **Module discovery.** If `run_with_cache` errors, run `discover(pipe)` (in
   `chronos_harness.py`) — it prints the attention/block module names. Adjust
   `ATTENTION_CLASS_HINT` / `BLOCK_CLASS_HINT` at the top if needed.
2. **Tokenizer signature.** `tokenize()` calls `context_input_transform`; on this
   version it returns `(token_ids, attention_mask, scale)`. If yours differs, fix the
   one-line unpack there.

Head patching splices at the *input* of each attention's output projection `o`, where
the tensor is still `[batch, L, num_heads * d_head]` (after `o` the heads are mixed).
`patch_heads` takes a list of `(layer, head)` pairs and patches them simultaneously —
use groups; single heads understate distributed circuits (see results above).

## Build sequence (from the proposal)

1. ✅ harness: synthetic generator + activation caching; 🔒 locked by
   `verify_harness.py` (4/4 models) + `PREREGISTRATION.md`
2. ✅ **RQ1 descriptive** — `demo.py` + `inventory.py`: rank heads by seasonal
   attention vs two nulls *(exploratory)*
3. ✅ **RQ1 causal** — `causal.py`: group/single-position patching on
   period/phase/trend-on-off/changepoint-location pairs with controls,
   permutation nulls, and path patching
   *(exploratory; confirmatory rerun on seeds 100–119 pending)*
4. **RQ2** — probe ✅ (`inventory.py`); steering the trend direction pending —
   note the informative null: zero-projecting the direction does not hurt trend
   forecasts (see `dissociation.py`)
5. **RQ3** — changepoint components inventoried + causally tested ✅
   (`inventory.py`, `causal.py`, `dissociation.py`)
6. ✅ **RQ4** — double dissociation table: `dissociation.py` *(exploratory)*
7. **RQ5** — replicate on `chronos-t5-base` (`--model amazon/chronos-t5-base` works
   everywhere)
8. SAE pass on `cache.resid` (drop in your existing TopK SAE)

`cache.resid[layer]` gives `[L, d]` activations per encoder block — that's the input to
your existing probing / SAE / calibration code, so phases 4–8 reuse what you already built.
