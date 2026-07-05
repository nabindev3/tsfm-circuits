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
| `PREREGISTRATION.md` | 🔒 frozen | hypotheses, thresholds, seed policy, stopping rules — **read first** |
| `synthetic.py` | ✅ tested | controlled series + **minimal-pair API** (the causal-patching backbone) |
| `attention_analysis.py` | ✅ tested | seasonal attention score + patching-effect math + **bootstrap CIs** |
| `chronos_harness.py` | ✅ verified on-device | tokenize → cache attention + residuals; per-head, group & encoder-output patching |
| `verify_harness.py` | ✅ 4/4 models pass | harness lock: identity patch, exact full recovery (effect=1.0000), reshape unit test |
| `demo.py` | ✅ verified on-device | RQ1 descriptive: score every head for seasonal attention |
| `patch_demo.py` | ✅ verified on-device | RQ1 causal: patch every head on period-7↔12 minimal pairs |
| `reproduce.sh` | ✅ | rerun everything with fixed seeds, logged to `logs/` |

The harness is verified on all four study models — `chronos-t5-{mini,small,base,large}`
(4L×8H / 6L×8H / 12L×12H / 24L×16H) — on this machine (MPS, interpreter
`~/.venvs/tsfm-sae-difficulty`; conda base has a broken chronos). On every model the
encoder-output patch reproduces clean logits exactly (effect 1.0000) and the identity
patch is a numerical no-op. Patching *all heads* gives effect ≈ 0.9–1.5 (not expected
to be 1.0: the residual stream carries corrupted embeddings past every splice; >1
means the attention pathway overshoots the behavioral gap).

## Quick start

```bash
pip install -r requirements.txt
python synthetic.py            # self-test the data engine
python attention_analysis.py   # self-test scoring + CIs (perfect head ~1.0, null ~0.05)
python verify_harness.py --device mps    # harness lock across all 4 model scales
python chronos_harness.py --device mps   # smoke: cache + forecast metric + patching
python demo.py --device mps    # RQ1 descriptive: seasonal heads in Chronos-T5-small
python patch_demo.py --device mps        # RQ1 causal: per-head patching effects
./reproduce.sh                 # all of the above, logged to logs/
```

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
2. ✅ **RQ1 descriptive** — `demo.py`: rank heads by seasonal attention *(exploratory)*
3. ✅ **RQ1 causal** — `patch_demo.py` + group patching on period-7↔12 pairs
   *(exploratory; confirmatory rerun on seeds 100–119 pending)*
4. **RQ2** — probe + steer a trend direction from cached residuals (`cache.resid`)
5. **RQ3** — changepoint-reset analysis on `season_plus_changepoint` (generator +
   `changepoint_pair` already in `synthetic.py`)
6. **RQ4** — double dissociation: ablate seasonal heads vs trend direction, split by family
7. **RQ5** — replicate on `chronos-t5-base` (`--model amazon/chronos-t5-base` works
   everywhere)
8. SAE pass on `cache.resid` (drop in your existing TopK SAE)

`cache.resid[layer]` gives `[L, d]` activations per encoder block — that's the input to
your existing probing / SAE / calibration code, so phases 4–8 reuse what you already built.
