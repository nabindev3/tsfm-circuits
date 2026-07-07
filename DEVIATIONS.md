# Deviations from PREREGISTRATION.md

## D1 — control-group sampling fallback (2026-07-07)

The prereg's control clause ("size-matched random control group (same layers
excluded)") was implemented as: sample control heads from layers containing no
selected head. On chronos-t5-large, the confirmatory selection (seeds 100-104)
returned 28 heads spanning ALL 24 layers — the layer-exclusion pool is empty and
the protocol as written cannot run.

**Deviation:** when the layer-exclusion pool is empty, fall back to sampling
size-matched controls from all heads not in the selected group (head-exclusion
instead of layer-exclusion), same fixed rng seed (12345).

**Blindness note:** this decision was made after seeing large's *selection*
output (the crash point) but before any causal evaluation on seeds 105-119 for
large. mini/small/base were unaffected (their runs are deterministic and
completed under the original interpretation before the crash).
