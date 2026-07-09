#!/usr/bin/env bash
# Reproduce everything: self-tests, harness verification on all study models,
# and the exploratory RQ1 results. All scripts use fixed seeds; every run is
# logged to logs/ with a timestamp.
#
#   PY=~/.venvs/tsfm-sae-difficulty/bin/python DEVICE=mps ./reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
DEVICE="${DEVICE:-mps}"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p logs

run() {
  local name="$1"; shift
  echo "=== $name ==="
  "$@" 2>&1 | tee "logs/$STAMP-$name.log"
}

run synthetic          "$PY" synthetic.py
run attention-analysis "$PY" attention_analysis.py
run verify-harness     "$PY" verify_harness.py --device "$DEVICE"
run harness-smoke      "$PY" chronos_harness.py --device "$DEVICE"
run demo               "$PY" demo.py --device "$DEVICE"
run patch-demo         "$PY" patch_demo.py --device "$DEVICE"
run inventory          "$PY" inventory.py --device "$DEVICE"
run causal             "$PY" causal.py --device "$DEVICE"
run stage2-verdicts    "$PY" stage2_verdicts.py
run dissociation       "$PY" dissociation.py --device "$DEVICE"
run bolt-replication   "$PY" bolt_replication.py
run emergence          "$PY" emergence.py
run mechanism          "$PY" mechanism.py --device "$DEVICE"
run payoff             "$PY" payoff_failure.py --device "$DEVICE"

echo "all reproduced — logs in logs/$STAMP-*.log"
