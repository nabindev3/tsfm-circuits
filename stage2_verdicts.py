"""Stage 2 definition-of-done: an explicit confirm/reject verdict per candidate.

Reads results/causal-<model>.json (no model runs). Per candidate head: mean
logit-diff recovery on period pairs over seeds, 95% paired-bootstrap CI, and a
verdict — CONFIRMED if the CI excludes 0, REJECTED otherwise. The pre-registered
effect threshold (recovery >= 0.15, CI lower bound > 0.05) was defined at the
GROUP level; the group verdict is reported against it (exploratory seeds, so
this is a preview of the confirmatory test, not a substitute for it).

    python stage2_verdicts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from attention_analysis import bootstrap_ci
from verify_harness import STUDY_MODELS


def main(out_dir: Path = Path("results")) -> None:
    verdicts = {}
    for model_id in STUDY_MODELS:
        name = model_id.split("/")[-1]
        blob = json.loads((out_dir / f"causal-{name}.json").read_text())
        rows, confirmed = [], []
        for head, effs in blob["per_head_candidates"].items():
            vals = [e[0] for e in effs]           # logit-diff recovery per seed
            m, lo, hi = bootstrap_ci(vals)
            ok = lo > 0
            rows.append((head, m, lo, hi, ok))
            if ok:
                confirmed.append(head)
        rows.sort(key=lambda r: -r[1])

        g = [e[0] for e in blob["effects"]["period"]["group"]]
        c = [e[0] for e in blob["effects"]["period"]["control"]]
        gm, glo, ghi = bootstrap_ci(g)
        cm = float(np.mean(c))
        group_ok = gm >= 0.15 and glo > 0.05 and cm < 0.05

        print(f"\n=== {name}: per-candidate verdicts (period pairs, "
              f"logit-diff recovery, 95% CI over {len(blob['seeds'])} seeds) ===")
        for head, m, lo, hi, ok in rows:
            print(f"  {head:8s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                  f"{'CONFIRMED' if ok else 'rejected'}")
        print(f"  -> {len(confirmed)}/{len(rows)} candidates individually "
              f"confirmed: {confirmed}")
        print(f"  GROUP: {gm:+.3f} [{glo:+.3f},{ghi:+.3f}] vs control {cm:+.3f} "
              f"-> {'MEETS' if group_ok else 'does NOT meet'} prereg thresholds "
              f"(recovery>=0.15, CI-lb>0.05, control<0.05) [exploratory]")

        verdicts[name] = dict(
            per_head=[dict(head=h, mean=m, ci=[lo, hi],
                           confirmed=bool(ok)) for h, m, lo, hi, ok in rows],
            confirmed=confirmed,
            group=dict(mean=gm, ci=[glo, ghi], control=cm,
                       meets_prereg_thresholds=bool(group_ok)))

    (out_dir / "stage2-verdicts.json").write_text(json.dumps(verdicts, indent=1))
    print(f"\nwrote {out_dir / 'stage2-verdicts.json'}")


if __name__ == "__main__":
    main()
