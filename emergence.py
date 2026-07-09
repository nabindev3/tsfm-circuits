"""Stage 4 synthesis: the emergence story across chronos-t5-{mini,small,base,
large}, from the already-computed results/ JSONs (no model runs).

Four quantities per scale, one panel each (results/emergence.png):
  A. lag-tracking candidate heads (scrambled-control rule, inventory)
  B. sharpest head's attention ratio vs null (best gm head, inventory)
  C. best SINGLE-head causal effect (per-head period patching, causal.py)
  D. candidate-GROUP causal effect vs matched control (causal.py)

The story the numbers tell: heads exist at every scale and their attention
sharpens monotonically, but single-head causal importance rises to a peak at
base (L9H1, +0.40) and then COLLAPSES at large (~0.02) while the group effect
stays strong — no LLM-style phase transition at these scales; instead the
circuit consolidates (mini->base) and then dissolves into redundancy (large).

    python emergence.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCALES = [("mini", 20e6), ("small", 46e6), ("base", 200e6), ("large", 710e6)]
BLUE, RED = "#2a78d6", "#e34948"      # validated (dataviz palette, light mode)
INK, MUTED = "#0b0b0b", "#52514e"


def collect(out_dir: Path) -> dict:
    rows = []
    for name, params in SCALES:
        inv = json.loads((out_dir / f"inventory-chronos-t5-{name}.json")
                         .read_text())
        cau = json.loads((out_dir / f"causal-chronos-t5-{name}.json")
                         .read_text())
        gm = np.array(inv["seasonal"]["gm"])
        per_head = {k: np.mean([e[0] for e in v])
                    for k, v in cau["per_head_candidates"].items()}
        group = [e[0] for e in cau["effects"]["period"]["group"]]
        control = [e[0] for e in cau["effects"]["period"]["control"]]
        rows.append(dict(
            name=name, params=params,
            n_candidates=len(inv["seasonal"]["candidates"]),
            max_ratio=float(gm.max()),
            best_head=max(per_head, key=per_head.get),
            best_head_effect=float(max(per_head.values())),
            group_effect=float(np.mean(group)),
            group_sd=float(np.std(group)),
            control_effect=float(np.mean(control)),
            redundancy=1.0 - max(per_head.values()) / np.mean(group)
            if np.mean(group) > 0 else float("nan"),
        ))
    return dict(scales=rows)


def figure(data: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = data["scales"]
    x = [r["params"] for r in rows]
    names = [r["name"] for r in rows]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    panels = [
        ("A  lag-tracking heads", [r["n_candidates"] for r in rows],
         "count", None),
        ("B  sharpest head, attention", [r["max_ratio"] for r in rows],
         "ratio vs null", None),
        ("C  best single-head effect", [r["best_head_effect"] for r in rows],
         "recovery", [r["best_head"] for r in rows]),
        ("D  group vs control effect", [r["group_effect"] for r in rows],
         "recovery", None),
    ]
    for ax, (title, y, ylab, labels) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=6,
                zorder=3, label="candidates" if "D" in title else None)
        if "D" in title:
            ax.plot(x, [r["control_effect"] for r in rows], color=RED,
                    linewidth=2, marker="o", markersize=6, zorder=3,
                    label="matched control")
            ax.legend(frameon=False, fontsize=7, loc="upper left")
        if labels:
            for xi, yi, lab in zip(x, y, labels):
                ax.annotate(lab, (xi, yi), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7,
                            color=MUTED)
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_ylabel(ylab, fontsize=8, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, axis="y", color="#e8e7e3", linewidth=0.7, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d2")
        ax.axhline(0, color="#d8d7d2", linewidth=0.7)
    fig.suptitle("Seasonal induction heads across Chronos-T5 scales: "
                 "sharpen, consolidate, then dissolve into redundancy",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}")


def main() -> None:
    out_dir = Path("results")
    data = collect(out_dir)
    for r in data["scales"]:
        print(f"  {r['name']:6s} ({r['params'] / 1e6:.0f}M): "
              f"{r['n_candidates']:2d} candidates | max ratio "
              f"{r['max_ratio']:5.1f}x | best head {r['best_head']} "
              f"{r['best_head_effect']:+.3f} | group {r['group_effect']:+.3f} "
              f"(control {r['control_effect']:+.3f}) | redundancy "
              f"{r['redundancy']:.2f}")
    (out_dir / "emergence.json").write_text(json.dumps(data, indent=1))
    figure(data, out_dir / "emergence.png")


if __name__ == "__main__":
    main()
