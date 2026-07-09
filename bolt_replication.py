"""Stage 4, second family (controlled): the H1 pipeline on Chronos-Bolt-small.

Bolt is the decisive comparison for the tokenization thesis: its encoder is the
SAME 6-layer x 8-head T5 stack as chronos-t5-small (where H1 confirmed), but
the input is 16-step patches with instance norm instead of value-bin tokens,
and the output is direct multi-quantile regression (deterministic, no
sampling). If the heads exist in token-Chronos but not in patch-Chronos on an
identical backbone, the induction motif is a property of the LM-style
tokenization, not of the architecture or training data.

EXPLORATORY seeds (<100). Adaptations fixed before running: periods {32, 48} =
patch-lags {2, 3}; context 512 = 32 patch tokens (+1 REG token, excluded from
scoring); bidirectional uniform null (encoder attention); scrambled control
and candidate rule as Stage 1; causal group patch = pre-`o` head splice during
the deterministic forward, recovery of the t+1 median forecast.

    python bolt_replication.py            # runs on CPU; model is small
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from attention_analysis import bootstrap_ci, head_scores, patching_effect
from synthetic import period_pair, seasonal

MODEL = "amazon/chronos-bolt-small"
CONTEXT = 512
PATCH = 16
PERIODS = (32, 48)         # -> patch lags 2, 3
NOISE = 0.05
DESC_SEEDS = range(5)
CAUSAL_SEEDS = range(10)


def load():
    from chronos import ChronosBoltPipeline
    pipe = ChronosBoltPipeline.from_pretrained(MODEL, device_map="cpu")
    m = pipe.model
    # the encoder T5Stack holds its own config copy - set flags there
    for cfg in (m.config, m.encoder.config):
        cfg._attn_implementation = "eager"
        cfg.output_attentions = True
    m.eval()
    return pipe, m


@torch.no_grad()
def predict_median(pipe, values) -> float:
    ctx = torch.as_tensor(np.asarray(values), dtype=torch.float32).unsqueeze(0)
    q = pipe.predict(ctx, prediction_length=1)   # [1, 9 quantiles, 1]
    return float(q[0, 4, 0])                     # 0.5 quantile


@torch.no_grad()
def cached_attention(pipe, m, values) -> list:
    store = {}

    def hook(module, args, output):
        if getattr(output, "attentions", None) is not None:
            store["attn"] = [a[0].float().cpu() for a in output.attentions]

    h = m.encoder.register_forward_hook(hook)
    try:
        predict_median(pipe, values)
    finally:
        h.remove()
    return store["attn"]


def _o_module(m, layer):
    return m.encoder.block[layer].layer[0].SelfAttention.o


@torch.no_grad()
def patch_group(pipe, m, clean_values, corr_values, heads) -> dict:
    n_heads, d_head = m.config.num_heads, m.config.d_kv
    by_layer: dict = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)

    y_clean = predict_median(pipe, clean_values)
    y_corr = predict_median(pipe, corr_values)

    captured: dict = {}
    handles = [
        _o_module(m, l).register_forward_pre_hook(
            lambda mod, args, _l=l: captured.__setitem__(
                _l, args[0].detach().clone()))
        for l in by_layer]
    try:
        predict_median(pipe, clean_values)
    finally:
        for h in handles:
            h.remove()

    def make_splice(clean_pre, hs):
        def hook(mod, args):
            out = args[0]
            if out.shape != clean_pre.shape:
                return None
            b, T, _ = out.shape
            out4 = out.view(b, T, n_heads, d_head).clone()
            c4 = clean_pre.view(b, T, n_heads, d_head)
            for h in hs:
                out4[:, :, h, :] = c4[:, :, h, :]
            return (out4.view(b, T, n_heads * d_head),)
        return hook

    handles = [_o_module(m, l).register_forward_pre_hook(
        make_splice(captured[l], tuple(hs))) for l, hs in by_layer.items()]
    try:
        y_patched = predict_median(pipe, corr_values)
    finally:
        for h in handles:
            h.remove()
    return dict(y_clean=y_clean, y_corr=y_corr, y_patched=y_patched,
                effect=patching_effect(y_clean, y_corr, y_patched))


def main() -> None:
    pipe, m = load()
    n_layers, n_heads = m.config.num_layers, m.config.num_heads
    print(f"{MODEL}: {n_layers}L x {n_heads}H (same backbone as "
          f"chronos-t5-small), patch={PATCH}, context={CONTEXT}")

    real, scram = {}, {}
    n_tokens = None
    for period in PERIODS:
        lag = period // PATCH
        rs, ss = [], []
        for seed in DESC_SEEDS:
            s = seasonal(period=period, length=CONTEXT, kind="pattern",
                         noise=NOISE, seed=seed)
            attn = cached_attention(pipe, m, s.values)
            n_tokens = attn[0].shape[-1]
            valid = n_tokens - 1 if m.config.chronos_config.get(
                "use_reg_token") else n_tokens
            rs.append(head_scores(attn, lag, tol=1, valid_len=valid))
            perm = np.random.default_rng(seed + 50_000).permutation(s.values)
            ss.append(head_scores(cached_attention(pipe, m, perm), lag, tol=1,
                                  valid_len=valid))
        real[period] = np.mean(rs, axis=0)
        scram[period] = np.mean(ss, axis=0)

    candidates = [(l, h) for l in range(n_layers) for h in range(n_heads)
                  if all(real[p][l, h] > 3.0
                         and real[p][l, h] > 2.0 * scram[p][l, h]
                         for p in PERIODS)]
    gm = np.exp(np.mean([np.log(np.maximum(real[p], 1e-9)) for p in PERIODS],
                        axis=0))
    print(f"\n{n_tokens} encoder tokens; top heads: ratio vs null @ "
          f"P32(lag2) | P48(lag3) (scrambled in parens)")
    for f in np.argsort(gm, axis=None)[::-1][:8]:
        l, h = np.unravel_index(f, gm.shape)
        mark = "  CANDIDATE" if (int(l), int(h)) in candidates else ""
        print(f"  L{l}H{h:<3} {real[32][l, h]:6.2f} ({scram[32][l, h]:5.2f}) | "
              f"{real[48][l, h]:6.2f} ({scram[48][l, h]:5.2f})  "
              f"gm={gm[l, h]:5.2f}{mark}")
    print(f"{len(candidates)} candidate(s): "
          f"{['L%dH%d' % c for c in candidates]}")

    blob = dict(model=MODEL, layers=n_layers, heads=n_heads,
                periods=list(PERIODS), context=CONTEXT,
                real={str(p): real[p].tolist() for p in PERIODS},
                scrambled={str(p): scram[p].tolist() for p in PERIODS},
                candidates=[list(c) for c in candidates])

    # causal: candidates if any, else the top-3 by gm (upper bound for "is
    # ANY head group causally seasonal here?")
    group = candidates or [tuple(map(int, np.unravel_index(f, gm.shape)))
                           for f in np.argsort(gm, axis=None)[::-1][:3]]
    label = "candidates" if candidates else "top-3 by ratio (no candidates)"
    rng = np.random.default_rng(7)
    pool = [(l, h) for l in range(n_layers) for h in range(n_heads)
            if (l, h) not in group]
    control = [pool[i] for i in rng.choice(len(pool), len(group),
                                           replace=False)]
    g_eff, c_eff = [], []
    for seed in CAUSAL_SEEDS:
        pair = period_pair(32, 48, length=CONTEXT, noise=NOISE, seed=seed)
        g_eff.append(patch_group(pipe, m, pair.clean.values,
                                 pair.corrupted.values, group)["effect"])
        c_eff.append(patch_group(pipe, m, pair.clean.values,
                                 pair.corrupted.values, control)["effect"])
    gm_, glo, ghi = bootstrap_ci(g_eff)
    cm_, clo, chi = bootstrap_ci(c_eff)
    print(f"\ncausal group patch ({label}, t+1 median recovery, "
          f"{len(list(CAUSAL_SEEDS))} seeds):")
    print(f"  group   {gm_:+.3f} [{glo:+.3f},{ghi:+.3f}]")
    print(f"  control {cm_:+.3f} [{clo:+.3f},{chi:+.3f}]")
    blob.update(causal_group=[list(c) for c in group], causal_label=label,
                group_effects=g_eff, control_effects=c_eff)

    out = Path("results/bolt-replication.json")
    out.write_text(json.dumps(blob, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
