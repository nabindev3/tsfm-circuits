"""Chronos-T5 harness: tokenize -> cache encoder attention + residuals; per-head
causal patching on minimal pairs.

Chronos quantizes the series into value-bins and runs a T5 over the bin tokens, so
all of this is standard encoder surgery:

  * `run_with_cache` runs the encoder once with output_attentions/hidden_states
    and returns per-layer attention maps and post-block residuals.
  * `predict_value` is the deterministic forecast metric: the expected value of
    the first-step decoder distribution (softmax over value-token logits dotted
    with the tokenizer's bin centers, times the context scale). No sampling.
  * `patch_head` splices ONE clean head into a corrupted run. T5's attention
    output projection `o` mixes heads, so we hook the *input* of `o`, where the
    tensor is still [batch, L, num_heads * d_head], and splice there.

Smoke test (downloads amazon/chronos-t5-small on first run):
    python chronos_harness.py --device mps
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from attention_analysis import patching_effect

# If `discover` shows different module names on your chronos/transformers version,
# adjust these hints — everything else keys off them only for discovery output.
ATTENTION_CLASS_HINT = "T5Attention"
BLOCK_CLASS_HINT = "T5Block"

DEFAULT_MODEL = "amazon/chronos-t5-small"


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(model_id: str = DEFAULT_MODEL, device: str | None = None):
    from chronos import ChronosPipeline

    device = device or auto_device()
    try:
        pipe = ChronosPipeline.from_pretrained(
            model_id, device_map=device, dtype=torch.float32,
            attn_implementation="eager",
        )
    except TypeError:
        pipe = ChronosPipeline.from_pretrained(
            model_id, device_map=device, dtype=torch.float32,
        )
        get_inner(pipe).config._attn_implementation = "eager"
    return pipe


def get_inner(pipe):
    """The T5ForConditionalGeneration inside the pipeline (ChronosModel wraps it)."""
    m = pipe.model
    return m.model if hasattr(m, "model") else m


def get_device(pipe) -> torch.device:
    return next(get_inner(pipe).parameters()).device


def discover(pipe) -> None:
    """Print attention/block module names — run this if run_with_cache errors on
    a new chronos/transformers version, then fix the hints above."""
    inner = get_inner(pipe)
    for name, mod in inner.named_modules():
        cls = type(mod).__name__
        if ATTENTION_CLASS_HINT in cls or BLOCK_CLASS_HINT in cls:
            print(f"{cls:28s} {name}")


def tokenize(pipe, values: np.ndarray):
    """series values -> (token_ids [1, L], attention_mask [1, L], scale [1])."""
    context = torch.as_tensor(np.asarray(values), dtype=torch.float32).unsqueeze(0)
    token_ids, attention_mask, scale = pipe.tokenizer.context_input_transform(context)
    device = get_device(pipe)
    return token_ids.to(device), attention_mask.to(device), scale


@dataclass
class Cache:
    tokens: torch.Tensor        # [L] token ids
    scale: float
    attn: list                  # per layer: [heads, L, L]
    resid: list                 # per layer: [L, d_model] post-block hidden states
    embed: torch.Tensor         # [L, d_model] pre-block-0 hidden states
    valid_len: int              # L minus trailing EOS, for attention scoring


@torch.no_grad()
def run_with_cache(pipe, values: np.ndarray) -> Cache:
    inner = get_inner(pipe)
    ids, mask, scale = tokenize(pipe, values)
    out = inner.encoder(input_ids=ids, attention_mask=mask,
                        output_attentions=True, output_hidden_states=True,
                        return_dict=True)
    if out.attentions is None or out.attentions[0] is None:
        raise RuntimeError(
            "encoder returned no attention weights — run discover(pipe) and force "
            "eager attention (see load_pipeline)")
    L = ids.shape[1]
    eos_id = inner.config.eos_token_id
    valid_len = L - 1 if eos_id is not None and int(ids[0, -1]) == eos_id else L
    return Cache(
        tokens=ids[0].cpu(),
        scale=float(scale[0]),
        attn=[a[0].float().cpu() for a in out.attentions],
        resid=[h[0].float().cpu() for h in out.hidden_states[1:]],
        embed=out.hidden_states[0][0].float().cpu(),
        valid_len=valid_len,
    )


# ---------------------------------------------------------------------------
# deterministic forecast metric
# ---------------------------------------------------------------------------


@torch.no_grad()
def _first_step_logits(pipe, ids, mask) -> torch.Tensor:
    inner = get_inner(pipe)
    dec = torch.full((ids.shape[0], 1), inner.config.decoder_start_token_id,
                     dtype=torch.long, device=ids.device)
    out = inner(input_ids=ids, attention_mask=mask, decoder_input_ids=dec)
    return out.logits[:, -1, :]  # [batch, vocab]


def _expected_value(pipe, logits: torch.Tensor, scale: float) -> float:
    """E[value] of the first forecast step: softmax over value tokens x centers."""
    tok = pipe.tokenizer
    n_special = tok.config.n_special_tokens
    value_logits = logits[0, n_special + 1:].float().cpu()  # ids offset by PAD/EOS + 1
    centers = tok.centers.float().cpu()
    assert value_logits.shape == centers.shape, (
        f"value-logit slice {tuple(value_logits.shape)} != centers "
        f"{tuple(centers.shape)}; check n_special_tokens offset for this version")
    probs = torch.softmax(value_logits, dim=-1)
    return float((probs * centers).sum() * scale)


@torch.no_grad()
def predict_value(pipe, values: np.ndarray) -> float:
    ids, mask, scale = tokenize(pipe, values)
    return _expected_value(pipe, _first_step_logits(pipe, ids, mask), float(scale[0]))


# ---------------------------------------------------------------------------
# per-head causal patching
# ---------------------------------------------------------------------------


def _encoder_o_module(pipe, layer: int):
    """The output projection of encoder layer `layer`'s self-attention. Its input
    is [batch, L, num_heads * d_head] with heads still separate."""
    return get_inner(pipe).encoder.block[layer].layer[0].SelfAttention.o


@torch.no_grad()
def _capture_pre_o(pipe, ids, mask, layers, capture_embed: bool = False) -> dict:
    """Run once, capture the pre-`o` tensor at each requested encoder layer
    (and optionally the encoder token embeddings)."""
    store: dict = {}
    handles = []
    for layer in layers:
        def hook(module, args, _layer=layer):
            store[_layer] = args[0].detach().clone()
        handles.append(_encoder_o_module(pipe, layer).register_forward_pre_hook(hook))
    if capture_embed:
        L = ids.shape[1]

        def ehook(module, args, output):
            # embed_tokens is shared with the decoder; only the encoder call
            # has our full sequence length
            if output.shape[1] == L:
                store["embed"] = output.detach().clone()
        handles.append(
            get_inner(pipe).encoder.embed_tokens.register_forward_hook(ehook))
    try:
        logits = _first_step_logits(pipe, ids, mask)
    finally:
        for h in handles:
            h.remove()
    store["logits"] = logits
    return store


@torch.no_grad()
def patch_heads(pipe, clean_values: np.ndarray, corr_values: np.ndarray,
                heads: list, _clean_store: dict | None = None,
                patch_embed: bool = False) -> dict:
    """Run the corrupted series with a GROUP of encoder heads spliced from the
    clean run. `heads` is a list of (layer, head) pairs, patched simultaneously —
    the group version matters because seasonal behavior is distributed across
    backup heads, so single-head patches understate the circuit. Returns
    yhat_clean / yhat_corr / yhat_patched (expected first-step forecast in value
    space) plus the normalized patching effect."""
    inner = get_inner(pipe)
    cfg = inner.config
    num_heads, d_head = cfg.num_heads, cfg.d_kv
    by_layer: dict = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    ids_cl, mask_cl, scale_cl = tokenize(pipe, clean_values)
    ids_co, mask_co, scale_co = tokenize(pipe, corr_values)
    if ids_cl.shape != ids_co.shape:
        raise ValueError(
            f"minimal pair must tokenize to equal length, got {ids_cl.shape} vs "
            f"{ids_co.shape} — use same-length series from synthetic.py")

    store = _clean_store or _capture_pre_o(pipe, ids_cl, mask_cl, list(by_layer),
                                           capture_embed=patch_embed)
    yhat_clean = _expected_value(pipe, store["logits"], float(scale_cl[0]))
    yhat_corr = _expected_value(pipe, _first_step_logits(pipe, ids_co, mask_co),
                                float(scale_co[0]))

    def make_splice(clean_pre_o, layer_heads):
        def splice(module, args):
            out = args[0]
            # out: [batch, L, num_heads * d_head] -> [batch, L, num_heads, d_head]
            b, Lc, _ = out.shape
            out4 = out.view(b, Lc, num_heads, d_head).clone()
            clean4 = clean_pre_o.view(b, Lc, num_heads, d_head)
            for h in layer_heads:
                out4[:, :, h, :] = clean4[:, :, h, :]
            return (out4.view(b, Lc, num_heads * d_head),)
        return splice

    handles = [
        _encoder_o_module(pipe, layer).register_forward_pre_hook(
            make_splice(store[layer], tuple(layer_heads)))
        for layer, layer_heads in by_layer.items()
    ]
    if patch_embed:
        clean_embed = store["embed"]

        def esplice(module, args, output):
            # shape guard: the shared embedding also embeds decoder tokens
            return clean_embed if output.shape == clean_embed.shape else output

        handles.append(get_inner(pipe).encoder.embed_tokens
                       .register_forward_hook(esplice))
    try:
        yhat_patched = _expected_value(
            pipe, _first_step_logits(pipe, ids_co, mask_co), float(scale_co[0]))
    finally:
        for h in handles:
            h.remove()

    return dict(
        yhat_clean=yhat_clean, yhat_corr=yhat_corr, yhat_patched=yhat_patched,
        effect=patching_effect(yhat_clean, yhat_corr, yhat_patched),
        heads=list(heads),
    )


@torch.no_grad()
def patch_head(pipe, clean_values: np.ndarray, corr_values: np.ndarray,
               layer: int, head: int, _clean_store: dict | None = None) -> dict:
    """Single-head patch; see patch_heads."""
    r = patch_heads(pipe, clean_values, corr_values, [(layer, head)], _clean_store)
    r.update(layer=layer, head=head)
    return r


@torch.no_grad()
def patch_encoder_output(pipe, clean_values: np.ndarray,
                         corr_values: np.ndarray) -> dict:
    """Upper-bound sanity patch: replace the ENTIRE encoder output of the
    corrupted run with the clean run's. The decoder then sees exactly the clean
    encodings, so first-step logits must match the clean run's to float precision
    and the effect must be ~1.0. This validates the metric + hook plumbing end to
    end; if it doesn't hit 1.0, nothing downstream can be trusted.

    (Note: patching all *heads* does NOT imply effect 1.0 — the residual stream
    still carries the corrupted token embeddings past every attention splice.)"""
    inner = get_inner(pipe)
    ids_cl, mask_cl, scale_cl = tokenize(pipe, clean_values)
    ids_co, mask_co, scale_co = tokenize(pipe, corr_values)
    if ids_cl.shape != ids_co.shape:
        raise ValueError("minimal pair must tokenize to equal length")

    logits_clean = _first_step_logits(pipe, ids_cl, mask_cl)
    yhat_clean = _expected_value(pipe, logits_clean, float(scale_cl[0]))
    yhat_corr = _expected_value(pipe, _first_step_logits(pipe, ids_co, mask_co),
                                float(scale_co[0]))

    enc_clean = inner.encoder(input_ids=ids_cl, attention_mask=mask_cl,
                              return_dict=True).last_hidden_state

    def swap(module, args, output):
        output.last_hidden_state = enc_clean
        return output

    handle = inner.encoder.register_forward_hook(swap)
    try:
        logits_patched = _first_step_logits(pipe, ids_co, mask_co)
    finally:
        handle.remove()

    # the patched decoder distribution lives in the clean run's normalized
    # space, so score it with the clean scale for the exact-recovery check
    yhat_patched = _expected_value(pipe, logits_patched, float(scale_cl[0]))
    return dict(
        yhat_clean=yhat_clean, yhat_corr=yhat_corr, yhat_patched=yhat_patched,
        effect=patching_effect(yhat_clean, yhat_corr, yhat_patched),
        logits_match=bool(torch.allclose(logits_patched, logits_clean, atol=1e-4)),
    )


@torch.no_grad()
def patch_all_heads(pipe, pair) -> dict:
    """patch_head for every encoder (layer, head) on a MinimalPair. One clean
    forward captures all layers; one corrupted forward per head. Returns
    effects [layers, heads] plus the baseline yhats."""
    inner = get_inner(pipe)
    cfg = inner.config
    layers = list(range(cfg.num_layers))

    ids_cl, mask_cl, _ = tokenize(pipe, pair.clean.values)
    store = _capture_pre_o(pipe, ids_cl, mask_cl, layers)

    effects = np.zeros((cfg.num_layers, cfg.num_heads))
    base = None
    for layer in layers:
        for head in range(cfg.num_heads):
            r = patch_head(pipe, pair.clean.values, pair.corrupted.values,
                           layer, head, _clean_store=store)
            effects[layer, head] = r["effect"]
            base = base or dict(yhat_clean=r["yhat_clean"], yhat_corr=r["yhat_corr"])
    return dict(effects=effects, **base)


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------


def _smoke(device: str | None, model_id: str) -> None:
    from synthetic import period_pair

    pipe = load_pipeline(model_id, device)
    inner = get_inner(pipe)
    cfg = inner.config
    print(f"loaded {model_id} on {get_device(pipe)} — "
          f"{cfg.num_layers} layers x {cfg.num_heads} heads, d_kv={cfg.d_kv}")

    pair = period_pair(7, 12, length=140, noise=0.05, seed=0)
    cache = run_with_cache(pipe, pair.clean.values)
    L = len(cache.tokens)
    assert len(cache.attn) == cfg.num_layers
    assert cache.attn[0].shape == (cfg.num_heads, L, L)
    assert cache.resid[0].shape == (L, cfg.d_model)
    print(f"cache OK: {L} tokens (valid {cache.valid_len}), "
          f"attn [{cfg.num_heads},{L},{L}] x {cfg.num_layers} layers, scale "
          f"{cache.scale:.3f}")

    yc = predict_value(pipe, pair.clean.values)
    yx = predict_value(pipe, pair.corrupted.values)
    print(f"forecast t+1: clean yhat={yc:+.3f} (truth {pair.clean.future[0]:+.3f}) | "
          f"corrupted yhat={yx:+.3f} (truth {pair.corrupted.future[0]:+.3f})")
    assert abs(yc - pair.clean.future[0]) < abs(yc - pair.corrupted.future[0]), (
        "clean forecast should be closer to the clean continuation — model or "
        "metric is off")

    mid = cfg.num_layers // 2
    print(f"patching all {cfg.num_heads} heads of layer {mid} "
          f"(clean head -> corrupted run):")
    for head in range(cfg.num_heads):
        r = patch_head(pipe, pair.clean.values, pair.corrupted.values, mid, head)
        print(f"  L{mid}H{head}: patched yhat={r['yhat_patched']:+.3f} "
              f"effect={r['effect']:+.3f}")
    print("chronos_harness.py: smoke test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="mps | cuda | cpu (default: auto)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    _smoke(args.device, args.model)
