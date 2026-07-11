"""Controlled synthetic series + minimal-pair API for circuit analysis of Chronos.

The design principle: every generator returns a `Series` with the context AND the
ground-truth continuation, so downstream metrics never have to re-derive "what the
model should have said". Minimal pairs differ in exactly one causal factor and are
guaranteed to *diverge* at the first forecast step, which is what makes activation
patching interpretable.

Run directly to self-test:  python synthetic.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Series:
    values: np.ndarray  # [length] the context fed to the model
    future: np.ndarray  # [horizon] ground-truth continuation
    meta: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.values)


@dataclass
class MinimalPair:
    """Two series differing in exactly one causal factor (`differs`)."""

    clean: Series
    corrupted: Series
    differs: str

    @property
    def divergence(self) -> float:
        """|clean - corrupted| at the first forecast step."""
        return float(abs(self.clean.future[0] - self.corrupted.future[0]))


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------


def _finish(full: np.ndarray, length: int, noise: float, rng: np.random.Generator,
            noise_values: np.ndarray | None, meta: dict) -> Series:
    """Split a length+horizon array into context/future, adding shared-able noise."""
    if noise > 0:
        if noise_values is None:
            noise_values = rng.normal(0.0, noise, size=len(full))
        full = full + noise_values
    return Series(values=full[:length], future=full[length:], meta=meta)


def seasonal(period: int = 7, length: int = 140, horizon: int = 1,
             kind: str = "pattern", amplitude: float = 1.0, level: float = 0.0,
             noise: float = 0.0, phase: int = 0, seed: int = 0,
             pattern_seed: int | None = None,
             noise_values: np.ndarray | None = None) -> Series:
    """Periodic series. kind='pattern' tiles a random per-cycle pattern (the sharpest
    induction test: the only way to predict position i is to copy position i-P).
    kind='sine' is the smooth control."""
    rng = np.random.default_rng(seed)
    total = length + horizon
    if kind == "pattern":
        prng = np.random.default_rng(seed if pattern_seed is None else pattern_seed)
        pattern = prng.uniform(-1.0, 1.0, size=period) * amplitude
        idx = (np.arange(total) + phase) % period
        full = level + pattern[idx]
    elif kind == "sine":
        t = np.arange(total) + phase
        full = level + amplitude * np.sin(2 * np.pi * t / period)
    else:
        raise ValueError(f"unknown kind: {kind}")
    meta = dict(family="seasonal", period=period, kind=kind, amplitude=amplitude,
                level=level, noise=noise, phase=phase, seed=seed)
    return _finish(full, length, noise, rng, noise_values, meta)


def trend(slope: float = 0.02, length: int = 140, horizon: int = 1,
          level: float = 0.0, noise: float = 0.0, seed: int = 0,
          noise_values: np.ndarray | None = None) -> Series:
    rng = np.random.default_rng(seed)
    full = level + slope * np.arange(length + horizon)
    meta = dict(family="trend", slope=slope, level=level, noise=noise, seed=seed)
    return _finish(full, length, noise, rng, noise_values, meta)


def seasonal_trend(period: int = 7, slope: float = 0.02, length: int = 140,
                   horizon: int = 1, kind: str = "pattern", amplitude: float = 1.0,
                   noise: float = 0.0, seed: int = 0) -> Series:
    s = seasonal(period, length, horizon, kind, amplitude, 0.0, 0.0, 0, seed)
    t = trend(slope, length, horizon, 0.0, 0.0, seed)
    rng = np.random.default_rng(seed)
    full = np.concatenate([s.values + t.values, s.future + t.future])
    meta = dict(family="seasonal_trend", period=period, slope=slope, kind=kind,
                amplitude=amplitude, noise=noise, seed=seed)
    return _finish(full, length, noise, rng, None, meta)


def season_plus_changepoint(period: int = 7, length: int = 140, horizon: int = 1,
                            cp_at: int | None = None, cp_kind: str = "pattern",
                            kind: str = "pattern", amplitude: float = 1.0,
                            noise: float = 0.0, seed: int = 0) -> Series:
    """Seasonal series whose generating process changes at `cp_at` (default: 60%
    through the context). cp_kind: 'pattern' (new random cycle, same period),
    'phase' (half-period shift), 'period' (period -> period+3),
    'level' (2*amplitude offset added)."""
    if cp_at is None:
        cp_at = int(length * 0.6)
    pre = seasonal(period, length, horizon, kind, amplitude, 0.0, 0.0, 0, seed)
    full = np.concatenate([pre.values, pre.future]).copy()
    post_len = length + horizon - cp_at
    if cp_kind == "pattern":
        post = seasonal(period, post_len, 0, kind, amplitude, 0.0, 0.0, 0, seed,
                        pattern_seed=seed + 10_000)
        full[cp_at:] = post.values
    elif cp_kind == "phase":
        post = seasonal(period, post_len, 0, kind, amplitude, 0.0, 0.0,
                        phase=cp_at + period // 2, seed=seed)
        full[cp_at:] = post.values
    elif cp_kind == "period":
        post = seasonal(period + 3, post_len, 0, kind, amplitude, 0.0, 0.0, 0, seed)
        full[cp_at:] = post.values
    elif cp_kind == "level":
        full[cp_at:] += 2.0 * amplitude
    else:
        raise ValueError(f"unknown cp_kind: {cp_kind}")
    rng = np.random.default_rng(seed)
    meta = dict(family="season_plus_changepoint", period=period, cp_at=cp_at,
                cp_kind=cp_kind, kind=kind, amplitude=amplitude, noise=noise,
                seed=seed)
    return _finish(full, length, noise, rng, None, meta)


# ---------------------------------------------------------------------------
# minimal pairs
# ---------------------------------------------------------------------------


def _shared_noise(noise: float, n: int, seed: int) -> np.ndarray | None:
    if noise <= 0:
        return None
    return np.random.default_rng(seed + 77_000).normal(0.0, noise, size=n)


def period_pair(p_clean: int = 7, p_corr: int = 12, length: int = 140,
                horizon: int = 1, kind: str = "pattern", amplitude: float = 1.0,
                noise: float = 0.0, seed: int = 0,
                min_divergence: float = 0.5) -> MinimalPair:
    """Same amplitude, same noise realization, same length — only the period
    differs. Retries the corrupted pattern/phase until the two continuations
    diverge by >= min_divergence * amplitude at the first forecast step."""
    nv = _shared_noise(noise, length + horizon, seed)
    clean = seasonal(p_clean, length, horizon, kind, amplitude, 0.0, noise,
                     0, seed, noise_values=nv)
    for attempt in range(200):
        corr = seasonal(p_corr, length, horizon, kind, amplitude, 0.0, noise,
                        phase=attempt if kind == "sine" else 0, seed=seed,
                        pattern_seed=seed + 20_000 + attempt, noise_values=nv)
        if abs(clean.future[0] - corr.future[0]) >= min_divergence * amplitude:
            return MinimalPair(clean, corr, differs="period")
    raise RuntimeError("could not build a divergent period pair; lower min_divergence")


def changepoint_pair(period: int = 7, length: int = 140, horizon: int = 1,
                     cp_kind: str = "pattern", kind: str = "pattern",
                     amplitude: float = 1.0, noise: float = 0.0,
                     seed: int = 0) -> MinimalPair:
    """clean = stationary seasonal; corrupted = same series with a changepoint.
    Identical before cp_at."""
    nv = _shared_noise(noise, length + horizon, seed)
    clean = seasonal(period, length, horizon, kind, amplitude, 0.0, noise,
                     0, seed, noise_values=nv)
    corr = season_plus_changepoint(period, length, horizon, None, cp_kind, kind,
                                   amplitude, 0.0, seed)
    if nv is not None:
        full = np.concatenate([corr.values, corr.future]) + nv
        corr = Series(full[:length], full[length:], corr.meta)
    return MinimalPair(clean, corr, differs="changepoint")


def trend_pair(slope: float = 0.02, length: int = 140, horizon: int = 1,
               noise: float = 0.0, seed: int = 0) -> MinimalPair:
    """Up-trend vs down-trend, same magnitude and noise. For probing/steering a
    trend direction (RQ2)."""
    nv = _shared_noise(noise, length + horizon, seed)
    clean = trend(slope, length, horizon, 0.0, noise, seed, noise_values=nv)
    corr = trend(-slope, length, horizon, 0.0, noise, seed, noise_values=nv)
    return MinimalPair(clean, corr, differs="trend_sign")


def phase_pair(period: int = 7, length: int = 168, horizon: int = 1,
               shift: int | None = None, kind: str = "pattern",
               amplitude: float = 1.0, noise: float = 0.0, seed: int = 0,
               min_divergence: float = 0.5) -> MinimalPair:
    """Same period, same cycle pattern, same noise — only the phase differs
    (default shift: half a period). The sharpest test that a head tracks WHERE
    in the cycle we are, not just that a cycle exists."""
    if shift is None:
        shift = period // 2
    nv = _shared_noise(noise, length + horizon, seed)
    for attempt in range(200):
        pseed = seed + 30_000 + attempt
        clean = seasonal(period, length, horizon, kind, amplitude, 0.0, noise,
                         0, seed, pattern_seed=pseed, noise_values=nv)
        corr = seasonal(period, length, horizon, kind, amplitude, 0.0, noise,
                        shift, seed, pattern_seed=pseed, noise_values=nv)
        if abs(clean.future[0] - corr.future[0]) >= min_divergence * amplitude:
            return MinimalPair(clean, corr, differs="phase")
    raise RuntimeError("could not build a divergent phase pair")


def trend_onoff_pair(period: int = 7, slope: float = 0.03, length: int = 168,
                     horizon: int = 1, kind: str = "pattern",
                     amplitude: float = 1.0, noise: float = 0.0,
                     seed: int = 0) -> MinimalPair:
    """clean = seasonal + linear trend, corrupted = the same seasonal component
    with the trend removed. Same pattern, same noise; only the trend differs."""
    nv = _shared_noise(noise, length + horizon, seed)
    base = seasonal(period, length, horizon, kind, amplitude, 0.0, 0.0, 0, seed)
    ramp = trend(slope, length, horizon, 0.0, 0.0, seed)
    full_on = np.concatenate([base.values + ramp.values, base.future + ramp.future])
    full_off = np.concatenate([base.values, base.future])
    if nv is not None:
        full_on, full_off = full_on + nv, full_off + nv
    meta = dict(base.meta, slope=slope, noise=noise)
    clean = Series(full_on[:length], full_on[length:], dict(meta, trend="on"))
    corr = Series(full_off[:length], full_off[length:], dict(meta, trend="off"))
    return MinimalPair(clean, corr, differs="trend_onoff")


def changepoint_location_pair(period: int = 7, length: int = 168,
                              horizon: int = 1, cp_frac_clean: float = 0.4,
                              cp_frac_corr: float = 0.7, kind: str = "pattern",
                              amplitude: float = 1.0, noise: float = 0.0,
                              seed: int = 0,
                              min_divergence: float = 0.5) -> MinimalPair:
    """Both series switch from the same pre-pattern to the same post-pattern —
    only WHERE the changepoint sits differs. Isolates changepoint-location
    information from everything else. Retries pattern seeds until the two
    continuations diverge (different cp positions leave the post-pattern at
    different phases, but not every pattern separates at t+1)."""
    nv = _shared_noise(noise, length + horizon, seed)
    for attempt in range(200):
        s_eff = seed if attempt == 0 else seed + 40_000 + attempt
        out = []
        for frac in (cp_frac_clean, cp_frac_corr):
            s = season_plus_changepoint(period, length, horizon,
                                        cp_at=int(length * frac),
                                        cp_kind="pattern", kind=kind,
                                        amplitude=amplitude, noise=0.0,
                                        seed=s_eff)
            full = np.concatenate([s.values, s.future])
            if nv is not None:
                full = full + nv
            out.append(Series(full[:length], full[length:], s.meta))
        if abs(out[0].future[0] - out[1].future[0]) >= min_divergence * amplitude:
            return MinimalPair(out[0], out[1], differs="changepoint_location")
    raise RuntimeError("could not build a divergent changepoint-location pair")


# ---------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------


def _autocorr_peak_lag(x: np.ndarray, max_lag: int) -> int:
    x = x - x.mean()
    ac = np.array([np.dot(x[:-k], x[k:]) for k in range(1, max_lag + 1)])
    return int(np.argmax(ac)) + 1


def _self_test() -> None:
    # exact periodicity of the pattern generator
    s = seasonal(period=7, length=140, horizon=3, kind="pattern", seed=1)
    assert np.allclose(s.values[7:], s.values[:-7]), "pattern kind must tile exactly"
    assert np.isclose(s.future[0], s.values[140 - 7]), "future must continue the cycle"
    assert s.length == 140 and len(s.future) == 3

    # autocorrelation peak at the true period, even with noise
    s = seasonal(period=12, length=240, kind="pattern", noise=0.05, seed=2)
    assert _autocorr_peak_lag(s.values, 20) == 12

    # sine control
    s = seasonal(period=10, length=100, kind="sine", seed=0)
    assert np.isclose(s.values[0], 0.0, atol=1e-9)
    assert _autocorr_peak_lag(s.values, 15) == 10

    # trend
    t = trend(slope=0.05, length=50, horizon=2)
    assert np.isclose(t.future[1] - t.future[0], 0.05)

    # changepoint actually changes the process, and only after cp_at
    for cpk in ("pattern", "phase", "period", "level"):
        cp = season_plus_changepoint(period=7, length=140, cp_kind=cpk, seed=3)
        base = seasonal(period=7, length=140, seed=3)
        cp_at = cp.meta["cp_at"]
        assert np.allclose(cp.values[:cp_at], base.values[:cp_at]), cpk
        assert not np.allclose(cp.values[cp_at:], base.values[cp_at:]), cpk

    # period pair: aligned, shared noise, divergent
    pair = period_pair(7, 12, length=140, noise=0.05, seed=4)
    assert pair.clean.length == pair.corrupted.length == 140
    assert pair.divergence >= 0.5
    # noise realization has the right scale (shared across the pair)
    n_clean = pair.clean.values - seasonal(7, 140, 1, seed=4).values
    assert abs(np.std(n_clean) - 0.05) < 0.02

    # changepoint pair identical before the changepoint
    cpair = changepoint_pair(period=7, length=140, noise=0.05, seed=5)
    cp_at = cpair.corrupted.meta["cp_at"]
    assert np.allclose(cpair.clean.values[:cp_at], cpair.corrupted.values[:cp_at])
    assert not np.allclose(cpair.clean.values[cp_at:], cpair.corrupted.values[cp_at:])

    # trend pair mirrors around zero level
    tpair = trend_pair(slope=0.02, length=100, seed=6)
    assert np.isclose(tpair.clean.future[0], -tpair.corrupted.future[0],
                      atol=1e-9)

    # phase pair: same period & pattern values, shifted; divergent at t+1
    ppair = phase_pair(period=7, length=168, noise=0.05, seed=7)
    assert ppair.divergence >= 0.5
    c0 = phase_pair(period=7, length=168, seed=7)  # noiseless twin
    assert np.allclose(np.sort(np.unique(np.round(c0.clean.values, 9))),
                       np.sort(np.unique(np.round(c0.corrupted.values, 9)))), \
        "phase pair must reuse the same cycle values"
    shift = 7 // 2
    assert np.allclose(c0.clean.values[shift:], c0.corrupted.values[:-shift])

    # trend on/off: difference is exactly the ramp
    topair = trend_onoff_pair(period=7, slope=0.03, length=168, noise=0.05,
                              seed=8)
    d = topair.clean.values - topair.corrupted.values
    assert np.allclose(d, 0.03 * np.arange(168)), "on-off must differ by the ramp"
    assert topair.divergence > 3.0

    # changepoint-location pair: same pre and post patterns, different switch,
    # divergent continuation
    clpair = changepoint_location_pair(period=7, length=168, seed=9)
    cp_a = clpair.clean.meta["cp_at"]
    cp_b = clpair.corrupted.meta["cp_at"]
    assert cp_a < cp_b
    assert np.allclose(clpair.clean.values[:cp_a], clpair.corrupted.values[:cp_a])
    assert not np.allclose(clpair.clean.values[cp_a:cp_b],
                           clpair.corrupted.values[cp_a:cp_b])
    assert clpair.divergence >= 0.5

    print("synthetic.py: all self-tests passed")
    print(f"  example period pair divergence at t+1: {pair.divergence:.3f} "
          f"(amplitude 1.0)")


if __name__ == "__main__":
    _self_test()
