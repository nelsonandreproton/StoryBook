"""
ambient.py — Procedural ambient audio per theme.
Returns (sample_rate, numpy_int16_array) for gr.Audio.
Pure numpy — no external audio libraries required.

Design notes: noise beds are double-box-smoothed white noise (O(n) via
cumsum — np.convolve is far too slow for minutes of audio) with slow LFO
envelopes; melodic events (birds, bells) get proper attack/release envelopes
so nothing clicks. Raw sine drones sound like mains hum — avoid them.
"""
import numpy as np

SR = 22050
_PEAK = 0.10


def generate_ambient(theme: str, duration: float = 120.0):
    """Return (sample_rate, audio_array) matching the story theme."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, dtype=np.float64)
    tl = theme.lower()
    rng = np.random.default_rng(42)

    if "moon" in tl or "space" in tl:
        audio = _space(t, n, rng)
    elif "sea" in tl or "monster" in tl:
        audio = _ocean(t, n, rng)
    elif "dragon" in tl:
        audio = _magic(t, n, rng)
    elif "balloon" in tl:
        audio = _sky(t, n, rng)
    else:
        audio = _forest(t, n, rng)  # fox, forest, default

    # Normalise and fade edges for clean start/end
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * _PEAK
    fade = min(int(SR * 2.0), n // 4)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    # Convert to int16 — Gradio expects integer PCM to avoid auto-conversion warnings
    return SR, (audio * 32767).astype(np.int16)


# ── Building blocks ───────────────────────────────────────────────────────────

def _smooth(x, k):
    """O(n) box smoothing via cumsum."""
    k = max(int(k), 1)
    if k == 1:
        return x
    c = np.cumsum(np.concatenate(([0.0], x)))
    out = (c[k:] - c[:-k]) / k
    pad = np.full(k - 1, out[-1] if len(out) else 0.0)
    return np.concatenate((out, pad))


def _norm(x):
    p = np.abs(x).max()
    return x / p if p > 0 else x


def _wind(n, rng, k1, k2, lfo_hz, t, depth=0.5):
    """Double-smoothed noise bed with a slow amplitude LFO — wind/leaves/waves."""
    noise = rng.standard_normal(n)
    bed = _smooth(_smooth(noise, k1), k2)
    phase = rng.uniform(0, 2 * np.pi)
    env = (1 - depth) + depth * 0.5 * (1 + np.sin(2 * np.pi * lfo_hz * t + phase))
    return bed * env


# ── Themes ────────────────────────────────────────────────────────────────────

def _forest(t, n, rng):
    """Leafy wind bed + soft birdsong (real chirps with envelopes, not clicks)."""
    bed = _wind(n, rng, SR // 60, SR // 200, 0.05, t, depth=0.45)
    birds = np.zeros(n)
    pos = SR * 3
    while pos < n - SR:
        length = int(SR * rng.uniform(0.08, 0.16))
        lt = np.linspace(0, 1, length)
        f0 = rng.uniform(1800, 3200)
        f1 = f0 * rng.uniform(0.75, 0.95)
        # Downward chirp; phase integrates the frequency sweep over real time
        phase = 2 * np.pi * (f0 * lt + 0.5 * (f1 - f0) * lt**2) * (length / SR)
        env = np.sin(np.pi * lt) ** 2
        birds[pos:pos + length] += env * np.sin(phase)
        if rng.random() < 0.5:                       # quick second note
            pos += int(length * rng.uniform(1.2, 1.8))
        else:
            pos += int(SR * rng.uniform(2.5, 6.0))
    return _norm(bed) * 0.7 + _norm(birds) * 0.3


def _ocean(t, n, rng):
    """Two overlapping wave sets at different speeds."""
    surf = _wind(n, rng, SR // 30, SR // 150, 0.08, t, depth=0.7)
    swell = _wind(n, rng, SR // 15, SR // 100, 0.05, t, depth=0.8)
    return _norm(surf) * 0.6 + _norm(swell) * 0.5


def _space(t, n, rng):
    """Slowly beating warm pad (close detuned partials) + airy shimmer."""
    pad = np.sin(2 * np.pi * 110.0 * t) + np.sin(2 * np.pi * 110.7 * t)
    pad += 0.5 * np.sin(2 * np.pi * 165.3 * t)       # soft fifth
    pad *= 0.55 + 0.45 * np.sin(2 * np.pi * 0.04 * t)
    air = _wind(n, rng, SR // 400, SR // 800, 0.07, t, depth=0.6)
    return _norm(pad) * 0.55 + _norm(air) * 0.3


def _magic(t, n, rng):
    """Soft bell arpeggio with randomized timing over a breathing low pad."""
    freqs = [261.63, 329.63, 392.0, 523.25, 659.25]
    bells = np.zeros(n)
    pos = int(SR * 0.5)
    i = 0
    while pos < n - SR * 3:
        f = freqs[i % len(freqs)] * (2.0 if rng.random() < 0.15 else 1.0)
        length = int(SR * 2.8)
        lt = np.linspace(0, 2.8, length)
        env = np.exp(-lt * 1.8) * np.minimum(lt * 40, 1.0)   # fast attack, long decay
        bells[pos:pos + length] += env * (
            np.sin(2 * np.pi * f * lt) + 0.4 * np.sin(2 * np.pi * 2 * f * lt)
        )
        pos += int(SR * rng.uniform(2.0, 3.5))
        i += 1
    pad = np.sin(2 * np.pi * 130.81 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t))
    return _norm(bells) * 0.7 + _norm(pad) * 0.18


def _sky(t, n, rng):
    """Open-air breeze: broad mid noise + brighter high layer."""
    breeze = _wind(n, rng, SR // 150, SR // 400, 0.06, t, depth=0.6)
    high = _wind(n, rng, SR // 700, SR // 1200, 0.09, t, depth=0.7)
    return _norm(breeze) * 0.6 + _norm(high) * 0.3
