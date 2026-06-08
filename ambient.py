"""
ambient.py — Procedural ambient audio per theme.
Returns (sample_rate, numpy_float32_array) for gr.Audio.
Pure numpy — no external audio libraries required.
"""
import numpy as np

SR = 22050


def generate_ambient(theme: str, duration: float = 40.0):
    """Return (sample_rate, audio_array) matching the story theme."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, dtype=np.float64)
    tl = theme.lower()

    if "moon" in tl or "space" in tl:
        audio = _space(t, n)
    elif "sea" in tl or "monster" in tl:
        audio = _ocean(t, n)
    elif "dragon" in tl:
        audio = _magic(t, n)
    elif "balloon" in tl:
        audio = _sky(t, n)
    else:
        audio = _forest(t, n)  # fox, forest, default

    # Normalise and fade edges for clean start/end
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.25
    fade = min(int(SR * 1.5), n // 4)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    # Convert to int16 — Gradio expects integer PCM to avoid auto-conversion warnings
    return SR, (audio * 32767).astype(np.int16)


def _space(t, n):
    sweep = 55.0 * (1 + 0.15 * np.sin(2 * np.pi * 0.03 * t))
    phase = np.cumsum(sweep) * (2 * np.pi / SR)
    a = 0.15 * np.sin(phase)
    a += 0.05 * np.sin(2 * phase)
    a += 0.02 * np.sin(3 * phase)
    a *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t)
    return a


def _ocean(t, n):
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n)
    k = max(SR // 80, 1)
    noise = np.convolve(noise, np.ones(k) / k, mode="same")
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.12 * t)
    return 0.22 * noise * env


def _forest(t, n):
    rng = np.random.default_rng(7)
    base = 0.10 * np.sin(2 * np.pi * 110 * t)
    base *= 0.7 + 0.3 * np.sin(2 * np.pi * 0.06 * t)
    chirp_mask = (rng.random(n) > 0.9998).astype(np.float64)
    chirp = 0.05 * np.sin(2 * np.pi * 880 * t) * chirp_mask
    return base + chirp


def _magic(t, n):
    freqs = [261.63, 329.63, 392.0, 523.25, 659.25]
    audio = np.zeros(n)
    step = int(SR * 2.5)
    for i in range(0, n, step):
        length = min(int(SR * 2.0), n - i)
        freq = freqs[(i // step) % len(freqs)]
        lt = np.linspace(0, 2.0, length)
        bell = np.exp(-lt * 2.5) * np.sin(2 * np.pi * freq * lt)
        audio[i : i + length] += 0.14 * bell
    return audio


def _sky(t, n):
    rng = np.random.default_rng(13)
    noise = rng.standard_normal(n)
    k = max(SR // 200, 1)
    noise = np.convolve(noise, np.ones(k) / k, mode="same")
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t)
    return 0.14 * noise * env
