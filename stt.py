"""
stt.py — Speech-to-text + option matching.
Uses Modal GPU Whisper when credentials are set, falls back to transformers whisper-tiny.
"""
import os
from difflib import SequenceMatcher
from functools import lru_cache

MODAL_APP = os.getenv("MODAL_APP_NAME", "storyforge")


def _modal_ready() -> bool:
    return bool(os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))


@lru_cache(maxsize=2)
def _local_model(model_id: str):
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        return_timestamps=False,
    )


def transcribe(audio_tuple, language: str = "en") -> str | None:
    """
    audio_tuple: (sample_rate, numpy_array) from gr.Audio(type='numpy').
    language: whisper language code ("en", "pt", ...).
    Returns transcribed text, or None on failure.
    """
    if audio_tuple is None:
        return None
    try:
        import io
        import numpy as np
        import soundfile as sf

        sample_rate, audio = audio_tuple
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        if _modal_ready():
            try:
                import modal

                buf = io.BytesIO()
                sf.write(buf, audio, sample_rate, format="WAV")
                STTModel = modal.Cls.from_name(MODAL_APP, "STTModel")
                return STTModel().transcribe.remote(buf.getvalue(), language)
            except Exception:
                pass

        # Local fallback — the .en checkpoint is better for English, the
        # multilingual tiny model covers everything else.
        model_id = "openai/whisper-tiny.en" if language == "en" else "openai/whisper-tiny"
        asr = _local_model(model_id)
        result = asr({"array": audio, "sampling_rate": sample_rate})
        return result["text"].strip()
    except Exception:
        return None


def match_option(text: str, options: list[str]) -> int | None:
    """Return the index of the best-matching option, or None if confidence is too low."""
    if not text or not options:
        return None
    text_lower = text.lower()
    best_score = 0.0
    best_idx = None
    for i, opt in enumerate(options):
        seq = SequenceMatcher(None, text_lower, opt.lower()).ratio()
        t_words = set(text_lower.split())
        o_words = set(opt.lower().split())
        overlap = len(t_words & o_words) / max(len(o_words), 1)
        score = max(seq, overlap)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx if best_score > 0.15 else None
