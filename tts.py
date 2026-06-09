"""
tts.py — Text-to-speech narration.

Primary: Kokoro-82M served on Modal (open weights — keeps every model in the
stack ≤ 4B parameters). Fallback: edge-tts cloud voices, for local dev when
Modal credentials are absent.

generate_speech() returns a path to a temporary audio file, or None on failure.
"""
import asyncio
import os
import tempfile
import threading

MODAL_APP = os.getenv("MODAL_APP_NAME", "storyforge")
_KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
_EDGE_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")


def _modal_ready() -> bool:
    return bool(os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))


def generate_speech(text: str) -> str | None:
    """Returns path to a temp audio file (WAV via Kokoro, MP3 via edge-tts), or None."""
    if not text or not text.strip():
        return None
    if _modal_ready():
        try:
            import modal

            TTSModel = modal.Cls.from_name(MODAL_APP, "TTSModel")
            wav = TTSModel().speak.remote(text.strip(), _KOKORO_VOICE)
            if wav:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(wav)
                tmp.close()
                return tmp.name
        except Exception as e:
            print(f"[tts] Modal Kokoro failed, falling back to edge-tts: {e}")
    return _edge_tts(text)


# ── edge-tts fallback ─────────────────────────────────────────────────────────

async def _save(text: str, path: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, _EDGE_VOICE)
    await communicate.save(path)


def _edge_tts(text: str) -> str | None:
    result: list = [None]
    error: list = [None]

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.close()
            loop.run_until_complete(_save(text.strip(), tmp.name))
            result[0] = tmp.name
        except Exception as e:
            error[0] = e
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=30)

    if error[0] or result[0] is None:
        return None
    return result[0]
