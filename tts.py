"""
tts.py — Text-to-speech via edge-tts (Microsoft neural voices, no GPU needed).
Returns a path to a temporary MP3 file, or None on failure.
"""
import asyncio
import os
import tempfile
import threading


_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")


async def _save(text: str, path: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, _VOICE)
    await communicate.save(path)


def generate_speech(text: str) -> str | None:
    """Returns path to a temp MP3 file, or None on failure."""
    if not text or not text.strip():
        return None
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
