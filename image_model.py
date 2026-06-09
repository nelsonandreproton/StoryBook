"""
image_model.py — Image generation with optional character consistency.
Uses Modal GPU (Ghibli-Diffusion + IP-Adapter) when Modal credentials are set,
falls back to local CPU pipeline otherwise.
"""
import io
import os
from functools import lru_cache

MODAL_APP = os.getenv("MODAL_APP_NAME", "storyforge")
_STEPS_LOCAL = 10
_SIZE_LOCAL = 384


def _modal_ready() -> bool:
    return bool(os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))


_NEGATIVE = (
    "border, frame, box, square, panel, grid, letterbox, vignette, "
    "text, watermark, signature, logo, blur, dark, ugly, deformed"
)


def generate_image(prompt: str, reference_bytes: bytes = None) -> bytes:
    """Return PNG bytes. reference_bytes enables IP-Adapter character consistency."""
    if _modal_ready():
        try:
            import modal

            ImageModel = modal.Cls.from_name(MODAL_APP, "ImageModel")
            return ImageModel().generate.remote(prompt, reference_bytes, _NEGATIVE)
        except Exception as e:
            import traceback
            print(f"[image_model] Modal call failed: {e}")
            traceback.print_exc()
            return None
    return _generate_local(prompt)


@lru_cache(maxsize=1)
def _load_local():
    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "nitrosocke/Ghibli-Diffusion",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cpu")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _generate_local(prompt: str) -> bytes:
    pipe = _load_local()
    result = pipe(
        prompt,
        num_inference_steps=_STEPS_LOCAL,
        guidance_scale=6.0,
        height=_SIZE_LOCAL,
        width=_SIZE_LOCAL,
    )
    buf = io.BytesIO()
    result.images[0].save(buf, format="PNG")
    return buf.getvalue()
