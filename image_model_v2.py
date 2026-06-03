"""
image_model_v2.py — Ghibli-Diffusion image generation for StoryForge v2.

Model: nitrosocke/Ghibli-Diffusion (SD 1.5 fine-tune, ~1.7 GB)
CPU-only; no GPU required. All heavy imports are deferred until first call
so startup memory stays low on HF free-tier Spaces.
"""

import io
from functools import lru_cache

MODEL_ID = "nitrosocke/Ghibli-Diffusion"
_STEPS = 10
_GUIDANCE = 6.0
_SIZE = 384


@lru_cache(maxsize=1)
def _load_pipeline():
    # Deferred imports — torch + diffusers are not loaded until first image request
    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_image(prompt: str) -> bytes:
    """Return PNG bytes for *prompt*, or raise on failure."""
    pipe = _load_pipeline()
    result = pipe(
        prompt,
        num_inference_steps=_STEPS,
        guidance_scale=_GUIDANCE,
        height=_SIZE,
        width=_SIZE,
    )
    img = result.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Latency smoke-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    TEST_PROMPT = (
        "ghibli style, a small brave fox standing at the edge of an enchanted forest, "
        "soft morning light, watercolor, children's book illustration"
    )

    print("Loading pipeline...")
    t0 = time.perf_counter()
    _load_pipeline()
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    print("Generating image...")
    t1 = time.perf_counter()
    png_bytes = generate_image(TEST_PROMPT)
    gen_time = time.perf_counter() - t1
    print(f"  Generated in {gen_time:.1f}s  ({len(png_bytes)//1024} KB)")

    out = "test_image_v2.png"
    with open(out, "wb") as f:
        f.write(png_bytes)
    print(f"  Saved -> {out}")
    print(f"\nTotal wall time: {load_time + gen_time:.1f}s")
