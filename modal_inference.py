"""
StoryForge — Modal serverless GPU backend.

Deploy once:
    pip install modal
    modal token new
    modal deploy modal_inference.py

Then set MODAL_TOKEN_ID + MODAL_TOKEN_SECRET in your HF Space secrets.
"""
import modal

APP_NAME = "storyforge"
app = modal.App(APP_NAME)

model_vol = modal.Volume.from_name("storyforge-models", create_if_missing=True)

# ── Text generation — Qwen3-4B on A10G ───────────────────────────────────────

text_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51.0",
        "accelerate",
        "huggingface_hub",
        "sentencepiece",
    )
)


@app.cls(
    image=text_image,
    gpu="A10G",
    volumes={"/models": model_vol},
    timeout=120,
    scaledown_window=300,
)
class TextModel:
    model_id: str = "Qwen/Qwen3-4B"

    @modal.enter()
    def load(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.tok = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir="/models", trust_remote_code=True
        )
        self.mdl = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="/models",
            trust_remote_code=True,
        )

    @modal.method()
    def generate(self, system: str, user: str, max_tokens: int = 512) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tok(text, return_tensors="pt").to(self.mdl.device)
        with torch.no_grad():
            out = self.mdl.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tok.eos_token_id,
            )
        new_tok = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(new_tok, skip_special_tokens=True)


# ── Image generation — Ghibli-Diffusion + IP-Adapter on T4 ───────────────────

img_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "diffusers>=0.30.0",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "Pillow",
    )
)


@app.cls(
    image=img_image,
    gpu="T4",
    volumes={"/models": model_vol},
    timeout=120,
    scaledown_window=300,
)
class ImageModel:
    model_id: str = "nitrosocke/Ghibli-Diffusion"

    @modal.enter()
    def load(self):
        import torch
        from diffusers import StableDiffusionPipeline

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            cache_dir="/models",
        ).to("cuda")
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin",
        )
        self.pipe.set_progress_bar_config(disable=True)

    @modal.method()
    def generate(
        self,
        prompt: str,
        reference_bytes: bytes = None,
        steps: int = 20,
        size: int = 512,
    ) -> bytes:
        import io
        from PIL import Image

        kwargs = dict(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
            height=size,
            width=size,
        )
        if reference_bytes:
            ref = Image.open(io.BytesIO(reference_bytes)).convert("RGB")
            self.pipe.set_ip_adapter_scale(0.5)
            kwargs["ip_adapter_image"] = ref
        else:
            self.pipe.set_ip_adapter_scale(0.0)
            kwargs["ip_adapter_image"] = Image.new("RGB", (size, size), (128, 128, 128))

        result = self.pipe(**kwargs)
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return buf.getvalue()


# ── STT — Whisper small on T4 ─────────────────────────────────────────────────

stt_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai-whisper", "torch", "numpy")
    .apt_install("ffmpeg")
)


@app.cls(
    image=stt_image,
    gpu="T4",
    timeout=60,
    scaledown_window=300,
)
class STTModel:
    @modal.enter()
    def load(self):
        import whisper
        self.model = whisper.load_model("small")

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            path = f.name
        try:
            result = self.model.transcribe(path, language="en")
            return result["text"].strip()
        finally:
            os.unlink(path)
