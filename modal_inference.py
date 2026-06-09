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
    scaledown_window=120,
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

    @modal.method()
    def generate_stream(self, system: str, user: str, max_tokens: int = 512):
        """Generator variant — yields decoded text pieces as they are produced."""
        import threading

        from transformers import TextIteratorStreamer

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
        streamer = TextIteratorStreamer(
            self.tok, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tok.eos_token_id,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.mdl.generate, kwargs=gen_kwargs)
        thread.start()
        for piece in streamer:
            if piece:
                yield piece
        thread.join()


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
    scaledown_window=120,
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
        self.pipe.set_progress_bar_config(disable=True)
        # Explicitly unload any cached IP-Adapter state from volume
        self._ip_loaded = False
        try:
            self.pipe.unload_ip_adapter()
        except Exception:
            pass

    @modal.method()
    def generate(
        self,
        prompt: str,
        reference_bytes: bytes = None,
        negative_prompt: str = "",
        steps: int = 20,
        size: int = 512,
    ) -> bytes:
        import io
        from PIL import Image

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or "border, frame, box, ugly, blurry, deformed",
            num_inference_steps=steps,
            guidance_scale=8.5,
            height=size,
            width=size,
        )
        if reference_bytes:
            try:
                # Load once per container, not once per beat — the adapter
                # weights are identical every call and loading burns GPU time.
                if not self._ip_loaded:
                    self.pipe.load_ip_adapter(
                        "h94/IP-Adapter",
                        subfolder="models",
                        weight_name="ip-adapter_sd15.bin",
                    )
                    self._ip_loaded = True
                self.pipe.set_ip_adapter_scale(0.5)
                ref = Image.open(io.BytesIO(reference_bytes)).convert("RGB")
                kwargs["ip_adapter_image"] = ref
            except Exception:
                pass
        elif self._ip_loaded:
            # New story on a warm container: no reference means no adapter —
            # a loaded adapter without ip_adapter_image TypeErrors in diffusers.
            try:
                self.pipe.unload_ip_adapter()
            except Exception:
                pass
            self._ip_loaded = False

        result = self.pipe(**kwargs)
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return buf.getvalue()


# ── TTS — Kokoro-82M (open weights; CPU is plenty for an 82M model) ──────────

tts_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng")
    .pip_install("kokoro>=0.9.4", "soundfile", "numpy")
)


@app.cls(
    image=tts_image,
    cpu=2,
    volumes={"/models": model_vol},
    timeout=60,
    scaledown_window=120,
)
class TTSModel:
    @modal.enter()
    def load(self):
        import os

        os.environ.setdefault("HF_HOME", "/models/hf")
        from kokoro import KPipeline

        # American English eagerly; other languages created lazily per lang_code
        self.pipelines = {"a": KPipeline(lang_code="a")}

    def _pipeline(self, lang_code: str):
        if lang_code not in self.pipelines:
            from kokoro import KPipeline

            self.pipelines[lang_code] = KPipeline(lang_code=lang_code)
        return self.pipelines[lang_code]

    @modal.method()
    def speak(self, text: str, voice: str = "af_heart", lang_code: str = "a") -> bytes:
        import io

        import numpy as np
        import soundfile as sf

        chunks = []
        for result in self._pipeline(lang_code)(text, voice=voice):
            audio = result[2] if isinstance(result, tuple) else result.audio
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            return b""
        buf = io.BytesIO()
        sf.write(buf, np.concatenate(chunks), 24000, format="WAV")
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
    scaledown_window=120,
)
class STTModel:
    @modal.enter()
    def load(self):
        import whisper
        self.model = whisper.load_model("small")

    @modal.method()
    def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            path = f.name
        try:
            result = self.model.transcribe(path, language=language)
            return result["text"].strip()
        finally:
            os.unlink(path)
