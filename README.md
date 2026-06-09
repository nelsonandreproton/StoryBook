---
title: StoryForge
emoji: 📖
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
license: mit
---

# 📖 StoryForge

An interactive branching picture-book adventure for children (age 4–8).

## Features

- **Branching story** — 2–6 choices per moment, up to 15 moments
- **Ghibli-style illustrations** — one image generated per beat
- **Character consistency** — IP-Adapter keeps your hero looking the same throughout
- **TTS narration** — each beat is read aloud by Kokoro-82M (open weights)
- **Voice input** — speak your choice instead of clicking
- **Ambient music** — procedural background audio matched to your theme
- **PDF export** — download your illustrated story as a printable picture book
- **Custom hero** — name and describe your own hero before the story begins

## HF Space secrets

Add these in your Space settings → **Secrets**:

| Secret | Where to get it |
|---|---|
| `MODAL_TOKEN_ID` | `modal token new` → token ID |
| `MODAL_TOKEN_SECRET` | `modal token new` → token secret |

Without Modal secrets, text falls back to local GGUF inference and images are skipped (torch/diffusers not installed on HF Spaces).

## Deploy the Modal backend

```bash
pip install modal
modal token new
modal deploy modal_inference.py
```

This deploys Qwen3-4B (text), Ghibli-Diffusion + IP-Adapter (images), Whisper small (STT) and Kokoro-82M (TTS) as serverless Modal functions.

## Tiny stack — every model ≤ 4B parameters

| Model | Role | Params |
|---|---|---|
| Qwen3-4B | story text | 4B |
| Ghibli-Diffusion (SD 1.5) | illustrations | ~0.9B |
| Kokoro-82M | narration | 82M |
| Whisper small | voice input | 244M |
| IP-Adapter (SD 1.5) | character consistency | ~22M adapter |

Total: well under 5.5B parameters for a fully multimodal experience.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

> Requires Python 3.10+. Deployed on HF Spaces with Python 3.13 + Gradio 4.44.0.
> The local image fallback needs `torch`+`diffusers`, and the local STT fallback needs `transformers` — install them separately if you run without Modal.

## Upload to HF Space

```bash
$env:HF_TOKEN="hf_..."
python3 upload_space.py
```

## Architecture

| Layer | File | Backend |
|---|---|---|
| Text generation | `model.py` | Modal A10G → Qwen3-4B · fallback: local GGUF |
| Image generation | `image_model.py` | Modal T4 → Ghibli-Diffusion + IP-Adapter · fallback: local CPU |
| TTS narration | `tts.py` | Modal CPU → Kokoro-82M · fallback: edge-tts |
| Voice input STT | `stt.py` | Modal T4 → Whisper · fallback: transformers whisper-tiny |
| Ambient audio | `ambient.py` | numpy procedural, CPU |
| PDF export | `pdf_export.py` | reportlab, CPU |
| Story engine | `engine.py` | Pure Python, stateless |
| UI | `app.py` | Gradio |

## Configuration

| Env var | Default | Description |
|---|---|---|
| `MODEL_REPO` | `Qwen/Qwen3-1.7B-GGUF` | GGUF text model (local fallback only) |
| `MODEL_FILE` | `Qwen3-1.7B-Q8_0.gguf` | Filename within that repo |
| `MODAL_APP_NAME` | `storyforge` | Name used when deploying to Modal |
| `KOKORO_VOICE` | `af_heart` | Kokoro-82M voice (Modal TTS) |
| `TTS_VOICE` | `en-US-JennyNeural` | edge-tts voice (local fallback only) |
| `N_CTX` | `4096` | Context window (local fallback) |
| `N_THREADS` | `cpu_count` | Inference threads (local fallback) |
