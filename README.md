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
tags:
  - build-small-hackathon
short_description: Branching picture-book adventures from a tiny-model stack
---

# 📖 StoryForge

**A magical branching picture-book adventure for children (age 4–8) — written, illustrated and narrated live by a stack of tiny open models (every one of them ≤ 4B parameters).**

<!-- TODO(nelson): replace the three links below before submitting -->
🚀 [Try it live](TODO-space-url) · 🎬 [Demo video](TODO-video-url) · 📣 [Social post](TODO-post-url)

<!-- TODO(nelson): record a short GIF of one story turn and commit it as docs/demo.gif -->
![StoryForge demo](docs/demo.gif)

Pick a theme, meet your hero, and steer the story with buttons **or your voice**. Each moment is illustrated in a Ghibli-inspired style — with your hero kept consistent across pages — read aloud by a 82M-parameter narrator, and at the end you download the whole adventure as a printable picture-book PDF.

## Why small models? (the trick)

A 4B model can't remember a 10-turn story — so we never ask it to. The engine keeps every canonical fact (hero, world, established facts, full beat history) in an explicit `StoryState`, re-injects all of it into the prompt **every turn**, and asks for exactly one grounded next beat as JSON. Explicit state in, one small step out: coherence lives in 100 lines of Python, not in the model's memory. That's what makes a tiny stack feel big.

## Tiny stack — every model ≤ 4B parameters

| Model | Role | Params |
|---|---|---|
| Qwen3-4B | story text | 4B |
| Ghibli-Diffusion (SD 1.5) | illustrations | ~0.9B |
| Kokoro-82M | narration | 82M |
| Whisper small | voice input | 244M |
| IP-Adapter (SD 1.5) | character consistency | ~22M adapter |

**Total: ~5.2B parameters** for a fully multimodal experience — text, images, narration, voice control, music and PDF export.

## Features

- **Branching story** — 2–6 choices per moment, up to 15 moments, streamed word-by-word as the model writes
- **Ghibli-style illustrations** — one image per beat; IP-Adapter keeps your hero looking the same throughout
- **TTS narration** — each beat read aloud by Kokoro-82M (open weights)
- **Voice input** — speak your choice instead of clicking (Whisper)
- **Two story languages** — English and Português (text, narration and voice input)
- **Ambient music** — procedural background audio matched to your theme (pure numpy, 0 parameters!)
- **Custom hero** — name and describe your own hero before the story begins
- **PDF export** — download your illustrated story as a printable picture book

## Architecture

Serverless inference on **Modal** (four model classes on one app), with graceful local fallbacks:

| Layer | File | Backend |
|---|---|---|
| Text generation | `model.py` | Modal A10G → Qwen3-4B (streaming) · fallback: local GGUF |
| Image generation | `image_model.py` | Modal T4 → Ghibli-Diffusion + IP-Adapter · fallback: local CPU |
| TTS narration | `tts.py` | Modal CPU → Kokoro-82M · fallback: edge-tts |
| Voice input STT | `stt.py` | Modal T4 → Whisper · fallback: transformers whisper-tiny |
| Ambient audio | `ambient.py` | numpy procedural, CPU |
| PDF export | `pdf_export.py` | reportlab, CPU |
| Story engine | `engine.py` | Pure Python, stateless |
| UI | `app.py` | Gradio (fully custom storybook theme, CSS only) |

## Run it yourself

### 1. Deploy the Modal backend

```bash
pip install modal
modal token new
modal deploy modal_inference.py
```

### 2. HF Space secrets

| Secret | Where to get it |
|---|---|
| `MODAL_TOKEN_ID` | `modal token new` → token ID |
| `MODAL_TOKEN_SECRET` | `modal token new` → token secret |

Without Modal secrets, text falls back to local GGUF inference and images are skipped (torch/diffusers not installed on HF Spaces).

### 3. Run locally

```bash
pip install -r requirements.txt
python app.py
```

> Requires Python 3.10+. Deployed on HF Spaces with Python 3.13 + Gradio 4.44.0.
> The local image fallback needs `torch`+`diffusers`, and the local STT fallback needs `transformers` — install them separately if you run without Modal.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `MODEL_REPO` | `Qwen/Qwen3-1.7B-GGUF` | GGUF text model (local fallback only) |
| `MODEL_FILE` | `Qwen3-1.7B-Q8_0.gguf` | Filename within that repo |
| `MODAL_APP_NAME` | `storyforge` | Name used when deploying to Modal |
| `KOKORO_VOICE` | `af_heart` | Kokoro voice for English |
| `KOKORO_VOICE_PT` | `pf_dora` | Kokoro voice for Português |
| `TTS_VOICE` | `en-US-JennyNeural` | edge-tts English voice (local fallback only) |
| `TTS_VOICE_PT` | `pt-PT-RaquelNeural` | edge-tts Português voice (local fallback only) |
| `N_CTX` | `4096` | Context window (local fallback) |
| `N_THREADS` | `cpu_count` | Inference threads (local fallback) |

## Built for the Build Small Hackathon

Submitted to the **Thousand Token Wood** track — a strange, delightful interactive story where the AI is doing the fun thing. Also at home in **Backyard AI**: built to read bedtime stories with a real kid.
