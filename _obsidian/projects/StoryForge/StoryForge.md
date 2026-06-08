---
type: project
status: active
stack: python
last_sync: 2026-06-08
tags: [active, python, ai, gradio]
---

# StoryForge

> Interactive branching picture-book adventure for children (age 4–8), powered by local + cloud AI.

#active

---

## Overview

| Field | Value |
|-------|-------|
| **Status** | Active |
| **Last synced** | 2026-06-08 |
| **Location** | `C:\dev\StoryBook` |
| **GitHub** | https://github.com/nelsonandreproton/StoryBook |
| **Deployment** | HF Space — https://huggingface.co/spaces/nelsondiasandre/StoryForge |

---

## Stack

- **Runtime:** Python 3.11
- **Language:** Python
- **Framework:** Gradio 4.x / 6.x
- **Database:** None (stateless, session state only)
- **Infrastructure:** Modal (GPU inference) + HF Spaces (hosting)

---

## Services

| Service | Where | Notes |
|---------|-------|-------|
| Text generation | Modal A10G | Qwen3-4B via transformers · fallback: local GGUF |
| Image generation | Modal T4 | Ghibli-Diffusion + IP-Adapter · fallback: local CPU |
| TTS narration | edge-tts (internet) | Microsoft Jenny neural voice, no GPU |
| Voice STT | Modal T4 | Whisper small · fallback: whisper-tiny CPU |
| Ambient audio | CPU (numpy) | Procedural per theme |
| PDF export | CPU (reportlab) | Full illustrated storybook |

---

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio UI — all features wired, 18-output tuple |
| `engine_v2.py` | Story logic + image-prompt builder, stateless |
| `model.py` | Text gen wrapper (Modal → local fallback) |
| `image_model_v2.py` | Image gen wrapper (Modal + IP-Adapter → local fallback) |
| `modal_inference.py` | Modal serverless GPU backend — deploy once |
| `tts.py` | edge-tts narration |
| `stt.py` | Whisper STT + option matching |
| `pdf_export.py` | reportlab illustrated PDF |
| `ambient.py` | Numpy procedural ambient audio |
| `styles_v2.css` | Custom storybook theme (Fredoka, parchment palette) |

---

## Environment Variables

```
MODAL_TOKEN_ID=ak-...
MODAL_TOKEN_SECRET=as-...
MODAL_APP_NAME=storyforge        # default
TTS_VOICE=en-US-JennyNeural      # default
MODEL_REPO=Qwen/Qwen3-1.7B-GGUF  # local fallback only
MODEL_FILE=Qwen3-1.7B-Q8_0.gguf
N_CTX=4096
```

---

## HF Space Secrets

Set in Space settings → Secrets:
- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

---

## Modal Backend Setup

```bash
pip install modal
modal token new
modal deploy modal_inference.py
```

Modal credentials in `~/.modal.toml` after `modal token new`.

---

## Credits

- **Modal**: $250 GPU credits — A10G for text (Qwen3-4B), T4 for images + STT
- **HF ZeroGPU**: $20 credits — extended Space runtime
- **Models**: Qwen3-4B, Ghibli-Diffusion (nitrosocke), Whisper (OpenAI)

---

## Links

- [[system]] — back to overview
- [[projects/StoryForge/StoryForge-history]]
- [[patterns/stack]]

---

## Open Items

- [ ] Set MODAL_TOKEN_ID + MODAL_TOKEN_SECRET in HF Space secrets
- [ ] Run `modal deploy modal_inference.py` to activate GPU backend
- [ ] Test voice input on HF Space
- [ ] Test PDF export end-to-end

---

## Notes

Started as a hackathon project (Off-the-Grid badge). Evolved into a full-featured
GPU-accelerated illustrated storybook with narration, voice input, ambient music,
character consistency, and PDF export.
