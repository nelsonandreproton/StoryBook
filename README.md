---
title: StoryForge
emoji: 📖
colorFrom: yellow
colorTo: pink
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
---

# StoryForge 📖

A branching, illustrated storybook web app for young children (ages 4–8). Pick a theme, make choices, and watch your unique adventure unfold — complete with hand-drawn Ghibli-style illustrations generated for every beat.

All AI runs **entirely in-process**. No cloud APIs. No data sent anywhere. Just stories.

## Hackathon tracks & badges

**Track:** 🍄 An Adventure in Thousand Token Wood — *something delightful that wouldn't exist without AI*

| Badge | How StoryForge earns it |
|-------|------------------------|
| 🔌 **Off the Grid** | Zero cloud LLM calls; text model runs in-process via `llama-cpp-python` |
| 🎨 **Off-Brand** | Custom storybook CSS — Fredoka/Nunito fonts, parchment palette, animated beat cards |
| 🦙 **Llama Champion** | Text generation via llama.cpp runtime (Qwen3-1.7B Q4_K_M GGUF) |
| 🎯 **Well-Tuned** | Illustrations use `nitrosocke/Ghibli-Diffusion` — a fine-tuned Stable Diffusion 1.5 |

## How it works

1. Choose a story theme (brave fox, sleepy moon, shy dragon…)
2. Read the opening beat of your adventure
3. A Ghibli-style illustration generates in the background — fades in when ready
4. Tap a choice to steer the story
5. Repeat until you reach a happy ending ✨

### Architecture

Two models, fully local:

| Model | Role | Size |
|-------|------|------|
| Qwen3-1.7B Q4_K_M (GGUF) | Story text & branching choices | ~1 GB |
| nitrosocke/Ghibli-Diffusion (SD 1.5) | Per-beat illustrations | ~1.7 GB |

**Text coherence:** the full `StoryState` (hero, world, established facts, history) is re-injected into every prompt — the model never needs to "remember" anything.

**Non-blocking images:** every beat yields text and options immediately; the illustration generates in a background thread and fades in after ~60–90s on CPU.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Both models download automatically from Hugging Face on first run (~2.7 GB total).

### Alternate text model

```bash
MODEL_REPO=Qwen/Qwen3-4B-GGUF MODEL_FILE=Qwen3-4B-Q4_K_M.gguf python app.py
```

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `MODEL_REPO` | `Qwen/Qwen3-1.7B-GGUF` | HF repo for the GGUF text model |
| `MODEL_FILE` | `Qwen3-1.7B-Q4_K_M.gguf` | Filename within that repo |
| `N_CTX` | `4096` | Context window size |
| `N_THREADS` | `cpu_count` | Inference threads |

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main Gradio app (v2, with images) |
| `engine_v2.py` | Story logic + image-prompt builder |
| `model.py` | llama-cpp-python wrapper |
| `image_model_v2.py` | Ghibli-Diffusion pipeline |
| `styles_v2.css` | Custom storybook theme |
| `app_v1.py` | Original v1 (text-only) |

## Privacy

No network calls during inference. The only outbound requests are the one-time model downloads from Hugging Face Hub.
