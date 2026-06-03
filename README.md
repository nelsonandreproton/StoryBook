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

A branching, illustrated storybook web app for young children (ages 4–8). Pick a theme, make choices, and watch your unique adventure unfold — all powered by a tiny AI running **entirely on your machine**. No cloud. No data sent anywhere. Just stories.

## Hackathon tracks

- 🏡 **Backyard AI** — built for a real child, runs on a free CPU Space
- 🌲 **Off the Grid** — zero cloud LLM calls; model runs in-process via `llama-cpp-python`
- 🎨 **Off-Brand** — custom storybook UI, not the default Gradio look

## How it works

1. Choose a story theme (brave fox, sleepy moon, shy dragon…)
2. Read the opening beat of your adventure
3. Tap a choice to steer the story
4. Repeat until you reach a happy ending ✨

All inference runs locally using a small GGUF model (Qwen3-1.7B Q4_K_M by default). Story coherence is maintained by re-injecting the canonical story facts into every prompt — the model never needs to "remember" anything.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The model (~1 GB) downloads automatically from Hugging Face on first run.

To use a different model:

```bash
MODEL_REPO=Qwen/Qwen3-4B-GGUF MODEL_FILE=Qwen3-4B-Q4_K_M.gguf python app.py
```

## Configuration

| Env var | Default | Description |
|---|---|---|
| `MODEL_REPO` | `Qwen/Qwen3-1.7B-GGUF` | HF repo for the GGUF model |
| `MODEL_FILE` | `Qwen3-1.7B-Q4_K_M.gguf` | Filename within that repo |
| `N_CTX` | `4096` | Context window size |
| `N_THREADS` | `cpu_count` | Inference threads |

## All inference is local

StoryForge uses `llama-cpp-python` to run a quantized GGUF model in-process. No requests are made to OpenAI, Anthropic, the HF Inference API, or any other external LLM service. The only network call is the one-time model download from Hugging Face Hub.
