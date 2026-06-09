# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app
python app.py

# Smoke test (3-turn validation; without Modal creds it downloads a ~1.8 GB GGUF on first run)
python smoke_test.py

# Install deps
pip install -r requirements.txt

# Deploy the GPU/CPU backends (text, image, TTS, STT) to Modal
modal deploy modal_inference.py
```

Models load on first call and are cached via `@lru_cache`. Restart the process to reload.

To use a different local text model:
```bash
MODEL_REPO=Qwen/Qwen3-4B-GGUF MODEL_FILE=Qwen3-4B-Q4_K_M.gguf python app.py
```

## Architecture

Every inference module follows the same pattern: **Modal backend when `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` are set, graceful local fallback otherwise.** All Modal classes live in `modal_inference.py` (app name `storyforge`).

**`engine.py`** — Pure story logic, no I/O. `StoryState` dataclass holds all canonical facts (hero, world, facts list, history). `build_prompt()` re-injects the full state every turn so the model never needs memory. `parse_response()` strips Qwen3 `<think>` blocks and CJK noise, extracts JSON, salvages truncated beats. `extract_partial_beat()` pulls beat text out of incomplete JSON during streaming. `apply_turn()` updates state. `build_image_prompt()` builds CLIP-safe (≤77 token) prompts with the `ghibli style` trigger phrase.

**`model.py`** — Text generation. `generate()` and `generate_stream()` (accumulated-text generator). Stream fallback chain: Modal `generate_stream.remote_gen` → Modal `generate.remote` → local llama-cpp with `stream=True`. The local path appends `/no_think` to the system prompt (Qwen3 soft switch) so thinking doesn't eat the token budget.

**`image_model.py`** — Ghibli-Diffusion via Modal T4; IP-Adapter keeps the hero consistent using the first beat's image as reference. Local CPU fallback (requires `torch`+`diffusers`, not in requirements).

**`tts.py`** — Kokoro-82M on Modal CPU (primary); edge-tts cloud fallback for local dev. **`stt.py`** — Whisper small on Modal; transformers whisper-tiny fallback (requires `transformers`, not in requirements). `match_option()` fuzzy-matches the transcript to a choice. **`ambient.py`** — procedural numpy audio per theme. **`pdf_export.py`** — reportlab A5 picture book.

**`app.py`** — Gradio UI. Handlers are generators that yield progressive UI updates: loading → streamed partial beats (blinking cursor) → full text + options (+ image shimmer) → image/TTS → (ending) PDF. State is a plain dict in `gr.State`; private `_` keys (`_options`, `_current_beat`, `_beat_images`, `_reference_image`, `_muted`) are stripped before `StoryState(**d)` reconstruction in `_state_from()`. Generation errors yield a friendly `.status-error` banner and restore the previous screen so the user can retry.

**`styles.css`** — Loaded at startup via `gr.Blocks(css=)` AND re-injected via `gr.HTML(<style>)` at the end of the layout. The second injection is needed because Gradio's `StreamingBar` asset (`.generating.svelte-1uj8rng`) loads after the `css=` injection and overrides it. The end-of-file `/* Last-word rules */` block must stay last in the file to win the cascade.

## Key constraints

- `gr.HTML` sanitizes `<script>` tags — no JS in `<script>` blocks (inline event attributes like `onclick`/`onkeydown` do work). All animations are CSS-only.
- Theme cards are HTML divs; clicking one writes to a hidden `gr.Textbox` (`#theme-bus`) via JS native setter + `input` event dispatch. The JS clears the value first, then sets it, so re-picking the same theme (retry after error) still fires `.change`.
- Gradio renders `visible=False` buttons as disabled-but-present in the DOM during streaming. Hide them with `button.option-btn:disabled { display: none !important }` — this rule must stay in the Last-word block at the end of `styles.css`.
- The output tuple is 19 elements, index-stable: `[story_state, setup_col, story_col, ending_col, beat_display, status_html, progress_html, opt0..opt5, full_story_md, image_placeholder, beat_image, beat_audio, ambient_audio, pdf_file]`. All handlers must yield exactly this shape (canonical list in the `app.py` docstring).
- Qwen3 thinking is disabled on Modal (`enable_thinking=False`) and via `/no_think` on the local GGUF path, but `parse_response()` must still strip `<think>...</think>` before JSON extraction as a safety net.
- A global CSS rule kills all focus outlines; keyboard focus styles must use higher-specificity `:focus-visible` rules (see the Keyboard-focus section in `styles.css`).
- Pinned deps: `starlette<1.0` and `jinja2<3.2` — newer versions break Gradio 4.44.0 `TemplateResponse`. The Space runs Python 3.13 + Gradio 4.44.0.
