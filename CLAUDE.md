# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app
python app.py

# Smoke test (3-turn validation, downloads model on first run ~1 GB)
python smoke_test.py

# Install deps
pip install -r requirements.txt
```

Model loads on first call and is cached via `@lru_cache`. Restart the process to reload the model.

To use a different model:
```bash
MODEL_REPO=Qwen/Qwen3-4B-GGUF MODEL_FILE=Qwen3-4B-Q4_K_M.gguf python app.py
```

## Architecture

Four files. No framework beyond Gradio.

**`engine.py`** — Pure story logic, no I/O. `StoryState` dataclass holds all canonical facts (hero, world, facts list, history). `build_prompt()` re-injects the full state every turn so the model never needs memory. `parse_response()` strips Qwen3 `<think>` blocks, extracts JSON, and falls back gracefully. `apply_turn()` updates state from the parsed response.

**`model.py`** — Thin wrapper around `llama-cpp-python`. Single `generate(system, user)` function. Model path resolved via `hf_hub_download` and cached with `@lru_cache(maxsize=1)`. Configured via env vars: `MODEL_REPO`, `MODEL_FILE`, `N_CTX`, `N_THREADS`.

**`app.py`** — Gradio UI. All handlers yield a fixed 14-element tuple (index-stable output list). Private `_` keys in the state dict (`_options`, `_current_beat`) are stripped before `StoryState(**d)` reconstruction in `_state_from()`. Theme cards are plain HTML divs; clicking one writes to a hidden `gr.Textbox` (`#theme-bus`) via JS native setter + `input` event dispatch, which triggers the `.change` handler.

**`styles.css`** — Loaded at startup, passed to `gr.Blocks(css=)` AND re-injected via `gr.HTML(<style>)` at the end of the layout. The second injection is needed because Gradio's `StreamingBar` asset (`.generating.svelte-1uj8rng`) loads after the `css=` injection and overrides it. The end-of-file `/* Last-word rules */` block must stay last in the file to win the cascade.

## Key constraints

- `gr.HTML` sanitizes `<script>` tags — no JS in HTML components. All animations are CSS-only.
- Gradio renders `visible=False` buttons as disabled-but-present in the DOM during streaming. Hide them with `button.option-btn:disabled { display: none !important }` — this rule must appear after the general `.option-btn button` rule in the CSS.
- The 14-element output tuple order is: `[story_state, setup_col, story_col, ending_col, beat_display, status_html, progress_html, opt0..opt5, full_story_md]`. All handlers must yield exactly this shape.
- Qwen3 thinking mode is always on. `parse_response()` must strip `<think>...</think>` before JSON extraction.
