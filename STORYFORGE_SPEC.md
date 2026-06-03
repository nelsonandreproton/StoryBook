# StoryForge — Build Spec for Claude Code

> Hand this whole file to Claude Code. It is the single source of truth for
> generating the project. Build it exactly as specified; where a choice is left
> open it is marked **[DECIDE ON DAY 1]**.

---

## 0. What we are building (one paragraph)

A branching, illustrated-storybook web app for young children (ages 4–8). The
child picks an opening theme from a set of cards, the app generates the opening
beat of a story plus a set of choices, and the child taps a choice to continue.
This repeats for a configurable number of "moments" until the story reaches a
happy ending. All AI runs **locally inside the app process** (no cloud APIs).
English only for the MVP. Text only for the MVP (images are a later layer).

This is a submission to the **Hugging Face "Build Small" Hackathon**
(https://huggingface.co/build-small-hackathon). The relevant rules:
- Model total parameters **≤ 32B**.
- App **must be a Gradio app hosted as a Hugging Face Space**.
- Submission also needs a short demo video + a social post (not your job).
- Target track: **Backyard AI** (built for a real child) + lean into the
  **"Off the Grid"** badge (no cloud APIs, model runs in-process) and the
  **"Off-Brand"** badge (custom UI, not default Gradio look).

---

## 1. Hard constraints (do not violate)

1. **No cloud LLM APIs.** No OpenAI, Anthropic, or HF Inference API calls. The
   model is loaded and run **in-process** via `llama-cpp-python` (GGUF on CPU).
   This is what earns the "Off the Grid" badge and keeps it free.
2. **Runs on free HF Space CPU tier.** No PRO subscription, no ZeroGPU. So the
   model must be small (1.7B–4B class, Q4_K_M quant) and prompts/outputs short.
3. **Gradio SDK Space.** Not Docker, not Streamlit.
4. **English only**, **text only** for the MVP. Build the code so that adding
   PT and images later is easy (see §8), but do NOT build them now.
5. **Coherence comes from explicit state, not the model's memory.** Every turn
   re-injects the canonical story facts into the prompt. Never assume the model
   remembers previous turns.
6. **Child-safe content.** System prompt must forbid violence, fear, and unsafe
   themes. Keep language simple and warm.

---

## 2. Tech stack

- Python 3.12 (pin in Space README frontmatter).
- Gradio (latest stable) — the only UI layer.
- `llama-cpp-python` — in-process inference.
- `huggingface_hub` — to download the GGUF at runtime.
- Standard library for everything else (json, re, dataclasses).
- No database. Story state lives in `gr.State` for the session only.

---

## 3. Repository layout

```
storyforge/
├── app.py            # Gradio UI + event wiring
├── engine.py         # Story state, prompt building, JSON parsing (provided below)
├── model.py          # llama.cpp wrapper (provided below)
├── styles.css        # Custom theme (Off-Brand badge)
├── requirements.txt
└── README.md         # HF Space card with YAML frontmatter
```

---

## 4. The architecture (READ THIS — it is the whole design)

### 4.1 Coherence pattern

The model is small and forgets. We never rely on it to remember. Instead we keep
a `StoryState` object holding the **canonical facts**:
- the chosen theme,
- the hero description and world description (set once, then frozen),
- a list of established facts ("the bridge is broken", "the key is gold"),
- the full history of beats + the choice the child made at each.

On **every** turn we serialize this state into the prompt and ask the model only
for the **next beat** plus the next set of choices. The model returns JSON; we
parse it, fold any new facts back into the state, and render. This is the
RAG/agent pattern applied to narrative — exactly how the coherent-story-gen
papers (STORYTELLER, DOC) do it: structured planning + explicit state, not
"please be coherent".

### 4.2 The turn loop

```
START
  child picks a theme card  -> StoryState(theme=..., moment=0)
  generate(moment 0)        -> opening beat + N options ; set hero/world
LOOP (moment 1 .. total_moments-1)
  child taps an option      -> append {beat, choice} to history ; moment += 1
  generate(moment)          -> next beat + N options, grounded in state
FINAL (moment == total_moments-1)
  generate(final)           -> closing beat, options = []  ; finished = True
  show "The End" + a "Start over" button
```

### 4.3 Configurability

Two settings, exposed in the UI (sliders), default 10 and 5:
- `total_moments` (number of beats / decision points). Range 3–15.
- `num_options` (choices shown each turn). Range 2–6.

### 4.4 ZeroGPU note (for the future, not now)

We are NOT on ZeroGPU for the MVP (CPU only, no PRO). But write the model layer
so that IF we later move to ZeroGPU for images, we know the rules:
- decorate the GPU function with `@spaces.GPU(duration=...)`,
- `gr.State` is **pickled on every yield**, NOT shared by reference — so always
  pass state in and return it out explicitly; never mutate-and-expect-sharing,
- never return CUDA tensors across the boundary,
- decorate the outer per-turn function, not inner per-call helpers.
For the CPU MVP none of this bites, but keep state-passing explicit anyway.

---

## 5. Provided code — use these as-is (adapt only if needed)

### 5.1 `engine.py`

```python
"""
StoryForge engine — coherence lives HERE, not in the model's memory.
The small model forgets, so we never rely on it. Every turn we re-inject the
canonical facts (story_state) and ask only for the NEXT beat. Explicit state in,
one grounded step out. No cloud APIs.
"""

import json
import re
from dataclasses import dataclass, field, asdict


@dataclass
class StoryState:
    theme: str = ""
    hero: str = ""
    world: str = ""
    facts: list = field(default_factory=list)
    history: list = field(default_factory=list)   # [{"beat": str, "choice": str}]
    moment: int = 0
    total_moments: int = 10
    num_options: int = 5
    finished: bool = False

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return StoryState(**d) if d else StoryState()


SYSTEM = (
    "You are a warm, imaginative storyteller writing a branching picture-book "
    "adventure for a young child (age 4-8). Keep language simple, kind, and "
    "vivid. No violence, no scary or unsafe content. Every beat is 2-4 short "
    "sentences. You ALWAYS answer with valid JSON and nothing else."
)


def _state_block(s: StoryState) -> str:
    facts = "; ".join(s.facts) if s.facts else "none yet"
    recap = ""
    for i, h in enumerate(s.history, 1):
        recap += f"\n  Beat {i}: {h['beat']}\n    The child chose: {h['choice']}"
    return (
        f"STORY SO FAR (do not contradict any of this):\n"
        f"- Theme: {s.theme}\n"
        f"- Hero: {s.hero or '(define one)'}\n"
        f"- World: {s.world or '(define one)'}\n"
        f"- Established facts: {facts}\n"
        f"- Beats played so far:{recap or ' none'}\n"
    )


def build_prompt(s: StoryState) -> str:
    last = s.moment + 1
    is_final = last >= s.total_moments
    if s.moment == 0:
        task = (
            f"Begin the story based on the theme '{s.theme}'. Invent the hero and "
            f"the world. Write the opening beat, then offer exactly {s.num_options} "
            f"distinct choices for what happens next."
        )
    elif is_final:
        task = (
            "Write the FINAL beat that brings the adventure to a happy, satisfying "
            "close. This is the ending: provide an empty options list."
        )
    else:
        task = (
            f"Continue from the child's last choice. Write the next beat "
            f"(beat {last} of {s.total_moments}), staying consistent with the story "
            f"so far, then offer exactly {s.num_options} distinct choices."
        )
    schema = (
        '{"beat": "<2-4 sentence story text>", '
        '"hero": "<short hero description, only if newly established else repeat>", '
        '"world": "<short world description, same rule>", '
        '"new_facts": ["<any new canonical fact to remember>"], '
        '"options": ["<choice 1>", "..."]}'
    )
    return (
        f"{_state_block(s)}\n"
        f"TASK: {task}\n\n"
        f"Respond with ONLY this JSON shape, no preamble, no markdown:\n{schema}"
    )


def parse_response(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    candidate = m.group(0) if m else raw
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        data = {"beat": raw[:300] or "The story pauses for a moment...",
                "options": []}
    data.setdefault("beat", "")
    data.setdefault("options", [])
    data.setdefault("new_facts", [])
    data.setdefault("hero", "")
    data.setdefault("world", "")
    return data


def apply_turn(s: StoryState, data: dict) -> StoryState:
    if data.get("hero") and not s.hero:
        s.hero = data["hero"].strip()
    if data.get("world") and not s.world:
        s.world = data["world"].strip()
    for f in data.get("new_facts", []):
        f = (f or "").strip()
        if f and f.lower() != "none" and f not in s.facts:
            s.facts.append(f)
    return s
```

### 5.2 `model.py`

```python
"""
Model layer. Small GGUF model via llama.cpp, running IN-PROCESS on CPU.
No cloud APIs -> "Off the Grid". Swap MODEL_REPO/MODEL_FILE for whatever you
benchmark best on day 1.
"""

import os
from functools import lru_cache

MODEL_REPO = os.getenv("MODEL_REPO", "Qwen/Qwen3-1.7B-GGUF")
MODEL_FILE = os.getenv("MODEL_FILE", "Qwen3-1.7B-Q4_K_M.gguf")
N_CTX = int(os.getenv("N_CTX", "4096"))
N_THREADS = int(os.getenv("N_THREADS", str(os.cpu_count() or 4)))


@lru_cache(maxsize=1)
def _load():
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return Llama(model_path=path, n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)


def generate(system: str, user: str, max_tokens: int = 512) -> str:
    llm = _load()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.8,
        top_p=0.9,
    )
    return out["choices"][0]["message"]["content"]
```

---

## 6. `app.py` — build this (the part Claude Code must write)

Requirements for the Gradio app:

### 6.1 Layout / screens (single page, state-driven visibility)

- **Setup screen** (visible at start):
  - App title + one-line tagline.
  - Two sliders: "How many moments?" (3–15, default 10) and "How many choices?"
    (2–6, default 5).
  - A grid of **6 theme cards** (buttons styled as cards). Suggested themes
    (kid-friendly, English):
    1. 🦊 A brave little fox
    2. 🚀 A trip to a sleepy moon
    3. 🐙 The friendly sea monster
    4. 🌳 The whispering forest
    5. 🎈 The runaway balloon
    6. 🐉 The shy dragon
  - Tapping a card starts the story.
- **Story screen** (visible after start):
  - A "page" area showing the current beat text (large, readable, storybook
    styling).
  - A progress indicator: "Moment X of Y".
  - Below: the option buttons (one per choice), full-width, big tap targets.
  - A small "Start over" button.
- **Ending screen**: the final beat + "✨ The End ✨" + "Start over" button.
  Optionally show a "Read the whole story" expander that concatenates all beats.

### 6.2 State & event wiring

- One `gr.State` holding the story-state **as a dict** (`StoryState.to_dict()`).
- On every handler: reconstruct with `StoryState.from_dict(state)`, do the work,
  return `s.to_dict()` back into the State. Always pass explicitly; never rely on
  in-place mutation being shared.
- **Start handler** (`theme -> ...`):
  - build `StoryState(theme, total_moments, num_options, moment=0)`,
  - `build_prompt` -> `model.generate(SYSTEM, prompt)` -> `parse_response` ->
    `apply_turn`,
  - store the returned `options` so the choice buttons know their labels,
  - render beat + options, hide setup, show story.
- **Choice handler** (`choice_index -> ...`):
  - append `{"beat": current_beat, "choice": chosen_text}` to history,
  - `moment += 1`,
  - if `moment >= total_moments - 1` -> generate final beat, set `finished`,
    show ending screen,
  - else generate next beat + options, re-render.
- **Reset handler**: clear state, show setup screen again.

### 6.3 Dynamic option buttons

`num_options` is configurable (2–6), so create the **maximum (6)** option buttons
up front and toggle visibility per turn with `gr.update(visible=..., value=label)`.
Each button's click passes its fixed index; the handler maps index -> the option
text stored in state. (Do not try to create buttons dynamically at runtime —
pre-create 6 and show/hide.)

### 6.4 Latency UX (important on free CPU)

- Generation on CPU takes seconds. Show a loading state: disable buttons and show
  a "✍️ writing your story..." message while generating (use Gradio's built-in
  pending/queue behavior; set `.queue()` on launch).
- Keep `max_tokens` modest (≈400–512) so turns stay snappy.

### 6.5 Safety net

- If `parse_response` salvages (model returned junk and options is empty) on a
  non-final moment, regenerate once; if still empty, show the beat with a single
  "Continue" option so the story never dead-ends.

---

## 7. Custom styling — `styles.css` (Off-Brand badge)

Do NOT ship the default Gradio look. Commit to a **soft, warm storybook**
aesthetic:
- A distinctive display font for headings/beat text (e.g. a rounded or
  hand-drawn Google Font like "Fredoka", "Baloo 2", or "Quicksand" — pick one,
  load via `@import` in the CSS), and a clean readable body font.
- Warm palette: cream/parchment background, deep ink text, one or two accent
  colors (e.g. warm coral + leafy green). No purple-on-white AI-slop gradients.
- Theme cards: rounded corners, soft shadow, gentle hover lift, big emoji.
- Beat text: large (1.4–1.6rem), generous line height, centered "page" max-width
  ~640px, like a picture-book page.
- Option buttons: large, pill-shaped, full width, clear hover/active states.
- Subtle page-load fade/stagger animation (CSS only).
- Load the CSS via `gr.Blocks(css=...)` or `css_paths`.

---

## 8. Explicitly OUT of scope for the MVP (build hooks, not features)

Do not implement these now, but leave the code structured so they slot in later:
- **Images.** Later: an image per beat. Would need ZeroGPU (PRO) + SDXL; the beat
  text already implies a scene, so a future `illustrate(beat, hero, world)` could
  hang off `apply_turn`. Leave a `# TODO: illustrate` marker.
- **Portuguese.** Later: a language toggle. Keep all user-facing strings and the
  SYSTEM/prompt builder in one place so a `lang` param is a clean addition.
- **Character-consistent illustrations.** Hardest, last. Note only.
- **TTS narration** for pre-readers. Note only.
- **PDF "export your storybook"** at the end. Note only.

---

## 9. `requirements.txt`

```
gradio
llama-cpp-python
huggingface_hub
```
(Do NOT pin `spaces`; it is not needed for CPU and the platform manages it if
later added. `llama-cpp-python` installs a CPU wheel on the Space build.)

---

## 10. `README.md` (HF Space card)

Must start with YAML frontmatter so the Space builds correctly:

```yaml
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
```
Below the frontmatter: a short description, the two tracks/badges it targets,
how to run locally (`pip install -r requirements.txt && python app.py`), and a
line stating all inference is local (Off the Grid).

---

## 11. Day-1 validation gate (do this BEFORE building the full UI)

The single biggest risk is whether a tiny CPU model produces **coherent** kids'
stories with **clean JSON**. Before investing in UI, write a 20-line
`smoke_test.py` that:
1. builds an opening `StoryState` for one theme,
2. runs 3 full turns (start + 2 choices, auto-pick option 0),
3. prints each beat and the parsed options.

Judge: Are the beats coherent and on-theme? Is the hero/world consistent across
turns? Does JSON parse cleanly (no salvage path hit)? 
- If YES -> proceed with `Qwen3-1.7B`. 
- If coherence is weak -> try `Qwen3-4B-Instruct` GGUF (still CPU-OK, slower) or
  `Mistral-Small`-class small GGUF. **[DECIDE ON DAY 1]** which model ships.

Tune `temperature` down (0.6–0.7) if it drifts; up (0.85) if it's flat.

---

## 12. Build order (so there is always something submittable)

1. `engine.py`, `model.py` (provided) + `smoke_test.py` -> pass the §11 gate.
2. Minimal `app.py`: setup screen + start + one turn rendering. Confirm a full
   story plays start-to-end with default 10/5.
3. Configurable sliders + reset + ending screen + safety net.
4. `styles.css` storybook theme (Off-Brand badge).
5. Push to a Space under the `build-small-hackathon` org, confirm it builds on
   free CPU and a full story plays in the browser.
6. (Only if time/PRO) images layer. Out of scope for MVP.

Deliver a working, submittable app at the end of step 4 even if nothing after it
gets done.
