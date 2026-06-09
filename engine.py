"""
engine.py — StoryForge engine. Coherence lives HERE, not in the model's memory.
The small model forgets, so we never rely on it: every turn we re-inject the
canonical facts (StoryState) and ask only for the NEXT beat. Explicit state in,
one grounded step out. Also builds CLIP-safe image prompts per beat.
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
    language: str = "English"
    finished: bool = False

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return StoryState(**d) if d else StoryState()


def system_prompt(language: str = "English") -> str:
    return (
        "You are a warm, imaginative storyteller writing a branching picture-book "
        "adventure for a young child (age 4-8). Keep language simple, kind, and "
        "vivid. No violence, no scary or unsafe content. Every beat is 2-4 short "
        f"sentences. ALWAYS write the story, the choices and all descriptions in "
        f"{language} only — except the JSON keys and the 'scene' field, which are "
        "always English. You ALWAYS answer with valid JSON and nothing else."
    )


SYSTEM = system_prompt()


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
        '"scene": "<6-12 word visual description of this beat, in English>", '
        '"options": ["<choice 1>", "..."]}'
    )
    return (
        f"{_state_block(s)}\n"
        f"TASK: {task}\n\n"
        f"Respond with ONLY this JSON shape, no preamble, no markdown:\n{schema}"
    )


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_non_latin(text: str) -> str:
    """Remove CJK and other non-Latin unicode blocks that Qwen3 occasionally emits."""
    return re.sub(r"[⺀-鿿豈-﫿︰-﹏]+", "", text).strip()


def _clean(text: str) -> str:
    return _strip_non_latin(_strip_think(text))


def parse_response(raw: str) -> dict:
    raw = _strip_think(raw.strip())
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    candidate = m.group(0) if m else raw
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        beat_m = re.search(r'"beat"\s*:\s*"((?:[^"\\]|\\.)*)"', candidate)
        beat = beat_m.group(1) if beat_m else "The story pauses for a moment..."
        data = {"beat": beat, "options": []}
    data.setdefault("beat", "")
    data.setdefault("options", [])
    data.setdefault("new_facts", [])
    data.setdefault("hero", "")
    data.setdefault("world", "")
    data.setdefault("scene", "")
    data["scene"] = _clean(str(data["scene"] or ""))
    data["beat"] = _clean(data["beat"])
    data["options"] = [_clean(o) for o in data["options"] if o]
    return data


_PARTIAL_BEAT_RE = re.compile(r'"beat"\s*:\s*"((?:[^"\\]|\\.)*)')


def extract_partial_beat(raw: str) -> str:
    """Best-effort beat text from a partially generated JSON response.

    Used while streaming: the closing quote may not have arrived yet, and a
    <think> block may still be open.
    """
    raw = re.sub(r"<think>.*?(?:</think>|$)", "", raw, flags=re.DOTALL)
    m = _PARTIAL_BEAT_RE.search(raw)
    if not m:
        return ""
    text = m.group(1)
    try:
        text = json.loads(f'"{text}"')
    except json.JSONDecodeError:
        text = text.replace('\\"', '"').replace("\\n", " ")
    return _clean(text)


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


# ── Image prompt builder ──────────────────────────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "of",
    "for", "with", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "he", "she", "they", "their", "this", "that", "then",
    "as", "by", "from", "up", "out", "about", "into", "through", "after",
}

_MAX_BEAT_WORDS = 30


def _trim_beat(beat: str) -> str:
    """Extract the most visually descriptive words from a beat (≤18 words)."""
    words = beat.replace(",", " ").replace(".", " ").replace("!", " ").split()
    content = [w for w in words if w.lower() not in _STOPWORDS]
    return " ".join(content[:_MAX_BEAT_WORDS])


_NEGATIVE_PROMPT = (
    "border, frame, box, square, panel, grid, letterbox, vignette, "
    "text, watermark, signature, logo, blur, dark, ugly, deformed"
)


def build_image_prompt(beat: str, hero: str, world: str, scene: str = "") -> str:
    """
    Build a CLIP-safe image prompt (target ≤77 tokens).
    Always includes the 'ghibli style' trigger phrase required by Ghibli-Diffusion.
    Scene content leads so CLIP weights it highest. The model's English "scene"
    field is preferred over the raw beat — it stays English even when the story
    is written in another language (SD 1.5's CLIP understands English best).
    """
    scene = (scene or "").strip() or _trim_beat(beat)
    parts = [scene]
    if hero:
        parts.append(hero[:30])
    if world:
        parts.append(world[:30])
    parts += ["ghibli style", "soft watercolor", "warm light", "highly detailed", "cinematic"]
    return ", ".join(parts)
