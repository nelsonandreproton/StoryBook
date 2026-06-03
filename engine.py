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
    "sentences. ALWAYS write in English only. "
    "You ALWAYS answer with valid JSON and nothing else."
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


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_non_latin(text: str) -> str:
    """Remove CJK and other non-Latin unicode blocks that Qwen3 occasionally emits."""
    return re.sub(r"[⺀-鿿豈-﫿︰-﹏]+", "", text).strip()


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
        # Salvage beat from truncated JSON
        beat_m = re.search(r'"beat"\s*:\s*"((?:[^"\\]|\\.)*)"', candidate)
        beat = beat_m.group(1) if beat_m else "The story pauses for a moment..."
        data = {"beat": beat, "options": []}
    data.setdefault("beat", "")
    data.setdefault("options", [])
    data.setdefault("new_facts", [])
    data.setdefault("hero", "")
    data.setdefault("world", "")
    data["beat"] = _clean(data["beat"])
    data["options"] = [_clean(o) for o in data["options"] if o]
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
