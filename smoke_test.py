"""
Smoke test — 3 full turns (start + 2 auto-choices), prints each beat + options.
Judge: coherent? consistent hero/world? clean JSON (no salvage)?
Uses the same engine + model modules as the app (Modal when configured,
local GGUF otherwise — first local run downloads ~1.8 GB).
"""

from engine import StoryState, SYSTEM, build_prompt, parse_response, apply_turn
import model

THEME = "🦊 A brave little fox"
SALVAGE_HIT = False


def run_turn(s: StoryState) -> tuple[StoryState, dict]:
    global SALVAGE_HIT
    prompt = build_prompt(s)
    raw = model.generate(SYSTEM, prompt, max_tokens=512)
    print(f"\n--- RAW (moment {s.moment}) ---\n{raw}\n")
    data = parse_response(raw)
    if not data["options"] and not (s.moment + 1 >= s.total_moments):
        SALVAGE_HIT = True
        print("⚠️  SALVAGE PATH HIT — options were empty on non-final turn")
    s = apply_turn(s, data)
    return s, data


def main():
    print("=== StoryForge Smoke Test ===")
    print(f"Model: {model.MODEL_REPO} / {model.MODEL_FILE}\n")

    s = StoryState(theme=THEME, total_moments=5, num_options=3)

    # Turn 0: opening
    s, data = run_turn(s)
    beat = data["beat"]
    options = data["options"]
    print(f"BEAT 0: {beat}")
    print(f"OPTIONS: {options}")
    print(f"HERO: {s.hero}  |  WORLD: {s.world}")

    for turn in range(1, 3):
        chosen = options[0] if options else "Continue"
        s.history.append({"beat": beat, "choice": chosen})
        s.moment += 1
        s, data = run_turn(s)
        beat = data["beat"]
        options = data["options"]
        print(f"\nBEAT {turn}: {beat}")
        print(f"OPTIONS: {options}")
        print(f"HERO: {s.hero}  |  WORLD: {s.world}  |  FACTS: {s.facts}")

    print("\n=== RESULT ===")
    print(f"Salvage path hit: {SALVAGE_HIT}")
    print("Check above: coherent beats? consistent hero/world? valid JSON?")


if __name__ == "__main__":
    main()
