"""
StoryForge — Gradio app.
State-driven single page: setup → story → ending.
All AI runs in-process via model.py (no cloud APIs).

Output list (all handlers yield this shape, index-stable):
  0  story_state       gr.State
  1  setup_col         Column  visible
  2  story_col         Column  visible
  3  ending_col        Column  visible
  4  beat_display      HTML    value
  5  status_html       HTML    value   (unused — kept for index stability)
  6  progress_html     HTML    value
  7..12  option_btns[0..5]
  13 full_story_md     Markdown value
"""

import html as _html
import gradio as gr
from engine import StoryState, SYSTEM, build_prompt, parse_response, apply_turn
import model as story_model

THEMES = [
    ("🦊", "A brave little fox",      "courage & friendship"),
    ("🚀", "A trip to a sleepy moon",  "wonder & exploration"),
    ("🐙", "The friendly sea monster", "kindness & the deep"),
    ("🌳", "The whispering forest",    "nature & mystery"),
    ("🎈", "The runaway balloon",      "adventure & freedom"),
    ("🐉", "The shy dragon",           "belonging & bravery"),
]

MAX_OPTIONS = 6

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _beat_html(text: str) -> str:
    escaped = _html.escape(text)
    return f'<div class="beat-card beat-visible"><div class="beat-text">{escaped}</div></div>'


def _loading_html() -> str:
    return """
<div class="loading-wrap">
  <span class="loading-label">✍️ Writing your story</span>
  <span class="dots"><span>.</span><span>.</span><span>.</span></span>
</div>
"""


def _progress_html(moment: int, total: int) -> str:
    dots = "".join(
        f'<span class="dot dot-{"done" if i < moment else "current" if i == moment else "future"}"></span>'
        for i in range(1, total + 1)
    )
    return f'<div class="progress-wrap">{dots}<span class="progress-label">Moment {moment} of {total}</span></div>'


def _end_html(beat: str) -> str:
    escaped = _html.escape(beat)
    return (f'<div class="end-burst">✨</div>'
            f'<div class="beat-card beat-visible"><div class="beat-text">{escaped}</div></div>')


# ── Screen helpers ────────────────────────────────────────────────────────────

def _opt(options: list, disabled: bool = False) -> list:
    out = []
    for i in range(MAX_OPTIONS):
        if i < len(options):
            out.append(gr.update(visible=True, value=options[i], interactive=not disabled))
        else:
            out.append(gr.update(visible=False, value="", interactive=False))
    return out


def _setup_screen(state, total: int = 10):
    return (state, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
            "", "", "", *_opt([]), "")


def _story_screen(state, beat, options, moment, total, loading=False):
    return (state, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
            _loading_html() if loading else _beat_html(beat),
            "",
            _progress_html(moment, total),
            *_opt([] if loading else options),
            "")


def _ending_screen(state, beat, full_story, total):
    return (state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
            _end_html(beat), "", _progress_html(total, total), *_opt([]), full_story)


# ── State helpers ─────────────────────────────────────────────────────────────

def _state_from(d: dict) -> StoryState:
    return StoryState.from_dict({k: v for k, v in d.items() if not k.startswith("_")})


def _generate_beat(s: StoryState) -> tuple:
    prompt = build_prompt(s)
    is_final = (s.moment + 1) >= s.total_moments
    max_tok = 1024 if is_final else 512
    raw = story_model.generate(SYSTEM, prompt, max_tokens=max_tok)
    data = parse_response(raw)
    if not is_final and not data.get("options"):
        raw2 = story_model.generate(SYSTEM, prompt, max_tokens=512)
        data2 = parse_response(raw2)
        data = data2 if data2.get("options") else {**data, "options": ["Continue the adventure"]}
    s = apply_turn(s, data)
    return s, data["beat"], data.get("options", [])


# ── Event handlers ────────────────────────────────────────────────────────────

def start_story(theme: str, total_moments: int, num_options: int, state: dict):
    total, num = int(total_moments), int(num_options)
    s = StoryState(theme=theme, total_moments=total, num_options=num, moment=0)
    yield _story_screen(state, "", [], 1, total, loading=True)
    s, beat, options = _generate_beat(s)
    sd = s.to_dict()
    sd["_options"] = options
    sd["_current_beat"] = beat
    yield _story_screen(sd, beat, options, 1, total)


def on_theme_selected(theme_val: str, total_moments: int, num_options: int, state: dict):
    """Called when the hidden textbox changes value (a theme card was clicked)."""
    if not theme_val:
        return
    yield from start_story(theme_val, total_moments, num_options, state)


def choose_option(choice_idx: int, state: dict):
    s = _state_from(state)
    options = state.get("_options", [])
    current_beat = state.get("_current_beat", "")
    chosen = options[choice_idx] if choice_idx < len(options) else "Continue"
    s.history = list(s.history) + [{"beat": current_beat, "choice": chosen}]
    s.moment += 1
    yield _story_screen(state, current_beat, options, s.moment, s.total_moments, loading=True)
    s, beat, new_options = _generate_beat(s)
    is_final = s.moment >= s.total_moments - 1
    sd = s.to_dict()
    sd["_options"] = new_options
    sd["_current_beat"] = beat
    if is_final or not new_options:
        s.finished = True
        full_beats = [h["beat"] for h in s.history] + [beat]
        full_story = "\n\n".join(f"**Beat {i+1}:** {b}" for i, b in enumerate(full_beats))
        yield _ending_screen(s.to_dict(), beat, full_story, s.total_moments)
    else:
        yield _story_screen(sd, beat, new_options, s.moment + 1, s.total_moments)


def reset_story():
    return _setup_screen({})


# ── UI ────────────────────────────────────────────────────────────────────────

# NOTE: css= on gr.Blocks() inlines the stylesheet as a <style> tag — confirmed
# working in Gradio 6. The deprecation warning suggests launch(css=) but that
# path does not actually inject the styles.
with open("styles.css", encoding="utf-8") as _f:
    _CSS = _f.read()

with gr.Blocks(css=_CSS, title="StoryForge") as demo:

    story_state = gr.State({})

    gr.HTML('<h1 id="app-title">📖 StoryForge</h1>')
    gr.HTML('<p id="app-tagline">A magical branching adventure — just for you</p>')

    # ── Setup ──────────────────────────────────────────────────────────────
    with gr.Column(elem_id="setup-section") as setup_col:
        with gr.Row(elem_id="sliders-row"):
            moments_slider = gr.Slider(3, 15, value=10, step=1,
                                       label="How many moments?",
                                       info="More moments = longer story")
            options_slider = gr.Slider(2, 6, value=5, step=1,
                                       label="How many choices each turn?")

        gr.HTML('<p id="themes-label">Choose your adventure:</p>')

        # Message bus: visible=True so Gradio renders it, CSS hides it visually
        theme_bus = gr.Textbox(value="", visible=True, elem_id="theme-bus", label="")

        # Theme cards as plain HTML divs — JS writes to the hidden textbox
        # then dispatches an input event so Gradio picks it up
        cards_html = '<div class="theme-grid">'
        for emoji, title, subtitle in THEMES:
            theme_val = f"{emoji} {title}"
            cards_html += (
                f'<div class="theme-card" onclick="'
                f'(function(){{'
                f'var wrap=document.getElementById(\'theme-bus\');'
                f'var tb=wrap?wrap.querySelector(\'textarea,input[type=text]\'):null;'
                f'if(!tb){{tb=document.querySelector(\'[data-testid=textbox]\');}} '
                f'if(!tb)return;'
                f'var nativeSet=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,\'value\')||Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,\'value\');'
                f'nativeSet.set.call(tb,{repr(theme_val)});'
                f'tb.dispatchEvent(new Event(\'input\',{{bubbles:true}}));'
                f'}})()">'
                f'<span class="tc-emoji">{emoji}</span>'
                f'<span class="tc-title">{title}</span>'
                f'<span class="tc-sub">{subtitle}</span>'
                f'</div>'
            )
        cards_html += '</div>'
        gr.HTML(cards_html)

    # ── Shared content area ─────────────────────────────────────────────────
    progress_html = gr.HTML("", elem_id="progress-text")
    status_html   = gr.HTML("", elem_id="status-text")
    beat_display  = gr.HTML("", elem_id="beat-display")

    # ── Story section ───────────────────────────────────────────────────────
    with gr.Column(elem_id="story-section", visible=False) as story_col:
        option_btns = []
        for i in range(MAX_OPTIONS):
            btn = gr.Button(f"Option {i+1}", visible=False, elem_classes=["option-btn"])
            option_btns.append(btn)
        reset_btn_story = gr.Button("↩ Start over", elem_classes=["reset-btn"], size="sm")

    # ── Ending section ──────────────────────────────────────────────────────
    with gr.Column(elem_id="ending-section", visible=False) as ending_col:
        gr.HTML('<p id="the-end-text">✨ The End ✨</p>')
        with gr.Accordion("📜 Read the whole story", open=False):
            full_story_text = gr.Markdown("")
        reset_btn_end = gr.Button("🌟 Start a new adventure", elem_id="restart-big")

    # ── Output list ────────────────────────────────────────────────────────
    ALL_OUTPUTS = (
        [story_state, setup_col, story_col, ending_col,
         beat_display, status_html, progress_html]
        + option_btns
        + [full_story_text]
    )

    # ── Wire theme bus ──────────────────────────────────────────────────────
    theme_bus.change(
        fn=on_theme_selected,
        inputs=[theme_bus, moments_slider, options_slider, story_state],
        outputs=ALL_OUTPUTS,
    )

    # ── Wire option buttons ─────────────────────────────────────────────────
    for i, btn in enumerate(option_btns):
        def _choice(state, _i=i):
            yield from choose_option(_i, state)
        btn.click(fn=_choice, inputs=[story_state], outputs=ALL_OUTPUTS)

    # ── Wire reset buttons ──────────────────────────────────────────────────
    reset_btn_story.click(fn=reset_story, inputs=[], outputs=ALL_OUTPUTS)
    reset_btn_end.click(fn=reset_story, inputs=[], outputs=ALL_OUTPUTS)

    # ── Override styles injected late (beats Gradio's StreamingBar CSS) ────
    with open("styles.css", encoding="utf-8") as _sf:
        gr.HTML(f"<style>{_sf.read()}</style>")


if __name__ == "__main__":
    demo.queue()
    demo.launch()
