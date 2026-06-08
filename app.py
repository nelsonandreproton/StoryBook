"""
StoryForge — Gradio app.

Output tuple (18 elements, index-stable across all handlers):
  0  story_state
  1  setup_col
  2  story_col
  3  ending_col
  4  beat_display
  5  status_html       (kept for index stability)
  6  progress_html
  7-12  option_btns[0..5]
  13 full_story_md
  14 beat_image
  15 beat_audio        (TTS narration, edge-tts MP3)
  16 ambient_audio     (background music, numpy WAV)
  17 pdf_file          (downloadable PDF at story end)
"""

import html as _html
import io
import threading

# Patch gradio_client bool-schema bug (gradio-client 1.3.0 + pydantic schemas)
# _json_schema_to_python_type recurses into additionalProperties which can be bool
try:
    import gradio_client.utils as _gcu
    _orig_parse = _gcu._json_schema_to_python_type
    def _safe_parse(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], dict):
            schema = {k: v for k, v in schema.items() if k != "additionalProperties"}
        return _orig_parse(schema, defs)
    _gcu._json_schema_to_python_type = _safe_parse
except Exception:
    pass

import gradio as gr

import ambient
import pdf_export
import stt
import tts
from engine_v2 import (
    StoryState,
    SYSTEM,
    apply_turn,
    build_image_prompt,
    build_prompt,
    parse_response,
)
import image_model_v2 as img_model
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
    esc = _html.escape(text)
    return f'<div class="beat-card beat-visible"><div class="beat-text">{esc}</div></div>'


def _loading_html() -> str:
    return (
        '<div class="loading-wrap">'
        '<span class="loading-label">&#9999;&#65039; Writing your story</span>'
        '<span class="dots"><span>.</span><span>.</span><span>.</span></span>'
        "</div>"
    )


def _progress_html(moment: int, total: int) -> str:
    dots = "".join(
        f'<span class="dot dot-{"done" if i < moment else "current" if i == moment else "future"}"></span>'
        for i in range(1, total + 1)
    )
    return (
        f'<div class="progress-wrap">{dots}'
        f'<span class="progress-label">Moment {moment} of {total}</span></div>'
    )


def _end_html(beat: str) -> str:
    esc = _html.escape(beat)
    return (
        '<div class="end-burst">&#10024;</div>'
        f'<div class="beat-card beat-visible"><div class="beat-text">{esc}</div></div>'
    )


# ── Screen helpers — every handler yields exactly these 18 values ─────────────

def _opt(options: list, disabled: bool = False) -> list:
    out = []
    for i in range(MAX_OPTIONS):
        if i < len(options):
            out.append(gr.update(visible=True, value=options[i], interactive=not disabled))
        else:
            out.append(gr.update(visible=False, value="", interactive=False))
    return out


def _setup_screen(state):
    return (
        state,
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "", "", "",
        *_opt([]),
        "",
        gr.update(value=None, visible=False),   # beat_image
        gr.update(value=None, visible=False),   # beat_audio
        gr.update(value=None, visible=False),   # ambient_audio
        gr.update(visible=False),               # pdf_file
    )


def _story_screen(
    state, beat, options, moment, total,
    *,
    loading=False,
    image=None,
    audio=None,
    ambient_val=None,
):
    return (
        state,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        _loading_html() if loading else _beat_html(beat),
        "",
        _progress_html(moment, total),
        *_opt([] if loading else options),
        "",
        gr.update() if image is None else gr.update(value=image, visible=True),
        gr.update() if audio is None else gr.update(value=audio, autoplay=True, visible=True),
        gr.update() if ambient_val is None else gr.update(value=ambient_val, autoplay=True, visible=True),
        gr.update(visible=False),
    )


def _ending_screen(state, beat, full_story, total, *, image=None, audio=None, pdf=None):
    return (
        state,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        _end_html(beat),
        "",
        _progress_html(total, total),
        *_opt([]),
        full_story,
        gr.update() if image is None else gr.update(value=image, visible=True),
        gr.update() if audio is None else gr.update(value=audio, autoplay=True, visible=True),
        gr.update(),                            # ambient_audio — keep playing
        gr.update() if pdf is None else gr.update(value=pdf, visible=True),
    )


# ── State helper ──────────────────────────────────────────────────────────────

def _state_from(d: dict) -> StoryState:
    return StoryState.from_dict({k: v for k, v in d.items() if not k.startswith("_")})


# ── Core generation ───────────────────────────────────────────────────────────

def _generate_beat(s: StoryState) -> tuple:
    prompt = build_prompt(s)
    is_final = (s.moment + 1) >= s.total_moments
    raw = story_model.generate(SYSTEM, prompt, max_tokens=1024 if is_final else 512)
    data = parse_response(raw)
    if not is_final and not data.get("options"):
        raw2 = story_model.generate(SYSTEM, prompt, max_tokens=512)
        data2 = parse_response(raw2)
        data = data2 if data2.get("options") else {**data, "options": ["Continue the adventure"]}
    s = apply_turn(s, data)
    return s, data["beat"], data.get("options", [])


def _generate_image_bytes(beat: str, hero: str, world: str, ref: bytes = None) -> bytes | None:
    try:
        prompt = build_image_prompt(beat, hero, world)
        return img_model.generate_image(prompt, ref)
    except Exception:
        return None


def _pil_from_bytes(b: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(b))


def _parallel_image_and_audio(beat, hero, world, ref_bytes=None):
    """Run image gen + TTS in parallel threads; returns (img_bytes, audio_path)."""
    img_result: list = [None]
    aud_result: list = [None]

    def _img():
        img_result[0] = _generate_image_bytes(beat, hero, world, ref_bytes)

    def _aud():
        aud_result[0] = tts.generate_speech(beat)

    t1 = threading.Thread(target=_img, daemon=True)
    t2 = threading.Thread(target=_aud, daemon=True)
    t1.start(); t2.start()
    t1.join(); t2.join()
    return img_result[0], aud_result[0]


# ── Event handlers ────────────────────────────────────────────────────────────

def start_story(theme, total_moments, num_options, custom_hero, state):
    total, num = int(total_moments), int(num_options)
    s = StoryState(theme=theme, total_moments=total, num_options=num, moment=0)
    if custom_hero and custom_hero.strip():
        s.hero = custom_hero.strip()

    amb = ambient.generate_ambient(theme)

    # ① Loading screen with ambient music
    yield _story_screen(state, "", [], 1, total, loading=True, ambient_val=amb)

    s, beat, options = _generate_beat(s)
    sd = s.to_dict()
    sd["_options"] = options
    sd["_current_beat"] = beat
    sd["_beat_images"] = []

    # ② Text + choices
    yield _story_screen(sd, beat, options, 1, total)

    # ③ Image + TTS in parallel
    img_bytes, audio_path = _parallel_image_and_audio(beat, s.hero, s.world)
    if img_bytes:
        sd["_reference_image"] = img_bytes
        sd["_beat_images"] = [img_bytes]
        yield _story_screen(
            sd, beat, options, 1, total,
            image=_pil_from_bytes(img_bytes),
            audio=audio_path,
        )
    elif audio_path:
        yield _story_screen(sd, beat, options, 1, total, audio=audio_path)


def on_theme_selected(theme_val, total_moments, num_options, custom_hero, state):
    if not theme_val:
        return
    yield from start_story(theme_val, total_moments, num_options, custom_hero, state)


def choose_option(choice_idx: int, state: dict):
    s = _state_from(state)
    options = state.get("_options", [])
    current_beat = state.get("_current_beat", "")
    chosen = options[choice_idx] if choice_idx < len(options) else "Continue"
    s.history = list(s.history) + [{"beat": current_beat, "choice": chosen}]
    s.moment += 1

    # Keep accumulated images across beats
    beat_images = list(state.get("_beat_images", []))
    ref_bytes   = state.get("_reference_image")

    # ① Loading (keep previous image)
    yield _story_screen(state, current_beat, options, s.moment, s.total_moments, loading=True)

    s, beat, new_options = _generate_beat(s)
    is_final = s.moment >= s.total_moments - 1
    sd = s.to_dict()
    sd["_options"] = new_options
    sd["_current_beat"] = beat
    sd["_reference_image"] = ref_bytes
    sd["_beat_images"] = beat_images

    if is_final or not new_options:
        s.finished = True
        full_beats = [h["beat"] for h in s.history] + [beat]
        full_story = "\n\n".join(f"**Moment {i+1}:** {b}" for i, b in enumerate(full_beats))

        # ② Ending text
        yield _ending_screen(sd, beat, full_story, s.total_moments)

        # ③ Image + TTS
        img_bytes, audio_path = _parallel_image_and_audio(beat, s.hero, s.world, ref_bytes)
        if img_bytes:
            beat_images.append(img_bytes)
            sd["_beat_images"] = beat_images
        yield _ending_screen(
            sd, beat, full_story, s.total_moments,
            image=_pil_from_bytes(img_bytes) if img_bytes else None,
            audio=audio_path,
        )

        # ④ PDF
        try:
            pdf_path = pdf_export.build_pdf(
                theme=s.theme,
                hero=s.hero,
                world=s.world,
                beats=full_beats,
                images=sd["_beat_images"],
            )
            yield _ending_screen(sd, beat, full_story, s.total_moments, pdf=pdf_path)
        except Exception:
            pass

    else:
        # ② Text + choices
        yield _story_screen(sd, beat, new_options, s.moment + 1, s.total_moments)

        # ③ Image + TTS
        img_bytes, audio_path = _parallel_image_and_audio(beat, s.hero, s.world, ref_bytes)
        if img_bytes:
            beat_images.append(img_bytes)
            sd["_beat_images"] = beat_images
        yield _story_screen(
            sd, beat, new_options, s.moment + 1, s.total_moments,
            image=_pil_from_bytes(img_bytes) if img_bytes else None,
            audio=audio_path,
        )


def on_voice_input(audio_tuple, state: dict):
    if audio_tuple is None:
        return
    options = state.get("_options", [])
    if not options:
        return
    text = stt.transcribe(audio_tuple)
    if text:
        idx = stt.match_option(text, options)
        if idx is not None:
            yield from choose_option(idx, state)


def reset_story():
    return _setup_screen({})


# ── UI ────────────────────────────────────────────────────────────────────────

with open("styles_v2.css", encoding="utf-8") as _f:
    _CSS = _f.read()

with gr.Blocks(title="StoryForge", css=_CSS) as demo:

    story_state = gr.State({})

    gr.HTML('<h1 id="app-title">&#128214; StoryForge</h1>')
    gr.HTML('<p id="app-tagline">A magical branching adventure — just for you</p>')

    # ── Setup ──────────────────────────────────────────────────────────────
    with gr.Column(elem_id="setup-section") as setup_col:
        with gr.Row(elem_id="sliders-row"):
            moments_slider = gr.Slider(3, 15, value=10, step=1,
                                       label="How many moments?",
                                       info="More moments = longer story")
            options_slider = gr.Slider(2, 6, value=5, step=1,
                                       label="How many choices each turn?")

        hero_input = gr.Textbox(
            value="",
            placeholder="e.g. Luna, a small girl with silver hair and a red cape",
            label="Your hero (optional — leave blank to let the story invent one)",
            elem_id="hero-input",
            max_lines=1,
        )

        gr.HTML('<p id="themes-label">Choose your adventure:</p>')

        theme_bus = gr.Textbox(value="", visible=True, elem_id="theme-bus", label="")

        cards_html = '<div class="theme-grid">'
        for emoji, title, subtitle in THEMES:
            tv = f"{emoji} {title}"
            cards_html += (
                f'<div class="theme-card" onclick="'
                f'(function(){{'
                f'var wrap=document.getElementById(\'theme-bus\');'
                f'var tb=wrap?wrap.querySelector(\'textarea,input[type=text]\'):null;'
                f'if(!tb){{tb=document.querySelector(\'[data-testid=textbox]\');}} '
                f'if(!tb)return;'
                f'var nativeSet=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,\'value\')'
                f'||Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,\'value\');'
                f'nativeSet.set.call(tb,{repr(tv)});'
                f'tb.dispatchEvent(new Event(\'input\',{{bubbles:true}}));'
                f'}})()">'
                f'<span class="tc-emoji">{emoji}</span>'
                f'<span class="tc-title">{title}</span>'
                f'<span class="tc-sub">{subtitle}</span>'
                f'</div>'
            )
        cards_html += "</div>"
        gr.HTML(cards_html)

    # ── Shared content ─────────────────────────────────────────────────────
    progress_html = gr.HTML("", elem_id="progress-text")
    status_html   = gr.HTML("", elem_id="status-text")
    beat_display  = gr.HTML("", elem_id="beat-display")

    beat_image = gr.Image(
        value=None, visible=False, show_label=False,
        type="pil", interactive=False,
        elem_id="beat-image-wrap",
    )

    # TTS narration — auto-plays per beat, hidden player
    beat_audio = gr.Audio(
        value=None, visible=False,
        label="Story narration", autoplay=True,
        elem_id="beat-audio",
    )

    # Ambient background music — set once per story
    ambient_audio = gr.Audio(
        value=None, visible=False,
        label="Ambient music", autoplay=True,
        elem_id="ambient-audio",
    )

    # ── Story section ───────────────────────────────────────────────────────
    with gr.Column(elem_id="story-section", visible=False) as story_col:
        option_btns = [
            gr.Button(f"Option {i+1}", visible=False, elem_classes=["option-btn"])
            for i in range(MAX_OPTIONS)
        ]

        with gr.Row(elem_id="voice-row"):
            mic_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="🎙️ Or speak your choice",
                elem_id="mic-input",
                visible=True,
            )

        reset_btn_story = gr.Button("Start over", elem_classes=["reset-btn"], size="sm")

    # ── Ending section ──────────────────────────────────────────────────────
    with gr.Column(elem_id="ending-section", visible=False) as ending_col:
        gr.HTML('<p id="the-end-text">&#10024; The End &#10024;</p>')
        with gr.Accordion("Read the whole story", open=False):
            full_story_text = gr.Markdown("")
        pdf_file = gr.File(
            value=None, visible=False,
            label="📖 Download your storybook (PDF)",
            elem_id="pdf-download",
        )
        reset_btn_end = gr.Button("Start a new adventure", elem_id="restart-big")

    # ── Output list (18 elements) ───────────────────────────────────────────
    ALL_OUTPUTS = (
        [story_state, setup_col, story_col, ending_col,
         beat_display, status_html, progress_html]
        + option_btns
        + [full_story_text, beat_image, beat_audio, ambient_audio, pdf_file]
    )

    # ── Wire theme bus ──────────────────────────────────────────────────────
    theme_bus.change(
        fn=on_theme_selected,
        inputs=[theme_bus, moments_slider, options_slider, hero_input, story_state],
        outputs=ALL_OUTPUTS,
    )

    # ── Wire option buttons ─────────────────────────────────────────────────
    for i, btn in enumerate(option_btns):
        def _choice(state, _i=i):
            yield from choose_option(_i, state)
        btn.click(fn=_choice, inputs=[story_state], outputs=ALL_OUTPUTS)

    # ── Wire microphone ─────────────────────────────────────────────────────
    mic_input.stop_recording(
        fn=on_voice_input,
        inputs=[mic_input, story_state],
        outputs=ALL_OUTPUTS,
    )

    # ── Wire reset ──────────────────────────────────────────────────────────
    reset_btn_story.click(fn=reset_story, inputs=[], outputs=ALL_OUTPUTS)
    reset_btn_end.click(fn=reset_story, inputs=[], outputs=ALL_OUTPUTS)

    # ── Re-inject styles (beats StreamingBar override) ──────────────────────
    with open("styles_v2.css", encoding="utf-8") as _sf:
        gr.HTML(f"<style>{_sf.read()}</style>")


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0")
