"""
StoryForge — Gradio app.

Output tuple (19 elements, index-stable across all handlers):
  0  story_state
  1  setup_col
  2  story_col
  3  ending_col
  4  beat_display
  5  status_html       (kept for index stability)
  6  progress_html
  7-12  option_btns[0..5]
  13 full_story_md
  14 image_placeholder (shimmer skeleton while the image generates)
  15 beat_image
  16 beat_audio        (TTS narration, edge-tts MP3)
  17 ambient_audio     (background music, numpy WAV)
  18 pdf_file          (downloadable PDF at story end)
"""

import html as _html
import io
import threading
import traceback

import gradio as gr

# Patch gradio Blocks.get_api_info — gradio-client 1.3.0 crashes on bool
# additionalProperties in pydantic schemas (APIInfoParseError). Non-critical:
# only affects the /info endpoint used by gradio Python client, not the UI.
try:
    from gradio_client.utils import APIInfoParseError as _APIInfoParseError
    import gradio.blocks as _gb
    _orig_get_api_info = _gb.Blocks.get_api_info
    def _safe_get_api_info(self):
        try:
            return _orig_get_api_info(self)
        except (_APIInfoParseError, TypeError, Exception):
            return {"named_endpoints": {}, "unnamed_endpoints": {}}
    _gb.Blocks.get_api_info = _safe_get_api_info
except Exception:
    pass

import ambient
import pdf_export
import stt
import tts
from engine import (
    StoryState,
    SYSTEM,
    apply_turn,
    build_image_prompt,
    build_prompt,
    extract_partial_beat,
    parse_response,
)
import image_model as img_model
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

def _beat_html(text: str, streaming: bool = False) -> str:
    esc = _html.escape(text)
    if streaming:
        # No beat-visible class: the fade-in animation would restart on every
        # streamed chunk. The cursor blinks via CSS instead.
        return (
            '<div class="beat-card"><div class="beat-text">'
            f'{esc}<span class="stream-cursor">&#9612;</span></div></div>'
        )
    return f'<div class="beat-card beat-visible"><div class="beat-text">{esc}</div></div>'


_SHIMMER_HTML = '<div class="image-placeholder"></div>'


def _error_html(msg: str) -> str:
    return f'<div class="status-error">&#10024; {_html.escape(msg)}</div>'


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


# ── Screen helpers — every handler yields exactly these 19 values ─────────────

def _opt(options: list, disabled: bool = False) -> list:
    out = []
    for i in range(MAX_OPTIONS):
        if i < len(options):
            out.append(gr.update(visible=True, value=options[i], interactive=not disabled))
        else:
            out.append(gr.update(visible=False, value="", interactive=False))
    return out


def _setup_screen(state, status=""):
    return (
        state,
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "", status, "",
        *_opt([]),
        "",
        "",                                     # image_placeholder
        gr.update(value=None, visible=False),   # beat_image
        gr.update(value=None, visible=False),   # beat_audio
        gr.update(value=None, visible=False),   # ambient_audio
        gr.update(visible=False),               # pdf_file
    )


def _story_screen(
    state, beat, options, moment, total,
    *,
    loading=False,
    streaming=False,
    image=None,
    audio=None,
    ambient_val=None,
    shimmer=False,
    status="",
):
    return (
        state,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        _loading_html() if loading else _beat_html(beat, streaming=streaming),
        status,
        _progress_html(moment, total),
        *_opt([] if (loading or streaming) else options),
        "",
        _SHIMMER_HTML if shimmer else "",
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
        "",                                     # image_placeholder
        gr.update() if image is None else gr.update(value=image, visible=True),
        gr.update() if audio is None else gr.update(value=audio, autoplay=True, visible=True),
        gr.update(),                            # ambient_audio — keep playing
        gr.update() if pdf is None else gr.update(value=pdf, visible=True),
    )


# ── State helper ──────────────────────────────────────────────────────────────

def _state_from(d: dict) -> StoryState:
    return StoryState.from_dict({k: v for k, v in d.items() if not k.startswith("_")})


# ── Core generation ───────────────────────────────────────────────────────────

_STREAM_STEP_CHARS = 12


def _generate_beat_events(s: StoryState):
    """Yield ("partial", beat_text) while streaming, then ("done", (s, beat, options))."""
    prompt = build_prompt(s)
    is_final = (s.moment + 1) >= s.total_moments
    raw = ""
    shown = 0
    for acc in story_model.generate_stream(
        SYSTEM, prompt, max_tokens=1024 if is_final else 512
    ):
        raw = acc
        partial = extract_partial_beat(acc)
        if partial and len(partial) - shown >= _STREAM_STEP_CHARS:
            shown = len(partial)
            yield "partial", partial
    data = parse_response(raw)
    if not is_final and not data.get("options"):
        raw2 = story_model.generate(SYSTEM, prompt, max_tokens=512)
        data2 = parse_response(raw2)
        data = data2 if data2.get("options") else {**data, "options": ["Continue the adventure"]}
    s = apply_turn(s, data)
    yield "done", (s, data["beat"], data.get("options", []))


def _generate_image_bytes(beat: str, hero: str, world: str, ref: bytes = None) -> bytes | None:
    try:
        prompt = build_image_prompt(beat, hero, world)
        return img_model.generate_image(prompt, ref)
    except Exception:
        print("[app] image generation failed:")
        traceback.print_exc()
        return None


def _pil_from_bytes(b: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(b))


def _parallel_image_and_audio(beat, hero, world, ref_bytes=None, muted=False):
    """Run image gen + TTS in parallel threads; returns (img_bytes, audio_path).

    When muted, TTS is skipped entirely (saves a network round-trip too).
    """
    img_result: list = [None]
    aud_result: list = [None]

    def _img():
        img_result[0] = _generate_image_bytes(beat, hero, world, ref_bytes)

    def _aud():
        aud_result[0] = tts.generate_speech(beat)

    threads = [threading.Thread(target=_img, daemon=True)]
    if not muted:
        threads.append(threading.Thread(target=_aud, daemon=True))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return img_result[0], aud_result[0]


# ── Event handlers ────────────────────────────────────────────────────────────

def start_story(theme, total_moments, num_options, custom_hero, state):
    muted = bool(isinstance(state, dict) and state.get("_muted"))
    total, num = int(total_moments), int(num_options)
    s = StoryState(theme=theme, total_moments=total, num_options=num, moment=0)
    if custom_hero and custom_hero.strip():
        s.hero = custom_hero.strip()

    amb = None if muted else ambient.generate_ambient(theme)

    # ① Loading screen with ambient music
    yield _story_screen(state, "", [], 1, total, loading=True, ambient_val=amb)

    # ② Stream the beat text as the model writes it
    beat, options = "", []
    try:
        for kind, payload in _generate_beat_events(s):
            if kind == "partial":
                yield _story_screen(state, payload, [], 1, total, streaming=True)
            else:
                s, beat, options = payload
    except Exception:
        traceback.print_exc()
        yield _setup_screen(
            state if isinstance(state, dict) else {},
            status=_error_html(
                "The storyteller lost the thread of the tale — pick a theme to try again!"
            ),
        )
        return

    sd = s.to_dict()
    sd["_muted"] = muted
    sd["_options"] = options
    sd["_current_beat"] = beat
    sd["_beat_images"] = []

    # ③ Full text + choices, shimmer where the illustration will appear
    yield _story_screen(sd, beat, options, 1, total, shimmer=True)

    # ④ Image + TTS in parallel
    img_bytes, audio_path = _parallel_image_and_audio(beat, s.hero, s.world, muted=muted)
    if img_bytes:
        sd["_reference_image"] = img_bytes
        sd["_beat_images"] = [img_bytes]
    yield _story_screen(
        sd, beat, options, 1, total,
        image=_pil_from_bytes(img_bytes) if img_bytes else None,
        audio=audio_path,
    )


def on_theme_selected(theme_val, total_moments, num_options, custom_hero, state):
    if not theme_val:
        return
    yield from start_story(theme_val, total_moments, num_options, custom_hero, state)


def choose_option(choice_idx: int, state: dict):
    s = _state_from(state)
    muted = bool(state.get("_muted"))
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

    # ② Stream the next beat
    beat, new_options = "", []
    try:
        for kind, payload in _generate_beat_events(s):
            if kind == "partial":
                yield _story_screen(state, payload, [], s.moment + 1, s.total_moments, streaming=True)
            else:
                s, beat, new_options = payload
    except Exception:
        traceback.print_exc()
        # Restore the pre-click view (stored state was never advanced) so the
        # same option can simply be clicked again.
        yield _story_screen(
            state, current_beat, options, s.moment, s.total_moments,
            status=_error_html("The magic fizzled for a moment — try that choice again!"),
        )
        return

    is_final = s.moment >= s.total_moments - 1
    sd = s.to_dict()
    sd["_muted"] = muted
    sd["_options"] = new_options
    sd["_current_beat"] = beat
    sd["_reference_image"] = ref_bytes
    sd["_beat_images"] = beat_images

    if is_final or not new_options:
        s.finished = True
        sd["finished"] = True
        full_beats = [h["beat"] for h in s.history] + [beat]
        full_story = "\n\n".join(f"**Moment {i+1}:** {b}" for i, b in enumerate(full_beats))

        # ③ Ending text
        yield _ending_screen(sd, beat, full_story, s.total_moments)

        # ④ Image + TTS
        img_bytes, audio_path = _parallel_image_and_audio(beat, s.hero, s.world, ref_bytes, muted=muted)
        if img_bytes:
            beat_images.append(img_bytes)
            sd["_beat_images"] = beat_images
        yield _ending_screen(
            sd, beat, full_story, s.total_moments,
            image=_pil_from_bytes(img_bytes) if img_bytes else None,
            audio=audio_path,
        )

        # ⑤ PDF
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
            print("[app] PDF export failed:")
            traceback.print_exc()

    else:
        # ③ Text + choices — shimmer only while there is no illustration yet
        yield _story_screen(sd, beat, new_options, s.moment + 1, s.total_moments,
                            shimmer=not beat_images)

        # ④ Image + TTS
        img_bytes, audio_path = _parallel_image_and_audio(beat, s.hero, s.world, ref_bytes, muted=muted)
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


def reset_story(state):
    keep = {}
    if isinstance(state, dict) and state.get("_muted"):
        keep["_muted"] = True
    return _setup_screen(keep)


def toggle_sound(enabled, state):
    """Mute/unmute both audio players; restart ambient music when re-enabled mid-story."""
    state = dict(state) if isinstance(state, dict) else {}
    state["_muted"] = not enabled
    if not enabled:
        return (
            state,
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )
    s = _state_from(state)
    if s.theme and not s.finished:
        amb = ambient.generate_ambient(s.theme)
        return state, gr.update(), gr.update(value=amb, autoplay=True, visible=True)
    return state, gr.update(), gr.update()


# ── UI ────────────────────────────────────────────────────────────────────────

with open("styles.css", encoding="utf-8") as _f:
    _CSS = _f.read()

with gr.Blocks(title="StoryForge", css=_CSS) as demo:

    story_state = gr.State({})

    gr.HTML('<h1 id="app-title">&#128214; StoryForge</h1>')
    gr.HTML('<p id="app-tagline">A magical branching adventure — just for you</p>')

    # ── Setup ──────────────────────────────────────────────────────────────
    with gr.Column(elem_id="setup-section") as setup_col:
        with gr.Row(elem_id="sliders-row"):
            moments_slider = gr.Slider(3, 15, value=5, step=1,
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

        cards_html = '<div class="theme-grid" role="group" aria-label="Choose your adventure">'
        for emoji, title, subtitle in THEMES:
            tv = f"{emoji} {title}"
            cards_html += (
                f'<div class="theme-card" role="button" tabindex="0" '
                f'aria-label="{title} — {subtitle}" '
                f'onkeydown="if(event.key===\'Enter\'||event.key===\' \')'
                f'{{event.preventDefault();this.click();}}" '
                f'onclick="'
                f'(function(){{'
                f'var wrap=document.getElementById(\'theme-bus\');'
                f'var tb=wrap?wrap.querySelector(\'textarea,input[type=text]\'):null;'
                f'if(!tb){{tb=document.querySelector(\'[data-testid=textbox]\');}} '
                f'if(!tb)return;'
                f'var nativeSet=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,\'value\')'
                f'||Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,\'value\');'
                # Clear first so re-picking the same theme (e.g. retry after an
                # error) still produces a change event.
                f'nativeSet.set.call(tb,\'\');'
                f'tb.dispatchEvent(new Event(\'input\',{{bubbles:true}}));'
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
    with gr.Row(elem_id="sound-row"):
        sound_toggle = gr.Checkbox(
            value=True, label="🔊 Sound", elem_id="sound-toggle", container=False,
        )

    progress_html = gr.HTML("", elem_id="progress-text")
    status_html   = gr.HTML("", elem_id="status-text")
    beat_display  = gr.HTML("", elem_id="beat-display")

    image_placeholder = gr.HTML("", elem_id="image-placeholder-slot")

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

    # ── Output list (19 elements) ───────────────────────────────────────────
    ALL_OUTPUTS = (
        [story_state, setup_col, story_col, ending_col,
         beat_display, status_html, progress_html]
        + option_btns
        + [full_story_text, image_placeholder, beat_image,
           beat_audio, ambient_audio, pdf_file]
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

    # ── Wire sound toggle ───────────────────────────────────────────────────
    sound_toggle.change(
        fn=toggle_sound,
        inputs=[sound_toggle, story_state],
        outputs=[story_state, beat_audio, ambient_audio],
    )

    # ── Wire reset ──────────────────────────────────────────────────────────
    reset_btn_story.click(fn=reset_story, inputs=[story_state], outputs=ALL_OUTPUTS)
    reset_btn_end.click(fn=reset_story, inputs=[story_state], outputs=ALL_OUTPUTS)

    # ── Re-inject styles (beats StreamingBar override) ──────────────────────
    with open("styles.css", encoding="utf-8") as _sf:
        gr.HTML(f"<style>{_sf.read()}</style>")


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0")
