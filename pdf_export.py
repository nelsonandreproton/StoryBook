"""
pdf_export.py — Build an illustrated PDF storybook with reportlab.
Returns the path to a temporary PDF file.
"""
import io
import os
import tempfile


def build_pdf(
    theme: str,
    hero: str,
    world: str,
    beats: list[str],
    images: list[bytes],
) -> str:
    """
    Lay out beats + images as an A5 picture book.
    Returns path to a temporary PDF file (caller is responsible for cleanup).
    """
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A5
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()

    W, H = A5
    margin = 1.8 * cm
    body_w = W - 2 * margin

    doc = SimpleDocTemplate(
        tmp.name,
        pagesize=A5,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
    )

    gold = colors.HexColor("#d4a843")
    rust = colors.HexColor("#c85a33")
    tan  = colors.HexColor("#ecdbb8")
    ink  = colors.HexColor("#2c1810")

    h1 = ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=20,
                         alignment=TA_CENTER, spaceAfter=4, leading=26, textColor=rust)
    sub = ParagraphStyle("Sub", fontName="Helvetica", fontSize=10,
                          alignment=TA_CENTER, spaceAfter=3, textColor=colors.HexColor("#5a3e2b"))
    lbl = ParagraphStyle("Lbl", fontName="Helvetica-Bold", fontSize=8,
                          alignment=TA_CENTER, textColor=rust, spaceAfter=4)
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=11,
                           alignment=TA_JUSTIFY, leading=17, spaceAfter=6, textColor=ink)
    end_style = ParagraphStyle("End", fontName="Helvetica-Bold", fontSize=18,
                                alignment=TA_CENTER, textColor=colors.HexColor("#4a8c5c"))

    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("&#128214; StoryForge", h1))
    story.append(Paragraph(theme, sub))
    if hero:
        story.append(Paragraph(f"Hero: {hero}", sub))
    if world:
        story.append(Paragraph(f"World: {world}", sub))
    if images:
        story.append(Spacer(1, 0.4 * cm))
        _img(story, images[0], body_w, 8 * cm)
    story.append(Spacer(1, 0.8 * cm))
    story.append(HRFlowable(width="80%", color=gold, thickness=1.5))
    story.append(Spacer(1, 0.8 * cm))

    # ── Beats ─────────────────────────────────────────────────────────────────
    for i, beat_text in enumerate(beats):
        story.append(Paragraph(f"— Moment {i + 1} of {len(beats)} —", lbl))
        img_bytes = images[i] if i < len(images) else None
        if img_bytes:
            _img(story, img_bytes, body_w, 6 * cm)
            story.append(Spacer(1, 0.25 * cm))
        story.append(Paragraph(beat_text, body))
        story.append(Spacer(1, 0.5 * cm))
        story.append(HRFlowable(width="50%", color=tan, thickness=1))
        story.append(Spacer(1, 0.5 * cm))

    # ── The End ───────────────────────────────────────────────────────────────
    story.append(Paragraph("&#10024; The End &#10024;", end_style))

    doc.build(story)
    return tmp.name


def _img(story, img_bytes: bytes, max_w: float, max_h: float):
    from PIL import Image as PILImage
    from reportlab.platypus import Image as RLImage

    pil = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    w_px, h_px = pil.size
    aspect = h_px / w_px
    w = max_w
    h = w * aspect
    if h > max_h:
        h = max_h
        w = h / aspect
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    story.append(RLImage(buf, width=w, height=h))
