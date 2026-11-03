"""Static check of the deck: bounds, estimated text overflow, notes, chart data."""

from __future__ import annotations

import os

from PIL import ImageFont
from pptx import Presentation
from pptx.util import Emu

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "TMG301_What_Moves_the_Number.pptx")
FONT_PATH = r"C:\Windows\Fonts\segoeui.ttf"
FONT_BOLD = r"C:\Windows\Fonts\seguisb.ttf"

EMU_PER_INCH = 914400
SLIDE_W = 13.333
SLIDE_H = 7.5

presentation = Presentation(DECK)
print(f"slides: {len(presentation.slides)}")

problems: list[str] = []


def font_for(size_pt: float, bold: bool):
    path = FONT_BOLD if bold else FONT_PATH
    if not os.path.exists(path):
        path = FONT_PATH
    return ImageFont.truetype(path, int(size_pt * 96 / 72))


def wrap_lines(text: str, width_in: float, size_pt: float, bold: bool) -> int:
    font = font_for(size_pt, bold)
    max_px = width_in * 96
    lines = 0
    for block in text.split("\n"):
        words = block.split()
        if not words:
            lines += 1
            continue
        current = words[0]
        count = 1
        for word in words[1:]:
            trial = current + " " + word
            if font.getlength(trial) <= max_px:
                current = trial
            else:
                count += 1
                current = word
        lines += count
    return lines


for index, slide in enumerate(presentation.slides, 1):
    has_notes = slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip()
    if not has_notes:
        problems.append(f"slide {index}: no speaker notes")

    for shape in slide.shapes:
        left = shape.left / EMU_PER_INCH
        top = shape.top / EMU_PER_INCH
        width = shape.width / EMU_PER_INCH
        height = shape.height / EMU_PER_INCH
        if left < -0.01 or top < -0.01 or left + width > SLIDE_W + 0.01 or top + height > SLIDE_H + 0.01:
            problems.append(f"slide {index}: shape out of bounds ({left:.2f},{top:.2f},{width:.2f}x{height:.2f})")

        if not shape.has_text_frame:
            continue
        frame = shape.text_frame
        text = frame.text.strip()
        if not text:
            continue
        # largest run size in the frame
        sizes = [run.font.size.pt for para in frame.paragraphs for run in para.runs if run.font.size]
        bold = any(run.font.bold for para in frame.paragraphs for run in para.runs)
        size = max(sizes) if sizes else 18
        inner_w = width - 0.15
        inner_h = height - 0.05
        lines = wrap_lines(text, inner_w, size, bold)
        needed = lines * size * 1.22 / 72
        if needed > inner_h + 0.12:
            problems.append(
                f"slide {index}: text needs {needed:.2f}in but box is {inner_h:.2f}in "
                f"({text[:45]!r})"
            )

print("problems:", len(problems))
for item in problems:
    print(" -", item)
