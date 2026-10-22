"""
render.py - draws one still image per storyboard scene (1280x720 by default).

Scene types (the model picks one per scene in the storyboard):
  title    lesson title and goal
  slide    an original slide from the recording, shown large, with where it comes from
  bullets  a heading and 2-5 short key points
  steps    a process: numbered boxes connected by arrows (a simple diagram)
The voice explains; the screen only shows the key points, so the video is not a slide read
aloud. Text sizes adapt so everything fits. Fonts: Segoe UI on Windows, DejaVu on Linux.

Try it on an existing storyboard:  uv run render.py data/output/<folder>/storyboard.json
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config

BG = (248, 249, 251)
INK = (31, 41, 51)
MUTED = (100, 116, 139)
ACCENT = (13, 110, 123)
ACCENT_LIGHT = (225, 241, 241)
BOX = (255, 255, 255)
BORDER = (203, 213, 225)

FONTS = {
    False: ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"],
    True: ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
           "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
}


@lru_cache(maxsize=256)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """TrueType font of this size (Segoe UI on Windows, DejaVu on Linux); cached."""
    for path in FONTS[bold]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, max_w: float) -> list[str]:
    """Greedy word wrap; a single word wider than the line is split."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        while draw.textlength(word, font=fnt) > max_w and len(word) > 1:   # very long word
            cut = len(word)
            while cut > 1 and draw.textlength(word[:cut] + "-", font=fnt) > max_w:
                cut -= 1
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:cut] + "-")
            word = word[cut:]
        test = f"{cur} {word}".strip()
        if draw.textlength(test, font=fnt) <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def fit(draw, texts: list[str], max_w: float, max_h: float, sizes: range, bold: bool = False,
        gap: float = 0.0):
    """Largest font size at which all texts, wrapped to max_w, fit in max_h.
    Returns (font, wrapped texts, line height); falls back to the smallest size."""
    for size in sizes:
        fnt = font(size, bold)
        line_h = int(size * 1.28)
        wrapped = [wrap(draw, t, fnt, max_w) for t in texts]
        total = sum(len(w) for w in wrapped) * line_h + gap * size * max(len(texts) - 1, 0)
        if total <= max_h:
            return fnt, wrapped, line_h
    return fnt, wrapped, line_h


class Canvas:
    def __init__(self) -> None:
        """Empty frame of VIDEO_WIDTH x VIDEO_HEIGHT. Sizes in this file are designed for
        1280 px wide and scaled with px()."""
        self.w, self.h = config.VIDEO_WIDTH, config.VIDEO_HEIGHT
        self.s = self.w / 1280
        self.img = Image.new("RGB", (self.w, self.h), BG)
        self.d = ImageDraw.Draw(self.img)
        self.m = int(80 * self.s)   # side margin

    def px(self, v: float) -> int:
        """Design size (for a 1280 px wide frame) -> pixels at the real width."""
        return int(v * self.s)

    def sizes(self, hi: int, lo: int) -> range:
        """Font sizes to try, largest first, scaled to the real width."""
        return range(self.px(hi), max(self.px(lo), 10) - 1, -1)

    def text_lines(self, lines: list[str], x: float, y: float, fnt, line_h: int, fill=INK,
                   center_w: float | None = None) -> float:
        """Draw lines of text from (x, y), centred in center_w if given. Returns the y below them."""
        for line in lines:
            lx = x + (center_w - self.d.textlength(line, font=fnt)) / 2 if center_w else x
            self.d.text((lx, y), line, font=fnt, fill=fill)
            y += line_h
        return y

    def chrome(self, footer: str, page: str) -> None:
        """Accent bar at the top, lesson title and page number at the bottom."""
        self.d.rectangle([0, 0, self.w, self.px(8)], fill=ACCENT)
        fnt = font(self.px(17))
        y = self.h - self.px(40)
        self.d.text((self.m, y), footer[:90], font=fnt, fill=MUTED)
        self.d.text((self.w - self.m - self.d.textlength(page, font=fnt), y), page, font=fnt, fill=MUTED)

    def heading(self, text: str) -> int:
        """Scene heading at the top left with an accent underline. Returns the y below it."""
        fnt, (lines,), line_h = fit(self.d, [text], self.w - 2 * self.m, self.px(110), self.sizes(46, 28), True)
        y = self.text_lines(lines, self.m, self.px(56), fnt, line_h)
        self.d.rectangle([self.m, y + self.px(10), self.m + self.px(80), y + self.px(16)], fill=ACCENT)
        return int(y + self.px(50))


# --- Scene types ------------------------------------------------------------------------------
def draw_title(c: Canvas, heading: str, goal: str) -> None:
    """Title scene: small label, lesson title, accent line, goal."""
    width = c.w - 2 * c.px(140)
    label_f = font(c.px(20), True)
    t_font, (t_lines,), t_lh = fit(c.d, [heading], width, c.px(230), c.sizes(64, 34), True)
    g_font, (g_lines,), g_lh = fit(c.d, [goal], width, c.px(130), c.sizes(28, 18)) if goal else (None, ([],), 0)
    block = c.px(40) + len(t_lines) * t_lh + c.px(40) + len(g_lines) * g_lh
    y = (c.h - block) / 2 - c.px(10)
    x = c.px(140)
    c.d.text((x, y), "VIDEO LESSON", font=label_f, fill=ACCENT)
    y = c.text_lines(t_lines, x, y + c.px(40), t_font, t_lh)
    c.d.rectangle([x, y + c.px(14), x + c.px(120), y + c.px(20)], fill=ACCENT)
    if g_lines:
        c.text_lines(g_lines, x, y + c.px(40), g_font, g_lh, fill=MUTED)


def draw_bullets(c: Canvas, heading: str, items: list[str]) -> None:
    """Heading plus key points with accent dots. A heading alone is shown large in the middle."""
    if not items:   # heading only: show it large in the middle
        fnt, (lines,), lh = fit(c.d, [heading], c.w - 2 * c.px(140), c.px(300), c.sizes(56, 30), True)
        c.text_lines(lines, c.px(140), (c.h - len(lines) * lh) / 2, fnt, lh, center_w=c.w - 2 * c.px(140))
        return
    top = c.heading(heading)
    indent = c.px(44)
    avail_h = c.h - top - c.px(90)
    fnt, wrapped, lh = fit(c.d, items, c.w - 2 * c.m - indent, avail_h, c.sizes(36, 18), gap=0.7)
    total = sum(len(w) for w in wrapped) * lh + 0.7 * fnt.size * (len(items) - 1)
    y = top + max(0, (avail_h - total) / 3)   # a bit above centre looks balanced
    r = max(c.px(7), 4)
    for lines in wrapped:
        cy = y + lh / 2
        c.d.ellipse([c.m + c.px(6), cy - r, c.m + c.px(6) + 2 * r, cy + r], fill=ACCENT)
        y = c.text_lines(lines, c.m + indent, y, fnt, lh) + 0.7 * fnt.size


def _arrow(c: Canvas, x1: float, y1: float, x2: float, y2: float) -> None:
    """Arrow from (x1, y1) to (x2, y2), horizontal (pointing right) or vertical (pointing down)."""
    c.d.line([x1, y1, x2, y2], fill=ACCENT, width=max(c.px(4), 2))
    head = c.px(12)
    if abs(x2 - x1) >= abs(y2 - y1):   # horizontal, pointing right
        c.d.polygon([(x2, y2), (x2 - head, y2 - head * 0.7), (x2 - head, y2 + head * 0.7)], fill=ACCENT)
    else:                              # vertical, pointing down
        c.d.polygon([(x2, y2), (x2 - head * 0.7, y2 - head), (x2 + head * 0.7, y2 - head)], fill=ACCENT)


def _number(c: Canvas, cx: float, cy: float, n: int) -> None:
    """Filled circle with a step number, centred at (cx, cy)."""
    r = c.px(22)
    c.d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ACCENT)
    f = font(c.px(22), True)
    t = str(n)
    c.d.text((cx - c.d.textlength(t, font=f) / 2, cy - c.px(14)), t, font=f, fill=(255, 255, 255))


def draw_steps(c: Canvas, heading: str, items: list[str]) -> None:
    """Process diagram: numbered boxes with arrows from left to right, or from top to
    bottom when there are many steps or long texts."""
    if len(items) < 2:
        return draw_bullets(c, heading, items)
    top = c.heading(heading)
    avail_h = c.h - top - c.px(90)
    n = len(items)
    gap = c.px(56)
    box_w = (c.w - 2 * c.m - (n - 1) * gap) / n
    pad = c.px(16)
    if n <= 5 and box_w >= c.px(170):   # horizontal flow
        text_h = avail_h - c.px(90)
        fnt, wrapped, lh = fit(c.d, items, box_w - 2 * pad, text_h, c.sizes(28, 16))
        if max(len(w) for w in wrapped) * lh <= min(text_h, c.px(240)):
            box_h = max(c.px(190), c.px(80) + max(len(w) for w in wrapped) * lh + pad)
            y0 = top + (avail_h - box_h) / 2
            for i, lines in enumerate(wrapped):
                x0 = c.m + i * (box_w + gap)
                c.d.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=c.px(14),
                                      fill=BOX, outline=BORDER, width=max(c.px(2), 1))
                c.d.rectangle([x0 + c.px(14), y0, x0 + box_w - c.px(14), y0 + c.px(5)], fill=ACCENT)
                _number(c, x0 + box_w / 2, y0 + c.px(46), i + 1)
                ty = y0 + c.px(80) + (box_h - c.px(80) - pad - len(lines) * lh) / 2
                c.text_lines(lines, x0 + pad, ty, fnt, lh, center_w=box_w - 2 * pad)
                if i < n - 1:
                    _arrow(c, x0 + box_w + c.px(8), y0 + box_h / 2, x0 + box_w + gap - c.px(8), y0 + box_h / 2)
            return
    # vertical flow: numbered rows with arrows between them
    row_gap = c.px(26)
    fnt, wrapped, lh = fit(c.d, items, c.w - 2 * c.m - c.px(90), avail_h - (n - 1) * row_gap - n * c.px(12),
                           c.sizes(30, 16))
    y = top
    for i, lines in enumerate(wrapped):
        row_h = max(len(lines) * lh, c.px(44)) + c.px(12)
        _number(c, c.m + c.px(24), y + row_h / 2, i + 1)
        c.text_lines(lines, c.m + c.px(70), y + (row_h - len(lines) * lh) / 2, fnt, lh)
        y += row_h
        if i < n - 1:
            _arrow(c, c.m + c.px(24), y + c.px(2), c.m + c.px(24), y + row_gap - c.px(2))
            y += row_gap


def mask_area(im: Image.Image, spec: str) -> Image.Image:
    """Paint over part of a slide (e.g. the presenter's camera picture) with the slide's own
    background colour, taken from just outside that area."""
    import numpy as np
    import media
    l, t, r, b = media.parse_crop(spec)
    w, h = im.size
    box = [int(l * w), int(t * h), int(r * w), int(b * h)]
    arr = np.asarray(im)
    band = max(4, w // 100)
    samples = [arr[max(box[1] - band, 0):box[1], box[0]:box[2]].reshape(-1, 3),     # above
               arr[box[1]:box[3], max(box[0] - band, 0):box[0]].reshape(-1, 3)]     # left
    samples = [s for s in samples if len(s)]
    colour = tuple(int(v) for v in np.median(np.concatenate(samples), axis=0)) if samples else (255, 255, 255)
    im = im.copy()
    ImageDraw.Draw(im).rectangle(box, fill=colour)
    return im


def draw_slide(c: Canvas, image: Path, caption: str) -> None:
    """Original slide, as large as fits, with a soft shadow and its source underneath."""
    top, bottom = c.px(36), c.h - c.px(80)
    max_w, max_h = c.w - c.m, bottom - top
    with Image.open(image) as im:
        im = im.convert("RGB")
        if config.SLIDE_MASK:
            im = mask_area(im, config.SLIDE_MASK)
        scale = min(max_w / im.width, max_h / im.height)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    x, y = (c.w - im.width) // 2, top + (max_h - im.height) // 2
    sh = c.px(6)
    c.d.rectangle([x + sh, y + sh, x + im.width + sh, y + im.height + sh], fill=(226, 232, 240))
    c.img.paste(im, (x, y))
    c.d.rectangle([x, y, x + im.width, y + im.height], outline=BORDER, width=max(c.px(1), 1))
    if caption:
        f = font(c.px(17))
        c.d.text((x, y + im.height + c.px(10)), caption[:120], font=f, fill=MUTED)


def render_scene(scene: dict, out: Path, *, lesson_title: str, goal: str, page: str,
                 slide_image: Path | None = None, caption: str = "") -> Path:
    """Draw one scene to a PNG. `scene` is a storyboard scene dict."""
    c = Canvas()
    kind = scene.get("visual", "bullets")
    if kind == "title":
        draw_title(c, scene.get("heading") or lesson_title, goal)
    elif kind == "slide" and slide_image and slide_image.exists():
        draw_slide(c, slide_image, caption)
    elif kind == "steps":
        draw_steps(c, scene.get("heading", ""), scene.get("items", []))
    else:
        draw_bullets(c, scene.get("heading", ""), scene.get("items", []))
    if kind != "title":   # slide scenes already have a caption at the bottom
        c.chrome("" if kind == "slide" and slide_image else lesson_title, page)
    out.parent.mkdir(parents=True, exist_ok=True)
    c.img.save(out)
    return out


if __name__ == "__main__":
    # Re-draw the scenes of a storyboard.json (after editing it by hand) to check the look.
    sb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not sb_path or not sb_path.exists():
        print("Usage: uv run render.py data/output/<folder>/storyboard.json")
        sys.exit(1)
    sb = json.loads(sb_path.read_text(encoding="utf-8"))
    for i, sc in enumerate(sb["scenes"], 1):
        img = Path(sc["slide_image"]) if sc.get("slide_image") else None
        p = render_scene(sc, sb_path.parent / "scenes" / f"scene_{i:02d}.png", lesson_title=sb["title"],
                         goal=sb.get("goal", ""), page=f"{i} / {len(sb['scenes'])}",
                         slide_image=img, caption=sc.get("caption", ""))
        print(p)
