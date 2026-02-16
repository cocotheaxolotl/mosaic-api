"""
Mystery Mosaic Engine — Core generation logic (no CLI, no file I/O).
Called by the FastAPI app.  Returns PIL Images in memory.
"""

import io
import math
import zipfile
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.cluster import KMeans


# ── Fonts ────────────────────────────────────────────────────────────────

_FONT_PATHS = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    for fp in _FONT_PATHS:
        try:
            return ImageFont.truetype(fp, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ── Color helpers ────────────────────────────────────────────────────────

def rgb_to_name(r: int, g: int, b: int) -> str:
    """Human-readable colour name (EN)."""
    r, g, b = int(r), int(g), int(b)
    mx, mn = max(r, g, b), min(r, g, b)
    bri = (r + g + b) / 3
    if mx - mn < 30:
        if bri > 200: return "White"
        if bri > 140: return "Light Grey"
        if bri > 80:  return "Grey"
        return "Black"
    if r >= g and r >= b:
        if g > b + 40:
            return "Yellow" if r > 200 and g > 200 else "Orange"
        if b > g + 40: return "Purple"
        if r > 180 and g < 100 and b < 100: return "Red"
        if r > 180 and g > 100: return "Salmon"
        return "Pink"
    if g >= r and g >= b:
        if b > r + 40: return "Turquoise"
        if r > b + 20 and r > 150: return "Yellow-Green"
        return "Light Green" if g > 180 else "Green"
    if r > g + 30 and r > 100: return "Purple"
    if g > r + 10: return "Sky Blue"
    return "Light Blue" if b > 200 else "Blue"


# ── Geometry ─────────────────────────────────────────────────────────────

def _hex_vertices(cx: float, cy: float, size: float):
    angles = [math.radians(60 * i) for i in range(6)]
    return [(cx + size * math.cos(a), cy + size * math.sin(a)) for a in angles]


def _hex_grid(w: int, h: int, cell: int):
    hw = cell * 2
    hh = cell * math.sqrt(3)
    cs = hw * 0.75
    rs = hh
    centers = []
    for col in range(int(w / cs) + 2):
        for row in range(int(h / rs) + 2):
            cx = col * cs
            cy = row * rs + (rs / 2 if col % 2 else 0)
            if -cell < cx < w + cell and -cell < cy < h + cell:
                centers.append((cx, cy))
    return centers


# ── Quantization ─────────────────────────────────────────────────────────

def _quantize(img: Image.Image, n: int):
    pixels = np.array(img).reshape(-1, 3).astype(np.float64)
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    labels = km.fit_predict(pixels)
    palette = km.cluster_centers_.astype(np.uint8)

    # Shuffle so number order doesn't reveal the image
    rng = np.random.RandomState(12345)
    order = np.arange(n)
    rng.shuffle(order)
    remap = np.zeros(n, dtype=int)
    for new, old in enumerate(order):
        remap[old] = new
    labels = remap[labels]
    palette = palette[order]

    return labels.reshape(img.size[1], img.size[0]), palette


def _sample(label_map, cx, cy, cell, w, h):
    r = max(2, int(cell * 0.4))
    ys, ye = max(0, int(cy) - r), min(h, int(cy) + r)
    xs, xe = max(0, int(cx) - r), min(w, int(cx) + r)
    if ys >= ye or xs >= xe:
        return 0
    region = label_map[ys:ye, xs:xe]
    return np.bincount(region.flatten()).argmax() if region.size else 0


# ── Page sizes ───────────────────────────────────────────────────────────

PAGE_SIZES = {
    "letter":    (2550, 3300),   # 8.5 × 11 in @ 300 dpi
    "a4":        (2480, 3508),   # 210 × 297 mm @ 300 dpi
}


# ── Preset definitions ──────────────────────────────────────────────────

@dataclass
class Preset:
    name: str
    label: str
    page: str           # "letter" or "a4"
    cell_size: int      # hex cell px on the print page
    colors: int         # number of palette colours
    landscape: bool = False
    crop: Optional[str] = None       # "L,T,R,B" %
    blur: int = 0

PRESETS = {
    "kdp-letter": Preset("kdp-letter", "KDP 8.5×11 no bleed", "letter", 25, 12),
    "kdp-a4":     Preset("kdp-a4",     "A4 print-ready",      "a4",     25, 12),
    "kids-easy":  Preset("kids-easy",   "Kids 4-6 yrs (easy)", "letter", 35, 8),
    "kids-detail": Preset("kids-detail","Kids 7-10 yrs",       "letter", 25, 12),
    "adult-fine": Preset("adult-fine",  "Adult detailed",      "letter", 18, 16),
}


# ── Main generation ─────────────────────────────────────────────────────

@dataclass
class MosaicResult:
    mystery: Image.Image       # numbered hexagons (B/W)
    answer: Image.Image        # coloured answer key
    legend: Image.Image        # colour legend strip
    mystery_full: Image.Image  # mystery + legend combined
    hex_count: int
    color_count: int
    palette: list              # list of (r,g,b) tuples


def generate(
    img: Image.Image,
    colors: int = 12,
    cell_size: int = 25,
    page: str = "letter",
    landscape: bool = False,
    crop: Optional[str] = None,
    blur: int = 0,
    upscale: int = 2,
) -> MosaicResult:
    """Generate a mystery mosaic from a PIL Image.  Pure in-memory."""

    img = img.convert("RGB")

    # Optional crop
    if crop:
        parts = [float(x) for x in crop.split(",")]
        if len(parts) == 4:
            lp, tp, rp, bp = parts
            w, h = img.size
            img = img.crop((
                int(w * lp / 100), int(h * tp / 100),
                int(w * (100 - rp) / 100), int(h * (100 - bp) / 100),
            ))

    # Optional blur
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    # Fit to page
    if page and page in PAGE_SIZES:
        pw, ph = PAGE_SIZES[page]
        if landscape:
            pw, ph = ph, pw
        margin = 50
        tw, th = pw - 2 * margin, ph - 2 * margin
        s = min(tw / img.size[0], th / img.size[1])
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.LANCZOS)
    elif upscale > 1:
        img = img.resize((img.size[0] * upscale, img.size[1] * upscale), Image.LANCZOS)

    w, h = img.size

    # Quantize
    label_map, palette = _quantize(img, colors)

    # Hex grid
    centers = _hex_grid(w, h, cell_size)
    hex_data = [(cx, cy, _sample(label_map, cx, cy, cell_size, w, h)) for cx, cy in centers]

    # ── Mystery page (B/W numbered hexagons) ──
    mystery = Image.new("RGB", (w, h), (255, 255, 255))
    dm = ImageDraw.Draw(mystery)
    font = _get_font(max(6, int(cell_size * 0.45)))

    for cx, cy, ci in hex_data:
        verts = _hex_vertices(cx, cy, cell_size)
        dm.polygon(verts, outline=(0, 0, 0), width=1)
        num = str(ci + 1)
        bb = dm.textbbox((0, 0), num, font=font)
        dm.text((cx - (bb[2] - bb[0]) / 2, cy - (bb[3] - bb[1]) / 2),
                num, fill=(80, 80, 80), font=font)

    # ── Answer key (coloured) ──
    answer = Image.new("RGB", (w, h), (255, 255, 255))
    da = ImageDraw.Draw(answer)
    for cx, cy, ci in hex_data:
        da.polygon(_hex_vertices(cx, cy, cell_size),
                   outline=(40, 40, 40), fill=tuple(palette[ci]), width=1)

    # ── Legend ──
    legend = _make_legend(palette, w)

    # ── Combined ──
    mystery_full = _stack(mystery, legend)

    pal_list = [tuple(int(c) for c in row) for row in palette]

    return MosaicResult(
        mystery=mystery,
        answer=answer,
        legend=legend,
        mystery_full=mystery_full,
        hex_count=len(hex_data),
        color_count=colors,
        palette=pal_list,
    )


def generate_from_preset(img: Image.Image, preset_name: str) -> MosaicResult:
    """Convenience: generate using a named preset."""
    p = PRESETS[preset_name]
    return generate(img, colors=p.colors, cell_size=p.cell_size,
                    page=p.page, landscape=p.landscape,
                    crop=p.crop, blur=p.blur)


# ── Legend / stack helpers ───────────────────────────────────────────────

def _make_legend(palette, img_w: int) -> Image.Image:
    n = len(palette)
    sw, pad, tw = 40, 12, 200
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    lw = max(cols * (sw + tw + pad) + pad, img_w)
    lh = rows * (sw + pad) + pad + 45

    legend = Image.new("RGB", (lw, lh), (255, 255, 255))
    d = ImageDraw.Draw(legend)
    d.text((pad, 8), "Color Legend", fill=(0, 0, 0), font=_get_font(20))

    font = _get_font(16)
    for i, c in enumerate(palette):
        col, row = i % cols, i // cols
        x = pad + col * (sw + tw + pad)
        y = 45 + pad + row * (sw + pad)
        d.rectangle([x, y, x + sw, y + sw], fill=tuple(c), outline=(0, 0, 0), width=2)
        d.text((x + sw + 5, y + 10),
               f"  {i+1} = {rgb_to_name(c[0], c[1], c[2])}",
               fill=(0, 0, 0), font=font)
    return legend


def _stack(top: Image.Image, bottom: Image.Image, gap: int = 10) -> Image.Image:
    w = max(top.width, bottom.width)
    h = top.height + bottom.height + gap
    out = Image.new("RGB", (w, h), (255, 255, 255))
    out.paste(top, ((w - top.width) // 2, 0))
    out.paste(bottom, ((w - bottom.width) // 2, top.height + gap))
    return out


# ── ZIP helper (for bulk) ───────────────────────────────────────────────

def images_to_zip(results: List[Tuple[str, MosaicResult]]) -> bytes:
    """Pack multiple MosaicResults into a ZIP (in memory).
    results = [(name, MosaicResult), ...]
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, res in results:
            for suffix, img in [
                ("mystery", res.mystery),
                ("mystery-full", res.mystery_full),
                ("answer", res.answer),
                ("legend", res.legend),
            ]:
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG", dpi=(300, 300))
                zf.writestr(f"{name}/{name}-{suffix}.png", img_buf.getvalue())
    buf.seek(0)
    return buf.getvalue()
