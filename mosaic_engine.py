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
from scipy.spatial import Voronoi


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


def _sample_avg_color(pixels, cx, cy, cell, w, h):
    """Sample the average RGB color in a region around (cx, cy)."""
    r = max(2, int(cell * 0.45))
    ys, ye = max(0, int(cy) - r), min(h, int(cy) + r)
    xs, xe = max(0, int(cx) - r), min(w, int(cx) + r)
    if ys >= ye or xs >= xe:
        return (200, 200, 200)
    region = pixels[ys:ye, xs:xe]
    avg = region.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    return tuple(int(c) for c in avg)


# ── Voronoi helpers ──────────────────────────────────────────────────────

# Density presets calibrated for printable results at 300 DPI.
# min_cell_area ensures every cell is >= ~3-4 mm at 300 DPI on print.
# 300 DPI, 3 mm ≈ 35 px → area ≈ 35² ≈ 1225 px² (conservative floor).
_VORONOI_MIN_CELL_AREA = 900   # px² — cells below this get no number
_VORONOI_SKIP_AREA = 400       # px² — cells below this are merged

# Font size for numbers: clamp(min, k*sqrt(area), max)
_V_FONT_MIN = 8
_V_FONT_MAX = 28
_V_FONT_K = 0.28


def _voronoi_points(w: int, h: int, density: int) -> np.ndarray:
    """Generate Poisson-ish random points with margin mirroring.

    density = approximate number of cells desired.
    We add mirrored points outside the image so edge cells clip nicely.
    """
    rng = np.random.RandomState(42)
    n = max(50, density)
    pts = np.column_stack([rng.uniform(0, w, n), rng.uniform(0, h, n)])

    # Mirror points beyond edges to close boundary cells
    margin_pts = []
    for px, py in pts:
        if px < w * 0.15:
            margin_pts.append([-px, py])
        if px > w * 0.85:
            margin_pts.append([2 * w - px, py])
        if py < h * 0.15:
            margin_pts.append([px, -py])
        if py > h * 0.85:
            margin_pts.append([px, 2 * h - py])
    if margin_pts:
        pts = np.vstack([pts, margin_pts])
    return pts


def _clip_polygon(verts: list, w: int, h: int) -> list:
    """Sutherland-Hodgman clip polygon to [0,0]–[w,h] rectangle."""

    def _clip_edge(poly, x0, y0, x1, y1):
        """Clip polygon against one edge defined by two points."""
        if not poly:
            return []
        out = []
        n = len(poly)
        for i in range(n):
            cur = poly[i]
            nxt = poly[(i + 1) % n]
            cc = (x1 - x0) * (cur[1] - y0) - (y1 - y0) * (cur[0] - x0)
            cn = (x1 - x0) * (nxt[1] - y0) - (y1 - y0) * (nxt[0] - x0)
            if cc >= 0:
                out.append(cur)
                if cn < 0:
                    out.append(_intersect(cur, nxt, x0, y0, x1, y1))
            elif cn >= 0:
                out.append(_intersect(cur, nxt, x0, y0, x1, y1))
        return out

    def _intersect(p1, p2, x0, y0, x1, y1):
        dx, dy = x1 - x0, y1 - y0
        dp = (p2[0] - p1[0], p2[1] - p1[1])
        t1 = (dx * (p1[1] - y0) - dy * (p1[0] - x0))
        t2 = (dy * dp[0] - dx * dp[1])
        if abs(t2) < 1e-10:
            return p1
        t = t1 / t2
        return (p1[0] + t * dp[0], p1[1] + t * dp[1])

    # Clip against all 4 edges: left, bottom, right, top
    poly = list(verts)
    poly = _clip_edge(poly, 0, 0, w, 0)       # top
    poly = _clip_edge(poly, w, 0, w, h)       # right
    poly = _clip_edge(poly, w, h, 0, h)       # bottom
    poly = _clip_edge(poly, 0, h, 0, 0)       # left
    return poly


def _polygon_area(verts: list) -> float:
    """Shoelace formula for polygon area."""
    n = len(verts)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += verts[i][0] * verts[j][1]
        a -= verts[j][0] * verts[i][1]
    return abs(a) / 2.0


def _polygon_centroid(verts: list) -> Tuple[float, float]:
    """Compute centroid of a polygon. Falls back to mean if degenerate."""
    n = len(verts)
    if n < 3:
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        return (sum(xs) / max(len(xs), 1), sum(ys) / max(len(ys), 1))
    area = _polygon_area(verts)
    if area < 1e-6:
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        return (sum(xs) / n, sum(ys) / n)
    cx = cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = verts[i][0] * verts[j][1] - verts[j][0] * verts[i][1]
        cx += (verts[i][0] + verts[j][0]) * cross
        cy += (verts[i][1] + verts[j][1]) * cross
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return (abs(cx), abs(cy))


def _point_in_polygon(px: float, py: float, verts: list) -> bool:
    """Ray-casting test."""
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _safe_label_point(verts: list, w: int, h: int) -> Optional[Tuple[float, float]]:
    """Return a point guaranteed inside the polygon, or None if too small."""
    cx, cy = _polygon_centroid(verts)
    # Clamp to image bounds
    cx = max(0.0, min(float(w), cx))
    cy = max(0.0, min(float(h), cy))
    if _point_in_polygon(cx, cy, verts):
        return (cx, cy)
    # Centroid outside: try mean of vertices
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    if _point_in_polygon(mx, my, verts):
        return (mx, my)
    # Last resort: midpoint of longest edge
    best_len = 0
    best_mid = (cx, cy)
    for i in range(len(verts)):
        j = (i + 1) % len(verts)
        dx = verts[j][0] - verts[i][0]
        dy = verts[j][1] - verts[i][1]
        d = dx * dx + dy * dy
        if d > best_len:
            best_len = d
            best_mid = ((verts[i][0] + verts[j][0]) / 2,
                        (verts[i][1] + verts[j][1]) / 2)
    return best_mid


def _build_voronoi_cells(w: int, h: int, density: int):
    """Build clipped Voronoi cells.

    Returns list of (clipped_verts, area, label_point, seed_x, seed_y)
    for cells with area >= _VORONOI_SKIP_AREA.
    """
    pts = _voronoi_points(w, h, density)
    vor = Voronoi(pts)

    cells = []
    for idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        verts = [tuple(vor.vertices[i]) for i in region]
        clipped = _clip_polygon(verts, w, h)
        if len(clipped) < 3:
            continue
        area = _polygon_area(clipped)
        if area < _VORONOI_SKIP_AREA:
            continue
        lbl = _safe_label_point(clipped, w, h)
        sx, sy = pts[idx]
        # Only keep cells whose seed is inside the image
        if 0 <= sx <= w and 0 <= sy <= h:
            cells.append((clipped, area, lbl, sx, sy))
    return cells


def _voronoi_sample_color(pixels: np.ndarray, verts: list, w: int, h: int):
    """Average RGB inside a polygon using a bounding-box scan."""
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    x0 = max(0, int(min(xs)))
    x1 = min(w, int(max(xs)) + 1)
    y0 = max(0, int(min(ys)))
    y1 = min(h, int(max(ys)) + 1)
    if x0 >= x1 or y0 >= y1:
        return np.array([200, 200, 200], dtype=np.uint8)

    region = pixels[y0:y1, x0:x1]
    # Simple bbox average (fast, good enough for most cell sizes)
    avg = region.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    return avg


# ── Page sizes ───────────────────────────────────────────────────────────

PAGE_SIZES = {
    "letter":    (2550, 3300),   # 8.5 × 11 in @ 300 dpi
    "a4":        (2480, 3508),   # 210 × 297 mm @ 300 dpi
}


# ── Voronoi density presets (calibrated for print) ──────────────────────
# Each ensures min cell area ~3-4mm at 300 DPI on the given page size.

VORONOI_DENSITIES = {
    "letter": {"easy": 200, "standard": 500, "detailed": 900, "expert": 1400},
    "a4":     {"easy": 200, "standard": 500, "detailed": 900, "expert": 1400},
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
    mode: str = "hex"                # "hex" or "voronoi"
    voronoi_density: str = "standard"  # easy/standard/detailed/expert

PRESETS = {
    "kdp-letter": Preset("kdp-letter", "KDP 8.5×11 no bleed", "letter", 25, 12),
    "kdp-a4":     Preset("kdp-a4",     "A4 print-ready",      "a4",     25, 12),
    "kids-easy":  Preset("kids-easy",   "Kids 4-6 yrs (easy)", "letter", 35, 8),
    "kids-detail": Preset("kids-detail","Kids 7-10 yrs",       "letter", 25, 12),
    "adult-fine": Preset("adult-fine",  "Adult detailed",      "letter", 18, 16),
    # Voronoi presets
    "voronoi-easy":     Preset("voronoi-easy",     "Organic Mosaic (easy)",     "letter", 0, 10, mode="voronoi", voronoi_density="easy"),
    "voronoi-standard": Preset("voronoi-standard", "Organic Mosaic (standard)", "letter", 0, 12, mode="voronoi", voronoi_density="standard"),
    "voronoi-detailed": Preset("voronoi-detailed", "Organic Mosaic (detailed)", "letter", 0, 16, mode="voronoi", voronoi_density="detailed"),
    "voronoi-expert":   Preset("voronoi-expert",   "Organic Mosaic (expert)",   "letter", 0, 20, mode="voronoi", voronoi_density="expert"),
}


# ── Main generation ─────────────────────────────────────────────────────

@dataclass
class MosaicResult:
    mystery: Image.Image       # numbered hexagons (B/W)
    answer: Image.Image        # coloured answer key (quantized palette)
    beauty: Image.Image        # beauty answer (real avg colors, no outlines)
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

    # ── Answer key (coloured, quantized palette) ──
    answer = Image.new("RGB", (w, h), (255, 255, 255))
    da = ImageDraw.Draw(answer)
    for cx, cy, ci in hex_data:
        da.polygon(_hex_vertices(cx, cy, cell_size),
                   outline=(40, 40, 40), fill=tuple(palette[ci]), width=1)

    # ── Beauty answer (real average colors, no outlines — for preview) ──
    pixels = np.array(img)
    beauty = Image.new("RGB", (w, h), (255, 255, 255))
    db = ImageDraw.Draw(beauty)
    for cx, cy in centers:
        avg_color = _sample_avg_color(pixels, cx, cy, cell_size, w, h)
        db.polygon(_hex_vertices(cx, cy, cell_size),
                   fill=avg_color, outline=avg_color)

    # ── Legend ──
    legend = _make_legend(palette, w)

    # ── Combined ──
    mystery_full = _stack(mystery, legend)

    pal_list = [tuple(int(c) for c in row) for row in palette]

    return MosaicResult(
        mystery=mystery,
        answer=answer,
        beauty=beauty,
        legend=legend,
        mystery_full=mystery_full,
        hex_count=len(hex_data),
        color_count=colors,
        palette=pal_list,
    )


# ── Voronoi generation ──────────────────────────────────────────────────

def generate_voronoi(
    img: Image.Image,
    colors: int = 12,
    density: int = 500,
    page: str = "letter",
    landscape: bool = False,
    crop: Optional[str] = None,
    blur: int = 0,
    upscale: int = 2,
) -> MosaicResult:
    """Generate a Voronoi mystery mosaic from a PIL Image.

    3 guardrails:
    1. Cells with area < threshold get no number (or are skipped)
    2. Density is capped by presets to guarantee printable cell sizes
    3. Numbers have white halo + font clamped to cell size
    """

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
    pixels = np.array(img)

    # Cap density to sane limits
    density = max(50, min(2000, density))

    # Build Voronoi cells
    cells = _build_voronoi_cells(w, h, density)

    # Quantize
    label_map, palette = _quantize(img, colors)

    # Assign each cell a color index (dominant label in the cell region)
    cell_data = []
    for verts, area, lbl_pt, sx, sy in cells:
        # Use seed point to sample the dominant quantized color
        ci = _sample(label_map, sx, sy, max(10, int(math.sqrt(area) * 0.4)), w, h)
        cell_data.append((verts, area, lbl_pt, ci))

    # ── Mystery page (B/W numbered polygons) ──
    mystery = Image.new("RGB", (w, h), (255, 255, 255))
    dm = ImageDraw.Draw(mystery)

    unlabeled = 0
    for verts, area, lbl_pt, ci in cell_data:
        # Draw cell outline
        dm.polygon([tuple(v) for v in verts], outline=(0, 0, 0), width=1)

        # GUARDRAIL 1: skip number if cell too small
        if area < _VORONOI_MIN_CELL_AREA or lbl_pt is None:
            unlabeled += 1
            continue

        # GUARDRAIL 3: font size clamped to cell area
        font_size = int(max(_V_FONT_MIN, min(_V_FONT_MAX, _V_FONT_K * math.sqrt(area))))
        font = _get_font(font_size)

        num = str(ci + 1)
        bb = dm.textbbox((0, 0), num, font=font)
        tw_txt = bb[2] - bb[0]
        th_txt = bb[3] - bb[1]

        # Check if text fits in cell (rough: text bbox < 80% of sqrt(area))
        if tw_txt > math.sqrt(area) * 0.8 or th_txt > math.sqrt(area) * 0.8:
            unlabeled += 1
            continue

        tx = lbl_pt[0] - tw_txt / 2
        ty = lbl_pt[1] - th_txt / 2

        # White halo/stroke for readability
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx != 0 or dy != 0:
                    dm.text((tx + dx, ty + dy), num, fill=(255, 255, 255), font=font)
        dm.text((tx, ty), num, fill=(60, 60, 60), font=font)

    # ── Answer key (coloured, quantized palette) ──
    answer = Image.new("RGB", (w, h), (255, 255, 255))
    da = ImageDraw.Draw(answer)
    for verts, area, lbl_pt, ci in cell_data:
        da.polygon([tuple(v) for v in verts],
                   outline=(40, 40, 40), fill=tuple(palette[ci]), width=1)

    # ── Beauty answer (real average colors, no outlines) ──
    beauty = Image.new("RGB", (w, h), (255, 255, 255))
    db = ImageDraw.Draw(beauty)
    for verts, area, lbl_pt, ci in cells_and_avg_colors(cells, pixels, w, h):
        db.polygon([tuple(v) for v in verts], fill=ci, outline=ci)

    # ── Legend ──
    legend = _make_legend(palette, w)

    # ── Combined ──
    mystery_full = _stack(mystery, legend)

    pal_list = [tuple(int(c) for c in row) for row in palette]

    return MosaicResult(
        mystery=mystery,
        answer=answer,
        beauty=beauty,
        legend=legend,
        mystery_full=mystery_full,
        hex_count=len(cell_data),
        color_count=colors,
        palette=pal_list,
    )


def cells_and_avg_colors(cells, pixels, w, h):
    """Yield (verts, area, lbl_pt, avg_color_tuple) for beauty rendering."""
    for verts, area, lbl_pt, sx, sy in cells:
        avg = _voronoi_sample_color(pixels, verts, w, h)
        yield (verts, area, lbl_pt, tuple(int(c) for c in avg))


def generate_from_preset(img: Image.Image, preset_name: str) -> MosaicResult:
    """Convenience: generate using a named preset."""
    p = PRESETS[preset_name]
    if p.mode == "voronoi":
        density_map = VORONOI_DENSITIES.get(p.page, VORONOI_DENSITIES["letter"])
        density = density_map.get(p.voronoi_density, 500)
        return generate_voronoi(img, colors=p.colors, density=density,
                                page=p.page, landscape=p.landscape,
                                crop=p.crop, blur=p.blur)
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
                ("beauty", res.beauty),
                ("legend", res.legend),
            ]:
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG", dpi=(300, 300))
                zf.writestr(f"{name}/{name}-{suffix}.png", img_buf.getvalue())
    buf.seek(0)
    return buf.getvalue()
