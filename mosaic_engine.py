"""
Mystery Mosaic Engine — Core generation logic (no CLI, no file I/O).
Called by the FastAPI app.  Returns PIL Images in memory.
"""

import gc
import io
import math
import zipfile
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.cluster import MiniBatchKMeans
# scipy is lazy-imported in _build_voronoi_cells() to save ~150 MB RAM at startup


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
    pixels = np.array(img).reshape(-1, 3).astype(np.float32)
    # Subsample for fast clustering, then predict all pixels
    MAX_SAMPLES = 50_000
    if len(pixels) > MAX_SAMPLES:
        idx = np.random.RandomState(42).choice(len(pixels), MAX_SAMPLES, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels
    km = MiniBatchKMeans(n_clusters=n, random_state=42, batch_size=1024, n_init=3)
    km.fit(sample)
    labels = km.predict(pixels)
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


def _flatten_background(img: Image.Image, tolerance: int = 60) -> Image.Image:
    """Detect dominant border color and replace similar pixels with white.

    Samples a 5-pixel band around all four edges, finds the most common color
    cluster, then replaces all pixels within `tolerance` Euclidean distance
    of that color with pure white.  Returns a new PIL Image.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    band = 5

    # Collect border pixels (top, bottom, left, right bands)
    edges = np.concatenate([
        arr[:band, :].reshape(-1, 3),       # top
        arr[h - band:, :].reshape(-1, 3),   # bottom
        arr[:, :band].reshape(-1, 3),       # left
        arr[:, w - band:].reshape(-1, 3),   # right
    ])

    # Find dominant border color via small KMeans (3 clusters)
    from sklearn.cluster import MiniBatchKMeans as _KM
    n_clusters = min(3, len(edges))
    km = _KM(n_clusters=n_clusters, random_state=0, n_init=1, batch_size=512)
    km.fit(edges)
    counts = np.bincount(km.labels_, minlength=n_clusters)
    dominant = km.cluster_centers_[counts.argmax()]

    # Skip if border is already close to white (no need to flatten)
    if np.linalg.norm(dominant - np.array([255, 255, 255])) < 30:
        return img

    # Replace pixels close to the dominant border color with white
    dist = np.linalg.norm(arr - dominant, axis=2)
    mask = dist < tolerance
    out = arr.copy()
    out[mask] = [255, 255, 255]

    return Image.fromarray(out.astype(np.uint8), "RGB")


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
    try:
        from scipy.spatial import Voronoi as _Voronoi
    except ImportError:
        raise RuntimeError("Voronoi mode requires scipy. Install with: pip install scipy")
    pts = _voronoi_points(w, h, density)
    vor = _Voronoi(pts)

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
    mode: str = "hex"                # "hex" or "voronoi" or "cbn" or "lineart"
    lineart_detail: str = "standard"   # Line art: simple/standard/detailed/expert
    lineart_thickness: int = 2         # Line art: line thickness 1-4
    voronoi_density: str = "standard"  # easy/standard/detailed/expert
    min_zone_pixels: int = 500         # CBN: minimum zone size in pixels
    epsilon_factor: float = 0.002      # CBN: contour simplification factor

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
    # Color by Number presets
    "cbn-kids":     Preset("cbn-kids",     "CBN Kids (easy)",    "letter", 0, 6,  blur=3, mode="cbn", min_zone_pixels=1000),
    "cbn-standard": Preset("cbn-standard", "CBN Standard",       "letter", 0, 10, blur=2, mode="cbn", min_zone_pixels=500),
    "cbn-detailed": Preset("cbn-detailed", "CBN Detailed",       "letter", 0, 14, blur=1, mode="cbn", min_zone_pixels=300),
    "cbn-a4":       Preset("cbn-a4",       "CBN A4 Print-ready", "a4",     0, 10, blur=2, mode="cbn", min_zone_pixels=500),
    # Line Art presets
    "lineart-simple":   Preset("lineart-simple",   "Line Art Simple (Kids 3-6)",  "letter", 0, 0, mode="lineart", lineart_detail="simple",   lineart_thickness=3),
    "lineart-standard": Preset("lineart-standard", "Line Art Standard",           "letter", 0, 0, mode="lineart", lineart_detail="standard", lineart_thickness=2),
    "lineart-detailed": Preset("lineart-detailed", "Line Art Detailed",           "letter", 0, 0, mode="lineart", lineart_detail="detailed", lineart_thickness=2),
    "lineart-expert":   Preset("lineart-expert",   "Line Art Expert",             "letter", 0, 0, mode="lineart", lineart_detail="expert",   lineart_thickness=1),
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


@dataclass
class LineArtResult:
    line_art_svg: str            # SVG: black contours on white
    line_art_png: Image.Image    # PNG: 300 DPI print-ready
    preview_png: Image.Image     # PNG: lower-res for browser preview
    detail_level: str
    line_count: int


@dataclass
class CBNResult:
    mystery_svg: str              # SVG: numbered outlines (B&W)
    mystery_png: Image.Image      # PNG render of mystery
    answer_svg: str               # SVG: colored regions
    answer_png: Image.Image       # PNG render of answer
    legend_png: Image.Image       # Color legend
    mystery_full_png: Image.Image # mystery + legend stacked
    zone_count: int
    color_count: int
    palette: list                 # list of (r,g,b) tuples


@dataclass
class PBNResult:
    mystery_svg: str              # SVG: white zones, numbered, Bezier outlines
    mystery_png: Image.Image      # PNG render of mystery (300 DPI)
    answer_svg: str               # SVG: colored zones, Bezier outlines
    answer_png: Image.Image       # PNG render of answer
    legend_png: Image.Image       # Color legend
    mystery_full_png: Image.Image # mystery + legend stacked
    zone_count: int
    color_count: int
    palette: list                 # list of (r,g,b) tuples


# ── Color by Number helpers ──────────────────────────────────────────────

def _cbn_find_zones(
    label_map: np.ndarray,
    n_colors: int,
    min_zone_pixels: int = 500,
) -> Tuple[np.ndarray, list]:
    """Find connected zones per color in the quantized label map.

    Returns (zone_map, zones) where:
      - zone_map: int32 array, each pixel → zone_id (0 = unassigned/too small)
      - zones: list of {"id", "color", "centroid", "area"} dicts

    Optimized: uses find_objects() for fast bounding-box slicing instead of
    scanning the full image per zone.
    """
    from scipy import ndimage  # lazy import

    h, w = label_map.shape
    zone_map = np.zeros((h, w), dtype=np.int32)
    zones = []
    zone_id = 0

    for ci in range(n_colors):
        binary = (label_map == ci).astype(np.uint8)
        labeled, n_features = ndimage.label(binary)

        # find_objects gives bounding box slices for each component — avoids full scans
        slices = ndimage.find_objects(labeled)

        for rid, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            # Work within the bounding box only
            crop = labeled[sl]
            crop_mask = (crop == rid)
            area = int(crop_mask.sum())
            if area < min_zone_pixels:
                continue

            zone_id += 1
            zone_map[sl][crop_mask] = zone_id

            # Centroid within the crop, then offset to full image coords
            ys, xs = np.where(crop_mask)
            cy = int(ys.mean()) + sl[0].start
            cx = int(xs.mean()) + sl[1].start

            # Ensure centroid is inside the zone
            if cy < 0 or cy >= h or cx < 0 or cx >= w or not (zone_map[cy, cx] == zone_id):
                # Offset ys/xs to full coords
                ys_full = ys + sl[0].start
                xs_full = xs + sl[1].start
                dists = (ys_full - cy) ** 2 + (xs_full - cx) ** 2
                nearest = np.argmin(dists)
                cy, cx = int(ys_full[nearest]), int(xs_full[nearest])

            zones.append({
                "id": zone_id,
                "color": ci,
                "centroid": (cy, cx),
                "area": area,
            })

        del labeled
    return zone_map, zones


def generate_cbn(
    img: Image.Image,
    colors: int = 10,
    page: str = "letter",
    landscape: bool = False,
    crop: Optional[str] = None,
    blur: int = 2,
    min_zone_pixels: int = 500,
) -> CBNResult:
    """Generate a Color-by-Number page from a PIL Image.

    Pipeline: preprocess → quantize → find zones → Potrace trace →
              render SVG + PNG → legend → return CBNResult.
    """
    img = img.convert("RGB")

    # Crop
    if crop:
        parts = [float(x) for x in crop.split(",")]
        if len(parts) == 4:
            lp, tp, rp, bp = parts
            w, h = img.size
            img = img.crop((
                int(w * lp / 100), int(h * tp / 100),
                int(w * (100 - rp) / 100), int(h * (100 - bp) / 100),
            ))

    # Blur (smooths noise in photos → cleaner zones)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    # Flatten background: replace dominant border color with white
    img = _flatten_background(img)

    # Fit to page
    if page and page in PAGE_SIZES:
        pw, ph = PAGE_SIZES[page]
        if landscape:
            pw, ph = ph, pw
        margin = 50
        tw, th = pw - 2 * margin, ph - 2 * margin
        s = min(tw / img.size[0], th / img.size[1])
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.LANCZOS)

    w, h = img.size

    # Quantize
    label_map, palette = _quantize(img, colors)

    # Find zones
    zone_map, zones = _cbn_find_zones(label_map, colors, min_zone_pixels)
    del label_map

    # Extract contours with Potrace (smooth Bezier curves)
    zones = _pbn_extract_paths(zone_map, zones)

    # Render SVG
    mystery_svg = _pbn_to_svg(zones, palette, w, h, show_colors=False, show_numbers=True)
    answer_svg = _pbn_to_svg(zones, palette, w, h, show_colors=True, show_numbers=False)

    # Render PNG
    mystery_png = _pbn_render_png(zone_map, zones, palette, w, h, show_colors=False, show_numbers=True)
    answer_png = _pbn_render_png(zone_map, zones, palette, w, h, show_colors=True, show_numbers=False)
    del zone_map

    # Legend + combined
    legend_png = _make_legend(palette, w)
    mystery_full_png = _stack(mystery_png, legend_png)

    pal_list = [tuple(int(c) for c in row) for row in palette]

    gc.collect()
    return CBNResult(
        mystery_svg=mystery_svg,
        mystery_png=mystery_png,
        answer_svg=answer_svg,
        answer_png=answer_png,
        legend_png=legend_png,
        mystery_full_png=mystery_full_png,
        zone_count=len(zones),
        color_count=colors,
        palette=pal_list,
    )


# ── Smart Color by Number (GPT line art + original colors) ─────────────

def generate_smart_cbn(
    original: Image.Image,
    line_art: Image.Image,
    colors: int = 10,
    page: str = "letter",
    min_zone_pixels: int = 150,
) -> CBNResult:
    """Smart Color-by-Number: GPT line art + color zones from ORIGINAL image.

    Pipeline:
      1. Resize both images to page dimensions
      2. Quantize ORIGINAL image to N colors (KMeans on pixels)
      3. Build zone map from quantized image (connected components of same color)
      4. Mystery = GPT line art + numbers at zone centroids
      5. Answer = quantized original image (clean, vibrant colors)
    """
    from scipy import ndimage

    # Determine target size from page
    if page and page in PAGE_SIZES:
        pw, ph = PAGE_SIZES[page]
        margin = 50
        tw, th = pw - 2 * margin, ph - 2 * margin
    else:
        tw, th = 2400, 3000

    # Resize both to fit page
    s = min(tw / original.size[0], th / original.size[1])
    new_w, new_h = int(original.size[0] * s), int(original.size[1] * s)

    orig = original.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
    orig_arr = np.array(orig, dtype=np.float32)

    la_resized = line_art.convert("RGB").resize((new_w, new_h), Image.LANCZOS)

    # ── Step 1: Quantize original image to N colors ──
    # Subsample pixels for KMeans (performance)
    pixels = orig_arr.reshape(-1, 3)
    n_px = len(pixels)
    sample_size = min(n_px, 50000)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_px, sample_size, replace=False)
    sample_pixels = pixels[sample_idx]

    n_clusters = min(colors, 20)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=5, batch_size=1024)
    km.fit(sample_pixels)
    palette = km.cluster_centers_.astype(np.uint8)

    # Assign every pixel to nearest cluster
    pixel_labels = km.predict(pixels).reshape(new_h, new_w)

    # Build quantized image (answer key)
    quantized_arr = palette[pixel_labels]  # shape (h, w, 3)

    # ── Step 2: Find zones (connected components of same color label) ──
    valid_zones = []
    for color_id in range(n_clusters):
        color_mask = (pixel_labels == color_id).astype(np.uint8)
        labeled, n_feat = ndimage.label(color_mask)
        slices = ndimage.find_objects(labeled)
        for rid, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            crop = labeled[sl]
            mask = (crop == rid)
            area = int(mask.sum())
            if area < min_zone_pixels:
                continue
            # Centroid
            ys, xs = np.where(mask)
            cy = int(ys.mean()) + sl[0].start
            cx = int(xs.mean()) + sl[1].start
            valid_zones.append({
                "color_id": color_id,
                "area": area,
                "centroid": (cy, cx),
            })

    if not valid_zones:
        legend_png = Image.new("RGB", (new_w, 60), (255, 255, 255))
        full = _stack(la_resized, legend_png)
        return CBNResult(
            mystery_svg="", mystery_png=la_resized, answer_svg="", answer_png=la_resized,
            legend_png=legend_png, mystery_full_png=full,
            zone_count=0, color_count=0, palette=[],
        )

    # ── Step 3: Render Mystery = GPT line art + numbers ──
    mystery_img = la_resized.copy()
    draw_m = ImageDraw.Draw(mystery_img)
    for vz in valid_zones:
        num = str(vz["color_id"] + 1)
        cy, cx = vz["centroid"]
        fsize = max(14, min(32, int(math.sqrt(vz["area"]) / 5)))
        font = _get_font(fsize)
        bbox = font.getbbox(num)
        tw2, th2 = bbox[2] - bbox[0], bbox[3] - bbox[1]
        r = max(tw2, th2) // 2 + 4
        draw_m.ellipse([cx - r, cy - r, cx + r, cy + r], fill="white", outline="black", width=2)
        draw_m.text((cx - tw2 // 2, cy - th2 // 2), num, fill="black", font=font)

    # ── Step 4: Render Answer = quantized original + line art overlay ──
    answer_arr = quantized_arr.copy()
    # Overlay GPT line art lines (dark pixels only)
    la_gray = np.array(line_art.convert("L").resize((new_w, new_h), Image.LANCZOS))
    line_mask = la_gray < 80  # Only the darkest lines
    answer_arr[line_mask] = 0
    answer_img = Image.fromarray(answer_arr)

    del pixel_labels, quantized_arr, answer_arr

    # ── Legend + combined ──
    legend_png = _make_legend(palette, new_w)
    mystery_full_png = _stack(mystery_img, legend_png)

    pal_list = [tuple(int(c) for c in row) for row in palette]

    gc.collect()
    return CBNResult(
        mystery_svg="",
        mystery_png=mystery_img,
        answer_svg="",
        answer_png=answer_img,
        legend_png=legend_png,
        mystery_full_png=mystery_full_png,
        zone_count=len(valid_zones),
        color_count=n_clusters,
        palette=pal_list,
    )


# ── Paint by Number helpers (Potrace) ──────────────────────────────────

def _pbn_trace_zone(
    zone_mask_crop: np.ndarray,
    offset_x: int,
    offset_y: int,
    turdsize: int = 2,
    opticurve: bool = True,
    opttolerance: float = 0.2,
) -> list:
    """Trace a single zone's binary mask crop with Potrace.

    Returns list of {"d": str} dicts where "d" is an SVG path data string
    with M/C/L/z commands (cubic Bezier curves from Potrace).
    """
    from potrace import Bitmap  # lazy import (potracer package)

    # Potrace expects a PIL Image; blacklevel=0.5 thresholds at 128
    pil_mask = Image.fromarray(zone_mask_crop, mode="L")
    bm = Bitmap(pil_mask, blacklevel=0.5)
    plist = bm.trace(
        turdsize=turdsize,
        opticurve=opticurve,
        opttolerance=opttolerance,
    )

    paths = []
    for curve in plist:
        sp = curve.start_point
        sx = sp.x + offset_x
        sy = sp.y + offset_y
        d_parts = [f"M{sx:.1f},{sy:.1f}"]

        for seg in curve.segments:
            if seg.is_corner:
                a = seg.c
                b = seg.end_point
                d_parts.append(
                    f"L{a.x + offset_x:.1f},{a.y + offset_y:.1f} "
                    f"L{b.x + offset_x:.1f},{b.y + offset_y:.1f}"
                )
            else:
                c1 = seg.c1
                c2 = seg.c2
                ep = seg.end_point
                d_parts.append(
                    f"C{c1.x + offset_x:.1f},{c1.y + offset_y:.1f} "
                    f"{c2.x + offset_x:.1f},{c2.y + offset_y:.1f} "
                    f"{ep.x + offset_x:.1f},{ep.y + offset_y:.1f}"
                )

        d_parts.append("z")
        paths.append({"d": " ".join(d_parts)})

    del bm, plist
    return paths


def _pbn_extract_paths(
    zone_map: np.ndarray,
    zones: list,
    turdsize: int = 2,
    opticurve: bool = True,
    opttolerance: float = 0.2,
) -> list:
    """Extract Potrace Bezier paths for each zone.

    Adds "paths" key to each zone dict: list of {"d": str} SVG path strings.
    Uses Potrace for smooth Bezier curve extraction.
    """
    from scipy import ndimage  # lazy import

    max_id = max(z["id"] for z in zones) if zones else 0
    slices = ndimage.find_objects(zone_map.clip(0, max_id).astype(np.int32))

    for zone in zones:
        zid = zone["id"]
        sl = slices[zid - 1] if zid - 1 < len(slices) else None

        if sl is None:
            zone["paths"] = []
            continue

        # Binary mask crop for this zone (0/255)
        crop = (zone_map[sl] == zid).astype(np.uint8) * 255
        oy, ox = sl[0].start, sl[1].start

        zone["paths"] = _pbn_trace_zone(
            crop, ox, oy, turdsize, opticurve, opttolerance
        )
        del crop

    return zones


def _pbn_to_svg(
    zones: list,
    palette: np.ndarray,
    w: int,
    h: int,
    show_colors: bool = False,
    show_numbers: bool = True,
) -> str:
    """Render PBN zones as SVG with smooth Bezier curves from Potrace.

    Uses fill-rule="evenodd" for proper hole rendering.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n',
        '<rect width="100%" height="100%" fill="white"/>\n',
        '<style>\n',
        '  .zone { stroke: #222; stroke-width: 1.2; stroke-linejoin: round; stroke-linecap: round; }\n',
    ]
    if show_numbers:
        parts.append(
            '  .num { font-family: Arial, Helvetica, sans-serif; '
            'font-weight: bold; text-anchor: middle; '
            'dominant-baseline: central; fill: #333; }\n'
        )
    parts.append('</style>\n')

    # Draw zones with Bezier paths
    for zone in zones:
        ci = zone["color"]
        r, g, b = int(palette[ci][0]), int(palette[ci][1]), int(palette[ci][2])
        fill = f"rgb({r},{g},{b})" if show_colors else "white"

        d_combined = " ".join(p["d"] for p in zone["paths"])
        if d_combined:
            parts.append(
                f'<path class="zone" fill-rule="evenodd" '
                f'd="{d_combined}" fill="{fill}"/>\n'
            )

    # Numbers layer (on top)
    if show_numbers:
        for zone in zones:
            ci = zone["color"]
            cy, cx = zone["centroid"]
            num = str(ci + 1)
            area = zone["area"]

            fs = max(8, min(28, int(14 * (area / 3000) ** 0.25)))
            r_bg = fs * 0.55 * max(1.0, len(num) * 0.6)

            parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r_bg:.0f}" '
                f'fill="white" fill-opacity="0.85" stroke="none"/>\n'
            )
            parts.append(
                f'<text class="num" x="{cx}" y="{cy}" '
                f'font-size="{fs}">{num}</text>\n'
            )

    parts.append('</svg>')
    return ''.join(parts)


def _pbn_render_png(
    zone_map: np.ndarray,
    zones: list,
    palette: np.ndarray,
    w: int,
    h: int,
    show_colors: bool = False,
    show_numbers: bool = True,
) -> Image.Image:
    """Render PBN zones to PNG using numpy (no OpenCV).

    Colors via direct zone_map indexing; outlines via morphological erosion.
    """
    from scipy import ndimage  # lazy import

    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    struct = np.ones((3, 3), dtype=bool)

    # Paint all zones
    for zone in zones:
        ci = zone["color"]
        mask = (zone_map == zone["id"])
        if show_colors:
            arr[mask] = palette[ci]
        # Outline: erode mask, XOR to get boundary pixels
        eroded = ndimage.binary_erosion(mask, structure=struct)
        boundary = mask & ~eroded
        arr[boundary] = [30, 30, 30]

    img = Image.fromarray(arr, "RGB")
    del arr

    # Draw numbers with Pillow
    if show_numbers:
        draw = ImageDraw.Draw(img)
        for zone in zones:
            ci = zone["color"]
            cy, cx = zone["centroid"]
            num = str(ci + 1)
            area = zone["area"]

            fs = max(8, min(28, int(14 * (area / 3000) ** 0.25)))
            font = _get_font(fs)
            bb = draw.textbbox((0, 0), num, font=font)
            tw_txt, th_txt = bb[2] - bb[0], bb[3] - bb[1]

            tx = cx - tw_txt // 2
            ty = cy - th_txt // 2

            pad = 3
            draw.ellipse(
                [tx - pad, ty - pad, tx + tw_txt + pad, ty + th_txt + pad],
                fill=(255, 255, 255), outline=(200, 200, 200),
            )
            draw.text((tx, ty), num, fill=(60, 60, 60), font=font)

    return img


def generate_pbn(
    img: Image.Image,
    colors: int = 10,
    page: str = "letter",
    landscape: bool = False,
    crop: Optional[str] = None,
    blur: int = 2,
    min_zone_pixels: int = 500,
    turdsize: int = 2,
    opticurve: bool = True,
    opttolerance: float = 0.2,
) -> PBNResult:
    """Generate a Paint-by-Numbers page using Potrace for smooth Bezier curves.

    Pipeline: preprocess → quantize → find zones → Potrace trace →
              render SVG + PNG → legend → return PBNResult.
    """
    img = img.convert("RGB")

    # Crop
    if crop:
        parts = [float(x) for x in crop.split(",")]
        if len(parts) == 4:
            lp, tp, rp, bp = parts
            w, h = img.size
            img = img.crop((
                int(w * lp / 100), int(h * tp / 100),
                int(w * (100 - rp) / 100), int(h * (100 - bp) / 100),
            ))

    # Blur (smooths noise → cleaner zones)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    # Flatten background: replace dominant border color with white
    img = _flatten_background(img)

    # Fit to page
    if page and page in PAGE_SIZES:
        pw, ph = PAGE_SIZES[page]
        if landscape:
            pw, ph = ph, pw
        margin = 50
        tw, th = pw - 2 * margin, ph - 2 * margin
        s = min(tw / img.size[0], th / img.size[1])
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.LANCZOS)

    w, h = img.size

    # Quantize
    label_map, palette = _quantize(img, colors)

    # Find connected zones
    zone_map, zones = _cbn_find_zones(label_map, colors, min_zone_pixels)
    del label_map

    # Extract Potrace Bezier paths
    zones = _pbn_extract_paths(zone_map, zones, turdsize, opticurve, opttolerance)

    # Render SVG
    mystery_svg = _pbn_to_svg(zones, palette, w, h, show_colors=False, show_numbers=True)
    answer_svg = _pbn_to_svg(zones, palette, w, h, show_colors=True, show_numbers=False)

    # Render PNG (from zone_map, no OpenCV)
    mystery_png = _pbn_render_png(zone_map, zones, palette, w, h, show_colors=False, show_numbers=True)
    answer_png = _pbn_render_png(zone_map, zones, palette, w, h, show_colors=True, show_numbers=False)
    del zone_map

    # Legend + combined
    legend_png = _make_legend(palette, w)
    mystery_full_png = _stack(mystery_png, legend_png)

    pal_list = [tuple(int(c) for c in row) for row in palette]

    gc.collect()
    return PBNResult(
        mystery_svg=mystery_svg,
        mystery_png=mystery_png,
        answer_svg=answer_svg,
        answer_png=answer_png,
        legend_png=legend_png,
        mystery_full_png=mystery_full_png,
        zone_count=len(zones),
        color_count=colors,
        palette=pal_list,
    )


# ── Line Art helpers ────────────────────────────────────────────────────

# Line art presets: (canny_lo, canny_hi, bilateral_d, sigma_color, sigma_space, passes)
# Uses Canny + large morphological close (5x5) to merge double edges cleanly.
_LINEART_PRESETS = {
    # (canny_lo, canny_hi, median_size, smooth_passes, gauss_sigma)
    "simple":   (55, 130, 9, 3, 2.0),   # Kids 3-6: smooth, bold outlines
    "standard": (40, 100, 7, 2, 1.5),   # General: clean outlines
    "detailed": (25,  75, 5, 1, 1.2),   # Teens/adults: more detail
    "expert":   (15,  50, 3, 1, 1.0),   # Maximum detail
}


def _lineart_edges_to_svg(
    edges: np.ndarray,
    turdsize: int = 5,
) -> Tuple[str, int]:
    """Convert a binary edge image to SVG paths using Potrace.

    Traces the edge bitmap into smooth Bezier curves.
    Returns (svg_string, line_count).
    """
    from potrace import Bitmap  # lazy import

    h, w = edges.shape
    pil_edges = Image.fromarray(edges, mode="L")
    bm = Bitmap(pil_edges, blacklevel=0.5)
    plist = bm.trace(turdsize=turdsize, opticurve=True, opttolerance=0.2)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n',
        '<rect width="100%" height="100%" fill="white"/>\n',
    ]

    line_count = 0
    for curve in plist:
        sp = curve.start_point
        d_parts = [f"M{sp.x:.1f},{sp.y:.1f}"]

        for seg in curve.segments:
            if seg.is_corner:
                a = seg.c
                b = seg.end_point
                d_parts.append(
                    f"L{a.x:.1f},{a.y:.1f} L{b.x:.1f},{b.y:.1f}"
                )
            else:
                c1 = seg.c1
                c2 = seg.c2
                ep = seg.end_point
                d_parts.append(
                    f"C{c1.x:.1f},{c1.y:.1f} "
                    f"{c2.x:.1f},{c2.y:.1f} "
                    f"{ep.x:.1f},{ep.y:.1f}"
                )

        d_parts.append("z")
        d = " ".join(d_parts)
        parts.append(
            f'<path d="{d}" fill="none" stroke="#000" '
            f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>\n'
        )
        line_count += 1

    del bm, plist
    parts.append('</svg>')
    return ''.join(parts), line_count


def _lineart_edges_to_png(edges: np.ndarray, w: int, h: int) -> Image.Image:
    """Convert binary edge image (white edges on black) to PNG (black lines on white)."""
    # Invert: edges are 255 (white), we want black lines on white
    inverted = 255 - edges
    return Image.fromarray(inverted).convert("RGB")


def _canny_scipy(
    gray: np.ndarray,
    low: int,
    high: int,
    sigma: float = 1.4,
) -> np.ndarray:
    """Canny-like edge detection using scipy (no OpenCV).

    Gaussian blur → Sobel gradients → non-maximum suppression →
    hysteresis via connected components.
    Returns binary edge image (0/255 uint8).
    """
    from scipy.ndimage import gaussian_filter, sobel, label

    smoothed = gaussian_filter(gray.astype(np.float64), sigma=sigma)

    gx = sobel(smoothed, axis=1)
    gy = sobel(smoothed, axis=0)
    mag = np.hypot(gx, gy)

    # Normalize using 99th percentile (robust to outlier gradients)
    p99 = np.percentile(mag, 99)
    if p99 > 0:
        mag = np.clip(mag * 255.0 / p99, 0, 255)

    # Non-maximum suppression (fully vectorized)
    angle = np.degrees(np.arctan2(gy, gx)) % 180
    h, w = mag.shape
    mag_pad = np.pad(mag, 1, mode="constant")
    nms = np.zeros((h, w), dtype=np.float64)

    for mask_fn, dy1, dx1, dy2, dx2 in [
        (lambda a: (a < 22.5) | (a >= 157.5),  0, -1,  0,  1),
        (lambda a: (a >= 22.5) & (a < 67.5),  -1,  1,  1, -1),
        (lambda a: (a >= 67.5) & (a < 112.5), -1,  0,  1,  0),
        (lambda a: (a >= 112.5) & (a < 157.5),-1, -1,  1,  1),
    ]:
        mask = mask_fn(angle)
        n1 = mag_pad[1 + dy1:h + 1 + dy1, 1 + dx1:w + 1 + dx1]
        n2 = mag_pad[1 + dy2:h + 1 + dy2, 1 + dx2:w + 1 + dx2]
        keep = mask & (mag >= n1) & (mag >= n2)
        nms[keep] = mag[keep]

    # Hysteresis via connected-component labelling
    strong = nms >= high
    weak = (nms >= low) & ~strong
    combined = strong | weak
    labeled, _ = label(combined)
    # Keep every connected region that touches at least one strong pixel
    strong_labels = set(labeled[strong].ravel())
    strong_labels.discard(0)
    edges = np.isin(labeled, list(strong_labels))

    return (edges.astype(np.uint8) * 255)


def _disk_kernel(size: int) -> np.ndarray:
    """Create a disk-shaped structuring element (replaces cv2 MORPH_ELLIPSE)."""
    r = size // 2
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(bool)


def generate_line_art(
    img: Image.Image,
    page: str = "letter",
    landscape: bool = False,
    crop: Optional[str] = None,
    detail_level: str = "standard",
    line_thickness: int = 2,
) -> LineArtResult:
    """Convert a photo to a line-art coloring page.

    Pipeline: preprocess → grayscale → median filter (multi-pass) →
              Canny edge detection (scipy) → morphological close →
              optional dilation → Potrace SVG + PNG.
    """
    from scipy.ndimage import median_filter, binary_closing, binary_dilation

    img = img.convert("RGB")

    # Crop
    if crop:
        parts = [float(x) for x in crop.split(",")]
        if len(parts) == 4:
            lp, tp, rp, bp = parts
            w, h = img.size
            img = img.crop((
                int(w * lp / 100), int(h * tp / 100),
                int(w * (100 - rp) / 100), int(h * (100 - bp) / 100),
            ))

    # Fit to page
    if page and page in PAGE_SIZES:
        pw, ph = PAGE_SIZES[page]
        if landscape:
            pw, ph = ph, pw
        margin = 50
        tw, th = pw - 2 * margin, ph - 2 * margin
        s = min(tw / img.size[0], th / img.size[1])
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.LANCZOS)

    w, h = img.size

    # Grayscale via Pillow
    gray = np.array(img.convert("L"))

    # Load preset
    preset = _LINEART_PRESETS.get(detail_level, _LINEART_PRESETS["standard"])
    canny_lo, canny_hi, median_sz, passes, gauss_sigma = preset

    # Median filter (edge-preserving smoothing)
    for _ in range(passes):
        gray = median_filter(gray, size=median_sz)

    # Canny edge detection (scipy implementation)
    edges = _canny_scipy(gray, canny_lo, canny_hi, sigma=gauss_sigma)

    # Morphological close with 5x5 disk: merges double-edge lines
    kernel_close = _disk_kernel(5)
    edges = (binary_closing(edges > 0, structure=kernel_close) * 255).astype(np.uint8)

    # Line thickness control via dilation
    if line_thickness > 1:
        kernel_dilate = _disk_kernel(line_thickness)
        edges = (binary_dilation(edges > 0, structure=kernel_dilate, iterations=1) * 255).astype(np.uint8)

    # Generate SVG with Potrace (smooth Bezier curves)
    turdsize = {"simple": 15, "standard": 8, "detailed": 4, "expert": 2}.get(detail_level, 8)
    line_art_svg, line_count = _lineart_edges_to_svg(edges, turdsize=turdsize)

    # Generate PNG (black lines on white)
    line_art_png = _lineart_edges_to_png(edges, w, h)

    # Lower-res preview for browser (max 800px wide)
    preview_scale = min(1.0, 800 / w)
    if preview_scale < 1.0:
        pw_prev = int(w * preview_scale)
        ph_prev = int(h * preview_scale)
        preview_png = line_art_png.resize((pw_prev, ph_prev), Image.LANCZOS)
    else:
        preview_png = line_art_png.copy()

    gc.collect()
    return LineArtResult(
        line_art_svg=line_art_svg,
        line_art_png=line_art_png,
        preview_png=preview_png,
        detail_level=detail_level,
        line_count=line_count,
    )


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

    gc.collect()
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

    gc.collect()
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


def generate_from_preset(img: Image.Image, preset_name: str):
    """Convenience: generate using a named preset. Returns MosaicResult, CBNResult, or LineArtResult."""
    p = PRESETS[preset_name]
    if p.mode == "lineart":
        return generate_line_art(img, page=p.page, landscape=p.landscape,
                                 crop=p.crop, detail_level=p.lineart_detail,
                                 line_thickness=p.lineart_thickness)
    if p.mode == "cbn":
        return generate_cbn(img, colors=p.colors, page=p.page,
                            landscape=p.landscape, crop=p.crop, blur=p.blur,
                            min_zone_pixels=p.min_zone_pixels,
                            epsilon_factor=p.epsilon_factor)
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

def images_to_zip(results: list, include_answer: bool = True,
                   plan: str = "pro") -> bytes:
    """Pack multiple results into a ZIP (in memory).

    results = [(name, MosaicResult | CBNResult | PBNResult | LineArtResult), ...]

    Plan-based tiering:
      - "free" / "creator": mystery-full PNG only (CBN/PBN/Mosaic),
                            lineart PNG only (LineArt)
      - "pro" / "studio":  full pack with SVG + all PNGs
    """
    include_svg = plan in ("pro", "studio")
    full_pack = plan in ("pro", "studio")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, res in results:
            if isinstance(res, LineArtResult):
                if include_svg:
                    zf.writestr(f"{name}/{name}-lineart.svg", res.line_art_svg)
                img_buf = io.BytesIO()
                res.line_art_png.save(img_buf, format="PNG", dpi=(300, 300))
                zf.writestr(f"{name}/{name}-lineart.png", img_buf.getvalue())
            elif isinstance(res, (CBNResult, PBNResult)):
                if full_pack:
                    # SVG files (pro/studio only)
                    zf.writestr(f"{name}/{name}-mystery.svg", res.mystery_svg)
                    if include_answer:
                        zf.writestr(f"{name}/{name}-answer.svg", res.answer_svg)
                    # All PNG files
                    for suffix, img in [
                        ("mystery", res.mystery_png),
                        ("mystery-full", res.mystery_full_png),
                        ("legend", res.legend_png),
                    ]:
                        img_buf = io.BytesIO()
                        img.save(img_buf, format="PNG", dpi=(300, 300))
                        zf.writestr(f"{name}/{name}-{suffix}.png", img_buf.getvalue())
                    if include_answer:
                        img_buf = io.BytesIO()
                        res.answer_png.save(img_buf, format="PNG", dpi=(300, 300))
                        zf.writestr(f"{name}/{name}-answer.png", img_buf.getvalue())
                else:
                    # Free/Creator: mystery-full PNG only
                    img_buf = io.BytesIO()
                    res.mystery_full_png.save(img_buf, format="PNG", dpi=(300, 300))
                    zf.writestr(f"{name}/{name}-mystery-full.png", img_buf.getvalue())
            else:
                # MosaicResult (hex/voronoi)
                if full_pack:
                    items = [
                        ("mystery", res.mystery),
                        ("mystery-full", res.mystery_full),
                        ("beauty", res.beauty),
                        ("legend", res.legend),
                    ]
                    if include_answer:
                        items.insert(2, ("answer", res.answer))
                    for suffix, img in items:
                        img_buf = io.BytesIO()
                        img.save(img_buf, format="PNG", dpi=(300, 300))
                        zf.writestr(f"{name}/{name}-{suffix}.png", img_buf.getvalue())
                else:
                    # Free/Creator: mystery-full PNG only
                    img_buf = io.BytesIO()
                    res.mystery_full.save(img_buf, format="PNG", dpi=(300, 300))
                    zf.writestr(f"{name}/{name}-mystery-full.png", img_buf.getvalue())
    buf.seek(0)
    return buf.getvalue()
