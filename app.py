"""
Mystery Mosaic API
==================
FastAPI backend for generating mystery mosaics.

Endpoints:
    POST /api/generate   — single image → PNG files
    POST /api/bulk       — multiple images → ZIP
    GET  /api/presets    — list available presets
    GET  /api/quota      — check remaining free uses
"""

import io
import os
import time
import hashlib
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from mosaic_engine import (
    generate, generate_from_preset, images_to_zip,
    PRESETS, MosaicResult,
)

# ── Config ───────────────────────────────────────────────────────────────

FREE_LIMIT = 3                # free generations per user
MAX_BULK = 50                 # max images in one bulk request
MAX_IMAGE_SIZE = 10_000_000   # 10 MB per image
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}
PROMO_CODES = {
    os.environ.get("PROMO_UNLIMITED", "COCO-ADMIN-2026"): {"limit": 999999, "label": "unlimited"},
}

# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mystery Mosaic API",
    version="1.0.0",
    description="Generate mystery mosaic color-by-number pages from any image.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cocotheaxolotl.org",
        "https://www.cocotheaxolotl.org",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Simple in-memory stores (replace with Redis/DB in production) ──

_usage: dict[str, dict] = {}   # ip → {"count": int, "first": timestamp}
_total_generated: int = 0       # global generation counter


def _user_key(request: Request) -> str:
    """Identify user by IP (+ forwarded header for proxies)."""
    forwarded = request.headers.get("x-forwarded-for")
    ip = forwarded.split(",")[0].strip() if forwarded else request.client.host
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def _check_promo(code: Optional[str]) -> Optional[dict]:
    """Check if a promo code is valid. Returns promo info or None."""
    if code and code.strip() in PROMO_CODES:
        return PROMO_CODES[code.strip()]
    return None


def _check_quota(request: Request, promo: Optional[dict] = None) -> int:
    """Return remaining free uses.  Raises 429 if exhausted."""
    if promo:
        return promo["limit"]

    key = _user_key(request)
    now = time.time()
    rec = _usage.get(key)

    if rec is None:
        _usage[key] = {"count": 0, "first": now}
        return FREE_LIMIT

    # Reset monthly
    if now - rec["first"] > 30 * 86400:
        _usage[key] = {"count": 0, "first": now}
        return FREE_LIMIT

    remaining = FREE_LIMIT - rec["count"]
    return max(0, remaining)


_last_gen: dict[str, float] = {}   # ip_key → last generation timestamp

def _consume(request: Request, n: int = 1):
    """Consume n quota units and increment global counter.
    Dedup: if same IP generated within last 60s, don't count again.
    """
    global _total_generated
    key = _user_key(request)
    now = time.time()

    # Dedup: don't double-count rapid successive calls (preview + zip)
    if key in _last_gen and now - _last_gen[key] < 60:
        _last_gen[key] = now
        return

    _last_gen[key] = now
    if key not in _usage:
        _usage[key] = {"count": 0, "first": now}
    _usage[key]["count"] += n
    _total_generated += n


# ── Helpers ──────────────────────────────────────────────────────────────

def _read_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _img_to_streaming(img: Image.Image, filename: str) -> StreamingResponse:
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(300, 300))
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _result_to_zip_stream(name: str, res: MosaicResult) -> StreamingResponse:
    data = images_to_zip([(name, res)])
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}-mosaic.zip"'},
    )


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/presets")
def list_presets():
    """List all available generation presets."""
    return {
        name: {
            "label": p.label,
            "page": p.page,
            "cell_size": p.cell_size,
            "colors": p.colors,
            "landscape": p.landscape,
        }
        for name, p in PRESETS.items()
    }


@app.get("/api/quota")
def check_quota(request: Request):
    """Check how many free generations remain."""
    remaining = _check_quota(request)
    return {"remaining": remaining, "limit": FREE_LIMIT}


@app.post("/api/generate")
async def generate_mosaic(
    request: Request,
    image: UploadFile = File(...),
    preset: Optional[str] = Form(None),
    colors: int = Form(12),
    cell_size: int = Form(25),
    page: str = Form("letter"),
    output: str = Form("zip"),   # "zip" | "mystery" | "answer" | "legend" | "full"
    promo_code: Optional[str] = Form(None),
):
    """Generate a mystery mosaic from a single uploaded image.

    - Use `preset` for quick configuration (overrides colors/cell_size/page).
    - `output` controls what you get back:
      - "zip"     → ZIP with all 4 files (default)
      - "mystery" → just the mystery page PNG
      - "answer"  → just the answer key PNG
      - "legend"  → just the legend PNG
      - "full"    → mystery + legend combined PNG
    """
    # Quota check
    promo = _check_promo(promo_code)
    remaining = _check_quota(request, promo)
    if remaining <= 0:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Free limit reached",
                "message": f"You've used all {FREE_LIMIT} free generations. Upgrade to Pro for unlimited access!",
                "upgrade_url": "https://cocotheaxolotl.org/mosaic/pricing/",
            },
        )

    # Validate upload
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported image type: {image.content_type}")

    data = await image.read()
    if len(data) > MAX_IMAGE_SIZE:
        raise HTTPException(400, "Image too large (max 10 MB)")

    img = _read_image(data)
    name = Path(image.filename).stem if image.filename else "mosaic"

    # Generate
    if preset and preset in PRESETS:
        result = generate_from_preset(img, preset)
    else:
        colors = max(4, min(20, colors))
        cell_size = max(12, min(50, cell_size))
        result = generate(img, colors=colors, cell_size=cell_size, page=page)

    # Consume quota
    _consume(request, 1)

    # Return
    if output == "mystery":
        return _img_to_streaming(result.mystery, f"{name}-mystery.png")
    if output == "answer":
        return _img_to_streaming(result.answer, f"{name}-answer.png")
    if output == "beauty":
        return _img_to_streaming(result.beauty, f"{name}-beauty.png")
    if output == "legend":
        return _img_to_streaming(result.legend, f"{name}-legend.png")
    if output == "full":
        return _img_to_streaming(result.mystery_full, f"{name}-mystery-full.png")

    # Default: ZIP
    return _result_to_zip_stream(name, result)


@app.post("/api/bulk")
async def bulk_generate(
    request: Request,
    images: List[UploadFile] = File(...),
    preset: Optional[str] = Form(None),
    colors: int = Form(12),
    cell_size: int = Form(25),
    page: str = Form("letter"),
    promo_code: Optional[str] = Form(None),
):
    """Generate mosaics for multiple images at once → returns a ZIP.

    - Max 50 images per request (Pro only beyond 3).
    - Each image counts as 1 quota unit.
    """
    n = len(images)
    if n > MAX_BULK:
        raise HTTPException(400, f"Too many images (max {MAX_BULK})")
    if n == 0:
        raise HTTPException(400, "No images provided")

    # Quota check
    promo = _check_promo(promo_code)
    remaining = _check_quota(request, promo)
    if remaining < n:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Not enough free credits",
                "remaining": remaining,
                "requested": n,
                "message": f"You have {remaining} free generation(s) left but requested {n}. Upgrade to Pro!",
                "upgrade_url": "https://cocotheaxolotl.org/mosaic/pricing/",
            },
        )

    # Process all images
    results = []
    for upload in images:
        if upload.content_type not in ALLOWED_TYPES:
            continue
        data = await upload.read()
        if len(data) > MAX_IMAGE_SIZE:
            continue
        img = _read_image(data)
        name = Path(upload.filename).stem if upload.filename else f"image-{len(results)+1}"

        if preset and preset in PRESETS:
            res = generate_from_preset(img, preset)
        else:
            res = generate(img, colors=max(4, min(20, colors)),
                           cell_size=max(12, min(50, cell_size)), page=page)
        results.append((name, res))

    if not results:
        raise HTTPException(400, "No valid images processed")

    # Consume quota
    _consume(request, len(results))

    # Pack into ZIP
    zip_data = images_to_zip(results)
    return StreamingResponse(
        io.BytesIO(zip_data),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="mystery-mosaics-bulk.zip"'},
    )


# ── Health ───────────────────────────────────────────────────────────────

@app.get("/api/stats")
def stats():
    """Public stats for social proof on the landing page."""
    return {"total_generated": _total_generated}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
