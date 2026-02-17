"""
Mystery Mosaic API
==================
FastAPI backend for generating mystery mosaics.

Endpoints:
    POST /api/generate          — single image → PNG files
    POST /api/bulk              — multiple images → ZIP
    POST /api/request-download  — email-gated download (generates + sends link)
    GET  /api/download/{token}  — serve stored ZIP from email link
    GET  /api/presets           — list available presets
    GET  /api/quota             — check remaining free uses
"""

import io
import gc
import os
import re
import json
import time
import uuid
import hashlib
import urllib.request
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from mosaic_engine import (
    generate, generate_voronoi, generate_from_preset, images_to_zip,
    PRESETS, VORONOI_DENSITIES, MosaicResult,
)

# ── Config ───────────────────────────────────────────────────────────────

FREE_LIMIT = 3                # free generations per user
MAX_BULK = 50                 # max images in one bulk request
MAX_IMAGE_SIZE = 10_000_000   # 10 MB per image
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}
PROMO_CODES = {
    os.environ.get("PROMO_UNLIMITED", "COCO-ADMIN-2026"): {"limit": 999999, "label": "unlimited"},
}
BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "noreply@univers.studio")
SENDER_NAME = os.environ.get("SENDER_NAME", "Univers Studio")
API_PUBLIC_URL = os.environ.get("API_PUBLIC_URL", "https://mosaic-api-y18j.onrender.com")
DOWNLOAD_TTL = 3600        # download links expire after 1 hour
DOWNLOAD_MAX_ENTRIES = 15   # max ZIPs stored in memory

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
        "https://univers.studio",
        "https://www.univers.studio",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
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
_download_store: dict[str, dict] = {}  # token → {zip_bytes, filename, created_at}

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
            "mode": p.mode,
            "voronoi_density": p.voronoi_density if p.mode == "voronoi" else None,
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
    output: str = Form("zip"),   # "zip" | "mystery" | "answer" | "legend" | "full" | "beauty"
    mode: str = Form("hex"),     # "hex" | "voronoi"
    density: str = Form("standard"),  # voronoi density: easy/standard/detailed/expert
    promo_code: Optional[str] = Form(None),
):
    """Generate a mystery mosaic from a single uploaded image.

    - Use `preset` for quick configuration (overrides all other params).
    - `mode` = "hex" (default grid) or "voronoi" (organic cells).
    - `density` = voronoi density preset: easy/standard/detailed/expert.
    - `output` controls what you get back:
      - "zip"     → ZIP with all files (default)
      - "mystery" → just the mystery page PNG
      - "answer"  → just the answer key PNG
      - "beauty"  → beauty preview PNG
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
    elif mode == "voronoi":
        colors = max(4, min(20, colors))
        density_map = VORONOI_DENSITIES.get(page, VORONOI_DENSITIES["letter"])
        num_cells = density_map.get(density, density_map["standard"])
        result = generate_voronoi(img, colors=colors, density=num_cells, page=page)
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
    mode: str = Form("hex"),
    density: str = Form("standard"),
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
        elif mode == "voronoi":
            density_map = VORONOI_DENSITIES.get(page, VORONOI_DENSITIES["letter"])
            num_cells = density_map.get(density, density_map["standard"])
            res = generate_voronoi(img, colors=max(4, min(20, colors)),
                                   density=num_cells, page=page)
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


# ── Download store helpers ───────────────────────────────────────────────

def _cleanup_downloads():
    """Remove expired entries from the download store."""
    now = time.time()
    expired = [k for k, v in _download_store.items() if now - v["created_at"] > DOWNLOAD_TTL]
    for k in expired:
        del _download_store[k]


def _store_download(zip_bytes: bytes, filename: str) -> str:
    """Store a ZIP in memory and return a unique token."""
    _cleanup_downloads()
    # Evict oldest if at capacity
    while len(_download_store) >= DOWNLOAD_MAX_ENTRIES:
        oldest = min(_download_store, key=lambda k: _download_store[k]["created_at"])
        del _download_store[oldest]
    token = uuid.uuid4().hex
    _download_store[token] = {
        "zip_bytes": zip_bytes,
        "filename": filename,
        "created_at": time.time(),
    }
    return token


def _send_brevo_email(to_email: str, download_url: str):
    """Send a transactional email via Brevo API with the download link."""
    if not BREVO_API_KEY:
        raise RuntimeError("BREVO_API_KEY not configured")

    html = f"""\
<div style="font-family:Inter,Helvetica,Arial,sans-serif;max-width:520px;margin:0 auto;padding:32px 20px">
  <h1 style="color:#1a1a2e;font-size:1.5em;margin:0 0 8px">Your Mystery Mosaic is ready!</h1>
  <p style="color:#6b7280;line-height:1.6;margin:0 0 24px">
    Your mosaic has been generated. Click the button below to download your full pack (ZIP)
    — includes the mystery page, answer key, beauty preview, and color legend at 300&nbsp;DPI.
  </p>
  <a href="{download_url}"
     style="display:inline-block;padding:14px 36px;background:#7c3aed;color:#fff;
            text-decoration:none;border-radius:10px;font-weight:700;font-size:1.05em">
    Download My Mosaic
  </a>
  <p style="color:#9ca3af;font-size:.85em;margin:24px 0 0;line-height:1.5">
    This link expires in 1 hour and can only be used once.<br>
    If the link has expired, simply generate a new mosaic on
    <a href="https://univers.studio/mosaic/" style="color:#7c3aed">univers.studio</a>.
  </p>
  <hr style="border:none;border-top:1px solid #e5e7eb;margin:28px 0 16px">
  <p style="color:#9ca3af;font-size:.78em;margin:0">
    Univers Studio — AI creative tools for publishers &amp; creators<br>
    <a href="https://univers.studio" style="color:#7c3aed">univers.studio</a>
  </p>
</div>"""

    payload = json.dumps({
        "sender": {"name": SENDER_NAME, "email": SENDER_EMAIL},
        "to": [{"email": to_email}],
        "subject": "Your Mystery Mosaic is ready!",
        "htmlContent": html,
    }).encode()

    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=payload,
        headers={
            "api-key": BREVO_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    urllib.request.urlopen(req, timeout=10)


def _add_brevo_contact(email: str):
    """Add email to the Brevo contact list (silently ignore errors)."""
    if not BREVO_API_KEY:
        return
    try:
        payload = json.dumps({
            "email": email,
            "updateEnabled": True,
        }).encode()
        req = urllib.request.Request(
            "https://api.brevo.com/v3/contacts",
            data=payload,
            headers={
                "api-key": BREVO_API_KEY,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # contact may already exist


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ── Email-gated download ────────────────────────────────────────────────

@app.post("/api/request-download")
async def request_download(
    request: Request,
    email: str = Form(...),
    image: UploadFile = File(...),
    preset: Optional[str] = Form(None),
    colors: int = Form(12),
    cell_size: int = Form(25),
    page: str = Form("letter"),
    mode: str = Form("hex"),
    density: str = Form("standard"),
    promo_code: Optional[str] = Form(None),
):
    """Generate a mosaic, store the ZIP, and email the download link."""
    # Validate email
    if not _EMAIL_RE.match(email):
        raise HTTPException(400, "Invalid email address")

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
    elif mode == "voronoi":
        colors = max(4, min(20, colors))
        density_map = VORONOI_DENSITIES.get(page, VORONOI_DENSITIES["letter"])
        num_cells = density_map.get(density, density_map["standard"])
        result = generate_voronoi(img, colors=colors, density=num_cells, page=page)
    else:
        colors = max(4, min(20, colors))
        cell_size = max(12, min(50, cell_size))
        result = generate(img, colors=colors, cell_size=cell_size, page=page)

    # Pack into ZIP
    zip_data = images_to_zip([(name, result)])
    filename = f"{name}-mosaic.zip"
    gc.collect()

    # Store and get token
    token = _store_download(zip_data, filename)
    download_url = f"{API_PUBLIC_URL}/api/download/{token}"

    # Send email + add to contacts
    try:
        _send_brevo_email(email, download_url)
    except Exception as e:
        # Remove stored ZIP if email fails
        _download_store.pop(token, None)
        raise HTTPException(500, f"Failed to send email: {e}")

    _add_brevo_contact(email)

    return {"ok": True, "message": "Download link sent to your email!"}


@app.get("/api/download/{token}")
def download_mosaic(token: str):
    """Serve a stored ZIP file from an email download link (one-time use)."""
    _cleanup_downloads()
    entry = _download_store.pop(token, None)
    if not entry:
        return HTMLResponse(
            content="""\
<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Link Expired — Univers Studio</title>
<style>body{font-family:Inter,Helvetica,sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;background:#fafafa;color:#1a1a2e;text-align:center;padding:20px}
.box{max-width:400px}.emoji{font-size:3em;margin:0 0 16px}h1{font-size:1.3em;margin:0 0 8px}p{color:#6b7280;line-height:1.6;margin:0 0 20px;font-size:.95em}
a{display:inline-block;padding:12px 28px;background:#7c3aed;color:#fff;text-decoration:none;border-radius:10px;font-weight:700}</style></head>
<body><div class="box"><div class="emoji">⏰</div><h1>This link has expired</h1>
<p>Download links are valid for 1 hour and can only be used once. Generate a new mosaic to get a fresh link.</p>
<a href="https://univers.studio/mosaic/">Generate New Mosaic</a></div></body></html>""",
            status_code=410,
        )
    return StreamingResponse(
        io.BytesIO(entry["zip_bytes"]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{entry["filename"]}"'},
    )


# ── Health ───────────────────────────────────────────────────────────────

@app.get("/api/stats")
def stats():
    """Public stats for social proof on the landing page."""
    return {"total_generated": _total_generated}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
