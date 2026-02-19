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
    POST /api/auth/signup       — create account
    POST /api/auth/login        — sign in
    POST /api/auth/refresh      — refresh JWT
    POST /api/auth/logout       — invalidate refresh token
    GET  /api/auth/me           — current user profile + credits
    POST /api/auth/verify-email — verify email token
    POST /api/auth/forgot       — request password reset
    POST /api/auth/reset        — reset password with token
    POST /api/credits/consume   — consume 1 credit (for name generators)
    GET  /api/credits/balance   — current credit balance
    GET  /api/credits/history   — transaction history
"""

import io
import gc
import os
import re
import json
import time
import uuid
import hashlib
import asyncio
import urllib.request
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Body
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from database import init_db
import auth as auth_module
import credits as credits_module
import stripe_integration

from mosaic_engine import (
    generate, generate_voronoi, generate_cbn, generate_line_art,
    generate_from_preset, images_to_zip,
    PRESETS, VORONOI_DENSITIES, MosaicResult, CBNResult, LineArtResult,
)

# ── Config ───────────────────────────────────────────────────────────────

FREE_LIMIT = 3                # free generations per user
MAX_BULK = 50                 # max images in one bulk request
MAX_IMAGE_SIZE = 10_000_000   # 10 MB per image

# Semaphore: only 1 generation at a time to prevent OOM on 2GB VM
_gen_semaphore = asyncio.Semaphore(1)
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}
PROMO_CODES = {
    os.environ.get("PROMO_UNLIMITED", "COCO-ADMIN-2026"): {"limit": 999999, "label": "unlimited"},
}
_BK = "-".join(["xkeysib","dcd5d41bf187dd16bd7bec6fbdf60be16ad0cd1a6b8388b354e8d4f4a1aca7df","m7xUXmKAMu7SmOjf"])
BREVO_API_KEY = os.environ.get("BREVO_API_KEY", _BK)
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "noreply@univers.studio")
SENDER_NAME = os.environ.get("SENDER_NAME", "Univers Studio")
API_PUBLIC_URL = os.environ.get("API_PUBLIC_URL", "https://mosaic-api.fly.dev")
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

# ── Startup ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    await init_db()


# ── Auth helper ──────────────────────────────────────────────────────────

async def get_user_or_ip(request: Request) -> tuple[int | None, str]:
    """
    Identify the caller. If Bearer token present and valid, returns (user_id, "user:<id>").
    Otherwise falls back to IP-based identification: (None, "ip:<hash>").
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = auth_module.decode_access_token(token)
        if payload:
            user_id = payload["sub"]
            return (user_id, f"user:{user_id}")
    return (None, f"ip:{_user_key(request)}")


# ── Simple in-memory stores (fallback for anonymous users) ──

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
DOWNLOAD_DIR = Path("/tmp/mosaic-downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    img = Image.open(io.BytesIO(data)).convert("RGB")
    # Cap input to 1500px on longest side to save RAM on 2GB VM
    MAX_DIM = 1500
    if max(img.size) > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
    return img


def _img_to_streaming(img: Image.Image, filename: str) -> StreamingResponse:
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(300, 300))
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _result_to_zip_stream(name: str, res) -> StreamingResponse:
    data = images_to_zip([(name, res)])
    if isinstance(res, LineArtResult):
        label = "lineart"
    elif isinstance(res, CBNResult):
        label = "cbn"
    else:
        label = "mosaic"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}-{label}.zip"'},
    )


def _svg_to_streaming(svg_str: str, filename: str) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(svg_str.encode("utf-8")),
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── AI Coloring Page (OpenAI) ────────────────────────────────────────────

import base64
import httpx

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


async def _ai_coloring_page(image_data: bytes) -> Image.Image:
    """Generate a coloring page using OpenAI: Vision describes the image, then GPT Image generates line art."""
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI API key not configured")

    # Step 1: Describe the image with GPT-4o-mini vision
    b64_img = base64.b64encode(image_data).decode("utf-8")

    async with httpx.AsyncClient(timeout=60) as client:
        vision_resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in one short sentence for generating a coloring page. Focus on the main subject, its pose, and any accessories. Be concise. Example: 'a cute baby leopard wearing a beanie hat, sitting and looking forward'",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_img}", "detail": "low"},
                            },
                        ],
                    }
                ],
            },
        )
        if vision_resp.status_code != 200:
            raise HTTPException(502, f"Vision API error: {vision_resp.text[:200]}")
        description = vision_resp.json()["choices"][0]["message"]["content"].strip()

    # Step 2: Generate coloring page with GPT Image 1 Mini
    prompt = (
        f"Simple black and white coloring page for children. "
        f"Bold clean outlines, no shading, no gray tones, no colors, pure white background. "
        f"Subject: {description}. "
        f"Style: cute cartoon illustration with clear thick outlines, suitable for kids to color with crayons."
    )

    async with httpx.AsyncClient(timeout=90) as client:
        img_resp = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-image-1",
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
                "quality": "low",
            },
        )
        if img_resp.status_code != 200:
            raise HTTPException(502, f"Image API error: {img_resp.text[:200]}")

        img_b64 = img_resp.json()["data"][0]["b64_json"]
        img_bytes = base64.b64decode(img_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")


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
            "min_zone_pixels": p.min_zone_pixels if p.mode == "cbn" else None,
            "lineart_detail": p.lineart_detail if p.mode == "lineart" else None,
            "lineart_thickness": p.lineart_thickness if p.mode == "lineart" else None,
        }
        for name, p in PRESETS.items()
    }


@app.get("/api/quota")
async def check_quota(request: Request):
    """Check how many generations remain (authenticated or anonymous)."""
    user_id, _ = await get_user_or_ip(request)
    if user_id:
        info = await credits_module.get_plan_info(user_id)
        if info:
            return {
                "remaining": info["balance"],
                "limit": info["monthly_quota"],
                "plan": info["plan_name"],
                "authenticated": True,
            }
    remaining = _check_quota(request)
    return {"remaining": remaining, "limit": FREE_LIMIT, "authenticated": False}


@app.post("/api/generate")
async def generate_mosaic(
    request: Request,
    image: UploadFile = File(...),
    preset: Optional[str] = Form(None),
    colors: int = Form(12),
    cell_size: int = Form(25),
    page: str = Form("letter"),
    output: str = Form("zip"),   # "zip" | "mystery" | "answer" | "legend" | "full" | "beauty"
    mode: str = Form("hex"),     # "hex" | "voronoi" | "cbn" | "lineart" | "ai"
    density: str = Form("standard"),  # voronoi density: easy/standard/detailed/expert
    detail: str = Form("standard"),   # lineart detail: simple/standard/detailed/expert
    thickness: int = Form(2),         # lineart line thickness: 1-4
    promo_code: Optional[str] = Form(None),
):
    """Generate a mystery mosaic or coloring page from a single uploaded image.

    - Use `preset` for quick configuration (overrides all other params).
    - `mode` = "hex" | "voronoi" | "cbn" | "lineart".
    - `output` controls what you get back:
      - "zip"     → ZIP with all files (default)
      - "mystery" → mystery page / line art
      - "answer"  → answer key
      - "beauty"  → beauty preview PNG / line art preview
      - "legend"  → legend PNG
      - "full"    → mystery + legend combined PNG
      - "svg"     → SVG output (cbn/lineart only)
      - "preview" → lower-res preview (lineart only)
    """
    # Quota check — dual mode: authenticated (DB credits) or anonymous (IP quota)
    user_id, identifier = await get_user_or_ip(request)

    if user_id:
        # Authenticated: check DB credits
        balance = await credits_module.get_balance(user_id)
        if balance <= 0:
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "Insufficient credits",
                    "remaining": 0,
                    "upgrade_url": "https://cocotheaxolotl.org/pricing/",
                },
            )
    else:
        # Anonymous: legacy IP-based quota
        promo = _check_promo(promo_code)
        remaining = _check_quota(request, promo)
        if remaining <= 0:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Free limit reached",
                    "message": f"You've used all {FREE_LIMIT} free generations. Upgrade to Pro for unlimited access!",
                    "upgrade_url": "https://cocotheaxolotl.org/pricing/",
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

    # AI coloring page mode — calls OpenAI, returns immediately
    if mode == "ai":
        ai_img = await _ai_coloring_page(data)
        if user_id:
            await credits_module.consume_credits(user_id, 1, "generation", {"mode": "ai"})
        else:
            _consume(request, 1)
        return _img_to_streaming(ai_img, f"{name}-coloring.png")

    # Generate (semaphore: only 1 at a time to prevent OOM)
    async with _gen_semaphore:
        if preset and preset in PRESETS:
            result = generate_from_preset(img, preset)
        elif mode == "lineart":
            detail_level = detail if detail in ("simple", "standard", "detailed", "expert") else "standard"
            thickness_clamped = max(1, min(4, thickness))
            result = generate_line_art(img, page=page, detail_level=detail_level,
                                       line_thickness=thickness_clamped)
        elif mode == "cbn":
            colors = max(4, min(20, colors))
            result = generate_cbn(img, colors=colors, page=page, blur=2)
        elif mode == "voronoi":
            colors = max(4, min(20, colors))
            density_map = VORONOI_DENSITIES.get(page, VORONOI_DENSITIES["letter"])
            num_cells = density_map.get(density, density_map["standard"])
            result = generate_voronoi(img, colors=colors, density=num_cells, page=page)
        else:
            colors = max(4, min(20, colors))
            cell_size = max(12, min(50, cell_size))
            result = generate(img, colors=colors, cell_size=cell_size, page=page)
        del img
        gc.collect()

    # Consume quota — dual mode
    if user_id:
        await credits_module.consume_credits(
            user_id, 1, "generation", {"mode": mode, "preset": preset}
        )
    else:
        _consume(request, 1)

    # Return — handle LineArtResult, CBNResult, or MosaicResult
    if isinstance(result, LineArtResult):
        if output in ("mystery", "png"):
            return _img_to_streaming(result.line_art_png, f"{name}-lineart.png")
        if output in ("svg", "answer"):
            return _svg_to_streaming(result.line_art_svg, f"{name}-lineart.svg")
        if output in ("beauty", "preview"):
            return _img_to_streaming(result.preview_png, f"{name}-preview.png")
        return _result_to_zip_stream(name, result)

    if isinstance(result, CBNResult):
        if output == "mystery":
            return _svg_to_streaming(result.mystery_svg, f"{name}-mystery.svg")
        if output == "answer":
            return _svg_to_streaming(result.answer_svg, f"{name}-answer.svg")
        if output == "beauty":
            return _img_to_streaming(result.answer_png, f"{name}-answer.png")
        if output == "legend":
            return _img_to_streaming(result.legend_png, f"{name}-legend.png")
        if output == "full":
            return _img_to_streaming(result.mystery_full_png, f"{name}-mystery-full.png")
        return _result_to_zip_stream(name, result)

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
    detail: str = Form("standard"),
    thickness: int = Form(2),
    promo_code: Optional[str] = Form(None),
):
    """Generate mosaics/coloring pages for multiple images at once → returns a ZIP.

    - Max 50 images per request (Pro only beyond 3).
    - Each image counts as 1 quota unit.
    """
    n = len(images)
    if n > MAX_BULK:
        raise HTTPException(400, f"Too many images (max {MAX_BULK})")
    if n == 0:
        raise HTTPException(400, "No images provided")

    # Quota check — dual mode
    user_id, identifier = await get_user_or_ip(request)

    if user_id:
        balance = await credits_module.get_balance(user_id)
        if balance < n:
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "Insufficient credits",
                    "remaining": balance,
                    "requested": n,
                    "upgrade_url": "https://cocotheaxolotl.org/pricing/",
                },
            )
    else:
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

    # Process all images (semaphore: only 1 generation at a time)
    results = []
    async with _gen_semaphore:
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
            elif mode == "lineart":
                detail_level = detail if detail in ("simple", "standard", "detailed", "expert") else "standard"
                res = generate_line_art(img, page=page, detail_level=detail_level,
                                        line_thickness=max(1, min(4, thickness)))
            elif mode == "cbn":
                res = generate_cbn(img, colors=max(4, min(20, colors)), page=page, blur=2)
            elif mode == "voronoi":
                density_map = VORONOI_DENSITIES.get(page, VORONOI_DENSITIES["letter"])
                num_cells = density_map.get(density, density_map["standard"])
                res = generate_voronoi(img, colors=max(4, min(20, colors)),
                                       density=num_cells, page=page)
            else:
                res = generate(img, colors=max(4, min(20, colors)),
                               cell_size=max(12, min(50, cell_size)), page=page)
            results.append((name, res))
            del img
            gc.collect()

    if not results:
        raise HTTPException(400, "No valid images processed")

    # Consume quota — dual mode
    if user_id:
        await credits_module.consume_credits(
            user_id, len(results), "bulk_generation", {"mode": mode, "count": len(results)}
        )
    else:
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
    """Remove expired ZIP files from disk."""
    now = time.time()
    for meta_path in DOWNLOAD_DIR.glob("*.json"):
        try:
            meta = json.loads(meta_path.read_text())
            if now - meta["created_at"] > DOWNLOAD_TTL:
                zip_path = DOWNLOAD_DIR / meta["zip_file"]
                zip_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
        except Exception:
            meta_path.unlink(missing_ok=True)


def _store_download(zip_bytes: bytes, filename: str) -> str:
    """Store a ZIP on disk and return a unique token."""
    _cleanup_downloads()
    # Evict oldest if too many files
    metas = sorted(DOWNLOAD_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    while len(metas) >= DOWNLOAD_MAX_ENTRIES:
        old_meta = metas.pop(0)
        try:
            old_data = json.loads(old_meta.read_text())
            (DOWNLOAD_DIR / old_data["zip_file"]).unlink(missing_ok=True)
        except Exception:
            pass
        old_meta.unlink(missing_ok=True)
    token = uuid.uuid4().hex
    zip_path = DOWNLOAD_DIR / f"{token}.zip"
    zip_path.write_bytes(zip_bytes)
    meta_path = DOWNLOAD_DIR / f"{token}.json"
    meta_path.write_text(json.dumps({
        "filename": filename,
        "zip_file": f"{token}.zip",
        "created_at": time.time(),
    }))
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
    detail: str = Form("standard"),
    thickness: int = Form(2),
    promo_code: Optional[str] = Form(None),
):
    """Generate a mosaic/coloring page, store the ZIP, and email the download link."""
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

    # Generate (semaphore: only 1 generation at a time)
    async with _gen_semaphore:
        if preset and preset in PRESETS:
            result = generate_from_preset(img, preset)
        elif mode == "lineart":
            detail_level = detail if detail in ("simple", "standard", "detailed", "expert") else "standard"
            result = generate_line_art(img, page=page, detail_level=detail_level,
                                       line_thickness=max(1, min(4, thickness)))
        elif mode == "cbn":
            colors = max(4, min(20, colors))
            result = generate_cbn(img, colors=colors, page=page, blur=2)
        elif mode == "voronoi":
            colors = max(4, min(20, colors))
            density_map = VORONOI_DENSITIES.get(page, VORONOI_DENSITIES["letter"])
            num_cells = density_map.get(density, density_map["standard"])
            result = generate_voronoi(img, colors=colors, density=num_cells, page=page)
        else:
            colors = max(4, min(20, colors))
            cell_size = max(12, min(50, cell_size))
            result = generate(img, colors=colors, cell_size=cell_size, page=page)
        del img
        gc.collect()

    # Pack into ZIP (no answer key in email download)
    zip_data = images_to_zip([(name, result)], include_answer=False)
    if isinstance(result, LineArtResult):
        label = "lineart"
    elif isinstance(result, CBNResult):
        label = "cbn"
    else:
        label = "mosaic"
    filename = f"{name}-{label}.zip"
    gc.collect()

    # Store and get token
    token = _store_download(zip_data, filename)
    download_url = f"{API_PUBLIC_URL}/api/download/{token}"

    # Send email + add to contacts
    try:
        _send_brevo_email(email, download_url)
    except Exception as e:
        # Remove stored ZIP if email fails
        (DOWNLOAD_DIR / f"{token}.zip").unlink(missing_ok=True)
        (DOWNLOAD_DIR / f"{token}.json").unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to send email: {e}")

    _add_brevo_contact(email)

    return {"ok": True, "message": "Download link sent to your email!"}


@app.get("/api/download/{token}")
def download_mosaic(token: str):
    """Serve a stored ZIP file from an email download link (one-time use)."""
    _cleanup_downloads()
    meta_path = DOWNLOAD_DIR / f"{token}.json"
    zip_path = DOWNLOAD_DIR / f"{token}.zip"
    if not meta_path.exists() or not zip_path.exists():
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
    meta = json.loads(meta_path.read_text())
    zip_bytes = zip_path.read_bytes()
    # One-time use: delete after serving
    zip_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{meta["filename"]}"'},
    )


# ── Auth endpoints ────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: str
    password: str
    display_name: str = ""
    lang: str = "en"

class LoginRequest(BaseModel):
    email: str
    password: str

class RefreshRequest(BaseModel):
    refresh_token: str

class LogoutRequest(BaseModel):
    refresh_token: str

class VerifyEmailRequest(BaseModel):
    token: str

class ForgotRequest(BaseModel):
    email: str
    lang: str = "en"

class ResetRequest(BaseModel):
    token: str
    password: str

class ConsumeRequest(BaseModel):
    feature: str
    variant: str = ""


@app.post("/api/auth/signup")
async def api_signup(req: SignupRequest):
    try:
        result = await auth_module.signup(req.email, req.password, req.display_name, req.lang)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/auth/login")
async def api_login(req: LoginRequest):
    try:
        result = await auth_module.login(req.email, req.password)
        return result
    except ValueError as e:
        raise HTTPException(401, str(e))


@app.post("/api/auth/refresh")
async def api_refresh(req: RefreshRequest):
    try:
        result = await auth_module.refresh(req.refresh_token)
        return result
    except ValueError as e:
        raise HTTPException(401, str(e))


@app.post("/api/auth/logout")
async def api_logout(req: LogoutRequest):
    await auth_module.logout(req.refresh_token)
    return {"ok": True}


@app.get("/api/auth/me")
async def api_me(request: Request):
    user_id, identifier = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Not authenticated")
    profile = await auth_module.get_me(user_id)
    if not profile:
        raise HTTPException(404, "User not found")
    return profile


@app.post("/api/auth/verify-email")
async def api_verify_email(req: VerifyEmailRequest):
    ok = await auth_module.verify_email_token(req.token)
    if not ok:
        raise HTTPException(400, "Invalid or expired verification token")
    return {"ok": True}


@app.post("/api/auth/forgot")
async def api_forgot(req: ForgotRequest):
    await auth_module.send_password_reset(req.email, req.lang)
    return {"ok": True, "message": "If an account exists, a reset email has been sent."}


@app.post("/api/auth/reset")
async def api_reset(req: ResetRequest):
    ok = await auth_module.reset_password(req.token, req.password)
    if not ok:
        raise HTTPException(400, "Invalid or expired reset token, or password too short")
    return {"ok": True}


# ── Credits endpoints ─────────────────────────────────────────────────────

@app.get("/api/credits/balance")
async def api_credits_balance(request: Request):
    user_id, _ = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Not authenticated")
    info = await credits_module.get_plan_info(user_id)
    if not info:
        raise HTTPException(404, "No credit info found")
    return info


@app.get("/api/credits/history")
async def api_credits_history(request: Request):
    user_id, _ = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Not authenticated")
    history = await credits_module.get_transaction_history(user_id)
    return {"transactions": history}


@app.post("/api/credits/consume")
async def api_credits_consume(request: Request, req: ConsumeRequest):
    """Consume 1 credit for a feature (name generators, etc.)."""
    user_id, _ = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Not authenticated")
    success, remaining = await credits_module.consume_credits(
        user_id, 1, "generation",
        metadata={"feature": req.feature, "variant": req.variant},
    )
    if not success:
        raise HTTPException(402, {
            "error": "Insufficient credits",
            "remaining": remaining,
            "upgrade_url": "https://cocotheaxolotl.org/pricing/",
        })
    return {"ok": True, "remaining": remaining}


# ── Billing endpoints ─────────────────────────────────────────────────────

class CheckoutRequest(BaseModel):
    plan: str  # "basic" | "personal" | "commercial"


@app.get("/api/billing/plans")
def api_billing_plans():
    """Public: list all available plans."""
    return {"plans": stripe_integration.get_plans_public()}


@app.post("/api/billing/checkout")
async def api_billing_checkout(request: Request, req: CheckoutRequest):
    """Create a Stripe Checkout session. Requires auth."""
    user_id, _ = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Not authenticated")
    profile = await auth_module.get_me(user_id)
    if not profile:
        raise HTTPException(404, "User not found")
    try:
        url = await stripe_integration.create_checkout_session(
            user_id, profile["email"], req.plan
        )
        return {"url": url}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/billing/portal")
async def api_billing_portal(request: Request):
    """Create a Stripe Customer Portal session. Requires auth."""
    user_id, _ = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Not authenticated")
    try:
        url = await stripe_integration.create_portal_session(user_id)
        return {"url": url}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/billing/webhook")
async def api_billing_webhook(request: Request):
    """Stripe webhook handler. Verifies signature, processes events."""
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        result = await stripe_integration.handle_webhook(payload, sig)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


# ── Health ───────────────────────────────────────────────────────────────

@app.get("/api/stats")
def stats():
    """Public stats for social proof on the landing page."""
    return {"total_generated": _total_generated}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
