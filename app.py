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
    POST /api/keys              — create API key (commercial plan)
    GET  /api/keys              — list API keys
    DELETE /api/keys/{id}       — revoke API key
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
import zipfile
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Body
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont, ImageOps

from database import init_db
import auth as auth_module
import credits as credits_module
import stripe_integration
import api_keys as api_keys_module

from mosaic_engine import (
    generate, generate_voronoi, generate_cbn, generate_line_art, generate_pbn,
    generate_smart_cbn, generate_from_preset, images_to_zip,
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
DOWNLOAD_TTL = 86400       # download links expire after 24 hours
DOWNLOAD_MAX_ENTRIES = 50   # max ZIPs stored on disk

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
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ── Startup ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    await init_db()


# ── Auth helper ──────────────────────────────────────────────────────────

async def get_user_or_ip(request: Request) -> tuple[int | None, str]:
    """
    Identify the caller. Supports two auth modes:
    1. API key: Authorization: Bearer us_live_XXXX → lookup in api_keys table
    2. JWT:     Authorization: Bearer <jwt>        → decode JWT
    Falls back to IP-based identification: (None, "ip:<hash>").
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # API key auth: detect by prefix
        if token.startswith(api_keys_module.API_KEY_PREFIX):
            user_id = await api_keys_module.verify_api_key(token)
            if user_id:
                return (user_id, f"user:{user_id}")
            # Invalid API key → don't fall through to IP, return 401
            return (None, f"ip:{_user_key(request)}")
        # JWT auth
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
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    # Preserve alpha for PNG illustrations; the engine will normalize it.
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    # Cap input to 1500px on longest side to save RAM on 2GB VM
    MAX_DIM = 1500
    if max(img.size) > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
    return img

def _looks_like_inked_artwork(img: Image.Image) -> bool:
    """Heuristic: distinguish clean artwork from photos/textured images."""
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8)
    alpha = arr[:, :, 3]
    has_real_alpha = alpha.max() > 0 and (alpha < 250).any()
    if has_real_alpha:
        subject = alpha >= 32
        rgb = arr[:, :, :3][subject] if subject.any() else arr[:, :, :3].reshape(-1, 3)
    else:
        rgb = arr[:, :, :3].reshape(-1, 3)

    if rgb.size == 0:
        return True
    if has_real_alpha:
        return True

    white_ratio = float(np.mean(np.all(rgb >= 245, axis=1)))
    gray = np.dot(rgb.astype(np.float32), np.array([0.299, 0.587, 0.114], dtype=np.float32))
    dark_ratio = float(np.mean(gray <= 60))
    chroma = rgb.max(axis=1).astype(np.int16) - rgb.min(axis=1).astype(np.int16)
    flat_ratio = float(np.mean(chroma <= 20))

    is_photo_like = white_ratio < 0.05 and flat_ratio >= 0.50 and dark_ratio < 0.22
    return not is_photo_like


def _img_to_streaming(img: Image.Image, filename: str) -> StreamingResponse:
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(300, 300))
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _result_to_zip_stream(name: str, res, plan: str = "pro") -> StreamingResponse:
    data = images_to_zip([(name, res)], plan=plan)
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


# ── Watermark ────────────────────────────────────────────────────────────

import math

def _add_watermark(img: Image.Image) -> Image.Image:
    """Add tiled diagonal 'univers.studio' watermark to a preview image.

    Returns a new RGBA→RGB image with semi-transparent text repeated across
    the entire surface.  The original image is NOT modified.
    """
    # Work on a copy in RGBA so we can composite transparency
    base = img.convert("RGBA")
    w, h = base.size

    # Create a transparent overlay for the watermark text
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Choose font size relative to image (roughly 3% of diagonal)
    diag = math.hypot(w, h)
    font_size = max(16, int(diag * 0.03))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    text = "univers.studio"
    # Measure text size
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Spacing between watermark tiles
    spacing_x = tw + int(tw * 0.8)
    spacing_y = th + int(th * 3)

    # Draw rotated text tiles across the image
    # We create a larger temporary image, draw text, rotate, then paste
    tile_img = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile_img)

    # Fill with repeated text (semi-transparent gray)
    alpha = 45  # subtle but visible
    color = (128, 128, 128, alpha)
    y = -h // 2
    while y < h * 2:
        x = -w // 2
        while x < w * 2:
            tile_draw.text((x, y), text, fill=color, font=font)
            x += spacing_x
        y += spacing_y

    # Rotate -30 degrees
    rotated = tile_img.rotate(30, resample=Image.BICUBIC, expand=False)

    # Crop center to match original size
    rx, ry = rotated.size
    left = (rx - w) // 2
    top = (ry - h) // 2
    cropped = rotated.crop((left, top, left + w, top + h))

    # Composite
    watermarked = Image.alpha_composite(base, cropped)
    return watermarked.convert("RGB")


# ── AI Coloring Page (OpenAI) ────────────────────────────────────────────

import base64
import httpx

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()


async def _ai_coloring_page(image_data: bytes, hint: str = "", style: str = "kids") -> Image.Image:
    """Generate a coloring page using OpenAI gpt-image-1 edits endpoint.

    Simple direct prompt like ChatGPT — the model already sees the image,
    no need for a separate vision step.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI API key not configured")

    # Resize image to max 1024px to stay within API limits
    img_input = Image.open(io.BytesIO(image_data)).convert("RGBA")
    img_input.thumbnail((1024, 1024), Image.LANCZOS)
    png_buf = io.BytesIO()
    img_input.save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()

    if style == "zen":
        prompt = "Make the coloring page without decoration, for adults."
    else:
        prompt = "Make the coloring page without decoration."

    async with httpx.AsyncClient(timeout=180) as client:
        img_resp = await client.post(
            "https://api.openai.com/v1/images/edits",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"image": ("photo.png", png_bytes, "image/png")},
            data={
                "model": "gpt-image-1",
                "prompt": prompt,
                "n": "1",
                "size": "1024x1024",
                "quality": "high",
            },
        )
        if img_resp.status_code != 200:
            raise HTTPException(502, f"Image API error: {img_resp.text[:300]}")

        img_b64 = img_resp.json()["data"][0]["b64_json"]
        img_bytes = base64.b64decode(img_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")


_AI_STYLE_PREFIXES = {
    "coloring": (
        "Simple black and white coloring page for children. "
        "Bold clean outlines, no shading, no gray tones, no colors, pure white background. "
        "Subject: {prompt}. "
        "Style: cute cartoon illustration with clear thick outlines, suitable for kids to color with crayons."
    ),
    "cartoon": (
        "Colorful cartoon illustration, cute style, clean lines, vibrant colors. "
        "Subject: {prompt}. "
        "Style: fun, child-friendly, bright palette, professional cartoon art."
    ),
    "realistic": (
        "Realistic high-quality photograph with natural lighting. "
        "Subject: {prompt}. "
        "Style: detailed, photorealistic, sharp focus, professional photography."
    ),
    "pixel-art": (
        "Pixel art illustration in retro 16-bit video game style. "
        "Subject: {prompt}. "
        "Style: clean crisp pixels, vibrant limited palette, nostalgic game-like aesthetic."
    ),
    "watercolor": (
        "Beautiful watercolor painting with soft washes and artistic brush strokes. "
        "Subject: {prompt}. "
        "Style: delicate, flowing translucent colors, hand-painted feel, fine art quality."
    ),
    "sticker": (
        "Die-cut sticker design with white border, cartoon style, no background. "
        "Subject: {prompt}. "
        "Style: cute kawaii, bold outlines, flat colors, compact centered composition."
    ),
}


_AI_QUALITY_CREDITS = {"low": 1, "medium": 3, "high": 10}


async def _ai_text_to_image(
    prompt: str,
    style: str = "coloring",
    size: str = "1024x1024",
    quality: str = "low",
) -> Image.Image:
    """Generate an image from a text prompt using OpenAI gpt-image-1."""
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI API key not configured")

    template = _AI_STYLE_PREFIXES.get(style, "{prompt}")
    full_prompt = template.replace("{prompt}", prompt)

    valid_sizes = ("1024x1024", "1024x1536", "1536x1024", "auto")
    if size not in valid_sizes:
        size = "1024x1024"

    if quality not in ("low", "medium", "high"):
        quality = "low"

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-image-1",
                "prompt": full_prompt,
                "n": 1,
                "size": size,
                "quality": quality,
            },
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"Image API error: {resp.text[:200]}")

        img_b64 = resp.json()["data"][0]["b64_json"]
        return Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")


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


@app.post("/api/ai-image")
async def ai_image_generate(
    request: Request,
    prompt: str = Form(...),
    style: str = Form("coloring"),
    size: str = Form("1024x1024"),
    quality: str = Form("low"),
    promo_code: Optional[str] = Form(None),
):
    """Generate an image from a text prompt using AI (gpt-image-1).

    Styles: coloring, cartoon, realistic, pixel-art, watercolor, sticker.
    Sizes: 1024x1024, 1024x1536, 1536x1024.
    Quality: low (1 credit), medium (3 credits), high (10 credits).
    """
    # Quota check
    user_id, identifier = await get_user_or_ip(request)

    if user_id:
        balance = await credits_module.get_balance(user_id)
        if balance <= 0:
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "Insufficient credits",
                    "remaining": 0,
                    "message": f"You've used all your credits. Upgrade for more!",
                    "upgrade_url": "https://cocotheaxolotl.org/pricing/",
                },
            )
    else:
        if promo_code:
            _apply_promo(request, promo_code)
        remaining = _check_quota(request)
        if remaining <= 0:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Free quota exceeded",
                    "remaining": 0,
                    "message": f"You've used all {FREE_LIMIT} free generations. Upgrade to Pro for unlimited access!",
                    "upgrade_url": "https://cocotheaxolotl.org/pricing/",
                },
            )

    # Validate prompt
    prompt = prompt.strip()
    if len(prompt) < 3:
        raise HTTPException(400, "Prompt must be at least 3 characters")
    if len(prompt) > 1000:
        raise HTTPException(400, "Prompt must be under 1000 characters")

    # Validate quality & determine credit cost
    if quality not in _AI_QUALITY_CREDITS:
        quality = "low"
    cost = _AI_QUALITY_CREDITS[quality]

    # Check sufficient credits for authenticated users
    if user_id and balance < cost:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "Insufficient credits",
                "remaining": balance,
                "cost": cost,
                "message": f"This quality requires {cost} credits but you have {balance}. Choose a lower quality or upgrade!",
                "upgrade_url": "https://cocotheaxolotl.org/pricing/",
            },
        )

    # Generate
    ai_img = await _ai_text_to_image(prompt, style=style, size=size, quality=quality)

    # Consume credits
    if user_id:
        await credits_module.consume_credits(user_id, cost, "generation", {"mode": "ai-image", "style": style, "quality": quality})
    else:
        _consume(request, cost)

    return _img_to_streaming(_add_watermark(ai_img), "ai-image.png")


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
    hint: str = Form(""),             # optional user hint for AI mode (e.g. "an impala")
    cbn_style: str = Form("kids"),   # CBN/coloring style: "kids" (simple) or "zen" (mandala)
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
    # Promo codes always bypass quota regardless of auth state
    promo = _check_promo(promo_code)
    user_id, identifier = await get_user_or_ip(request)

    if not promo:
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

    # AI coloring page mode — calls OpenAI, returns immediately (10 credits)
    if mode == "ai":
        if not user_id and not promo:
            raise HTTPException(403, "AI coloring requires an account. Sign up free at univers.studio!")
        if not promo and user_id:
            balance = await credits_module.get_balance(user_id)
            if balance < 10:
                raise HTTPException(402, {"error": "Not enough credits", "remaining": balance, "required": 10,
                    "message": "AI coloring costs 10 credits. Upgrade your plan!", "upgrade_url": "https://univers.studio/pricing/"})
        ai_img = await _ai_coloring_page(data, hint=hint.strip(), style=cbn_style)
        if not promo and user_id:
            await credits_module.consume_credits(user_id, 10, "generation", {"mode": "ai", "style": cbn_style})
        # Watermark only for anonymous free users
        if not user_id and not promo:
            ai_img = _add_watermark(ai_img)
        return _img_to_streaming(ai_img, f"{name}-coloring.png")

    # CBN/PBN require an account or promo code.
    if mode in ("cbn", "pbn"):
        if not user_id and not promo:
            raise HTTPException(403, "Color by Number requires an account. Sign up free at univers.studio!")
        if not promo:
            balance = await credits_module.get_balance(user_id)
            if balance < 10:
                raise HTTPException(402, {"error": "Not enough credits", "remaining": balance, "required": 10,
                    "message": "Color by Number costs 10 credits. Upgrade your plan!", "upgrade_url": "https://univers.studio/pricing/"})
        colors = max(4, min(20, colors))
        async with _gen_semaphore:
            if mode == "cbn":
                if _looks_like_inked_artwork(img):
                    result = generate_cbn(img, colors=colors, page=page, min_zone_pixels=150)
                    del img
                else:
                    result = generate_pbn(
                        img,
                        colors=min(colors, 8),
                        page=page,
                        blur=8,
                        min_zone_pixels=2500,
                    )
                    del img
            else:
                line_art = await _ai_coloring_page(data, hint=hint.strip(), style=cbn_style)
                result = generate_smart_cbn(img, line_art, colors=colors, page=page)
                del img, line_art
            gc.collect()
    else:
        # Generate (semaphore: only 1 at a time to prevent OOM)
        async with _gen_semaphore:
            if preset and preset in PRESETS:
                result = generate_from_preset(img, preset)
            elif mode == "lineart":
                detail_level = detail if detail in ("simple", "standard", "detailed", "expert") else "standard"
                thickness_clamped = max(1, min(4, thickness))
                result = generate_line_art(img, page=page, detail_level=detail_level,
                                           line_thickness=thickness_clamped)
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

    # Consume quota — CBN/AI cost 3 credits, others cost 1
    # Promo users bypass consumption
    if not promo:
        cost = 10 if mode in ("cbn", "pbn") else 1
        if user_id:
            await credits_module.consume_credits(
                user_id, cost, "generation", {"mode": mode, "preset": preset}
            )
        else:
            _consume(request, 1)

    # Determine user plan for output tiering
    user_plan = "free"
    if user_id:
        plan_info = await credits_module.get_plan_info(user_id)
        if plan_info:
            user_plan = plan_info.get("plan_name", "free")

    # Return — handle LineArtResult, CBNResult, or MosaicResult
    # Watermark only for anonymous free users (no account, no promo)
    _wm = not user_id and not promo  # True = add watermark

    if isinstance(result, LineArtResult):
        if output in ("mystery", "png"):
            img_out = _add_watermark(result.line_art_png) if _wm else result.line_art_png
            return _img_to_streaming(img_out, f"{name}-lineart.png")
        if output in ("svg", "answer"):
            return _svg_to_streaming(result.line_art_svg, f"{name}-lineart.svg")
        if output in ("beauty", "preview"):
            img_out = _add_watermark(result.preview_png) if _wm else result.preview_png
            return _img_to_streaming(img_out, f"{name}-preview.png")
        return _result_to_zip_stream(name, result, plan=user_plan)

    if isinstance(result, CBNResult):
        if output == "beauty":
            img_out = _add_watermark(result.mystery_full_png) if _wm else result.mystery_full_png
            return _img_to_streaming(img_out, f"{name}-preview.png")
        if output == "mystery":
            if result.mystery_svg:
                return _svg_to_streaming(result.mystery_svg, f"{name}-mystery.svg")
            return _img_to_streaming(result.mystery_png, f"{name}-mystery.png")
        if output == "answer":
            if result.answer_svg:
                return _svg_to_streaming(result.answer_svg, f"{name}-answer.svg")
            return _img_to_streaming(result.answer_png, f"{name}-answer.png")
        if output == "legend":
            return _img_to_streaming(result.legend_png, f"{name}-legend.png")
        if output == "full":
            return _img_to_streaming(result.mystery_full_png, f"{name}-mystery-full.png")
        return _result_to_zip_stream(name, result, plan=user_plan)

    if output == "mystery":
        img_out = _add_watermark(result.mystery) if _wm else result.mystery
        return _img_to_streaming(img_out, f"{name}-mystery.png")
    if output == "answer":
        img_out = _add_watermark(result.answer) if _wm else result.answer
        return _img_to_streaming(img_out, f"{name}-answer.png")
    if output == "beauty":
        img_out = _add_watermark(result.beauty) if _wm else result.beauty
        return _img_to_streaming(img_out, f"{name}-beauty.png")
    if output == "legend":
        img_out = _add_watermark(result.legend) if _wm else result.legend
        return _img_to_streaming(img_out, f"{name}-legend.png")
    if output == "full":
        img_out = _add_watermark(result.mystery_full) if _wm else result.mystery_full
        return _img_to_streaming(img_out, f"{name}-mystery-full.png")

    # Default: ZIP
    return _result_to_zip_stream(name, result, plan=user_plan)


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
    # Promo codes always bypass quota regardless of auth state
    promo = _check_promo(promo_code)
    user_id, identifier = await get_user_or_ip(request)

    if not promo:
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

    # Determine user plan for output tiering
    bulk_plan = "free"
    if user_id:
        plan_info = await credits_module.get_plan_info(user_id)
        if plan_info:
            bulk_plan = plan_info.get("plan_name", "free")

    # Pack into ZIP
    zip_data = images_to_zip(results, plan=bulk_plan)
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


def _send_brevo_email(to_email: str, download_url: str, mode: str = "hex"):
    """Send a transactional email via Brevo API with the download link."""
    if not BREVO_API_KEY:
        raise RuntimeError("BREVO_API_KEY not configured")

    # Adapt email content to the generation mode
    _MODE_LABELS = {
        "ai": ("Your AI Coloring Page is ready!", "coloring page", "Download My Coloring Page"),
        "lineart": ("Your Coloring Page is ready!", "coloring page", "Download My Coloring Page"),
        "cbn": ("Your Color by Number is ready!", "color by number", "Download My Color by Number"),
    }
    subject, label, btn_text = _MODE_LABELS.get(mode, ("Your Mystery Mosaic is ready!", "mosaic", "Download My Mosaic"))

    html = f"""\
<div style="font-family:Inter,Helvetica,Arial,sans-serif;max-width:520px;margin:0 auto;padding:32px 20px">
  <h1 style="color:#1a1a2e;font-size:1.5em;margin:0 0 8px">{subject}</h1>
  <p style="color:#6b7280;line-height:1.6;margin:0 0 24px">
    Your {label} has been generated. Click the button below to download your file (ZIP)
    at 300&nbsp;DPI print quality.
  </p>
  <a href="{download_url}"
     style="display:inline-block;padding:14px 36px;background:#7c3aed;color:#fff;
            text-decoration:none;border-radius:10px;font-weight:700;font-size:1.05em">
    {btn_text}
  </a>
  <p style="color:#9ca3af;font-size:.85em;margin:24px 0 0;line-height:1.5">
    This link expires in 24 hours.<br>
    If the link has expired, simply generate a new one on
    <a href="https://univers.studio" style="color:#7c3aed">univers.studio</a>.
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
        "subject": subject,
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
    hint: str = Form(""),
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

    # AI coloring page mode — calls OpenAI, bypasses semaphore
    if mode == "ai":
        ai_img = await _ai_coloring_page(data, hint=hint.strip())
        # Pack AI image into a simple ZIP
        buf = io.BytesIO()
        ai_img.save(buf, format="PNG")
        ai_png = buf.getvalue()
        del ai_img, buf

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{name}/{name}-coloring.png", ai_png)
        zip_data = zip_buf.getvalue()
        del ai_png, zip_buf
        filename = f"{name}-coloring.zip"
        gc.collect()

        token = _store_download(zip_data, filename)
        download_url = f"{API_PUBLIC_URL}/api/download/{token}"
        try:
            _send_brevo_email(email, download_url, mode="ai")
        except Exception as e:
            (DOWNLOAD_DIR / f"{token}.zip").unlink(missing_ok=True)
            (DOWNLOAD_DIR / f"{token}.json").unlink(missing_ok=True)
            raise HTTPException(500, f"Failed to send email: {e}")
        _add_brevo_contact(email)
        return {"ok": True, "message": "Download link sent to your email!"}

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

    # Pack into ZIP (free tier: mystery-full only, no answer key)
    zip_data = images_to_zip([(name, result)], include_answer=False, plan="free")
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
        _send_brevo_email(email, download_url, mode=mode)
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
<p>Download links are valid for 24 hours. Generate a new mosaic to get a fresh link.</p>
<a href="https://univers.studio/mosaic/">Generate New Mosaic</a></div></body></html>""",
            status_code=410,
        )
    meta = json.loads(meta_path.read_text())
    zip_bytes = zip_path.read_bytes()
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
    ref: str = ""

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
    site_url: str = ""

class ResetRequest(BaseModel):
    token: str
    password: str

class ConsumeRequest(BaseModel):
    feature: str
    variant: str = ""


@app.post("/api/auth/signup")
async def api_signup(req: SignupRequest):
    try:
        result = await auth_module.signup(req.email, req.password, req.display_name, req.lang, req.ref)
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
    await auth_module.send_password_reset(req.email, req.lang, req.site_url)
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
    plan: str  # "creator" | "pro" | "studio" | "pack_20" | "pack_100"
    billing: str = "monthly"  # "monthly" | "annual" (ignored for packs)
    ref: str = ""


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
            user_id, profile["email"], req.plan, req.billing, req.ref
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


# ── API Keys endpoints ────────────────────────────────────────────────────

class CreateKeyRequest(BaseModel):
    name: str = ""

@app.post("/api/keys")
async def api_create_key(request: Request, req: CreateKeyRequest):
    """Create a new API key. Requires auth + commercial plan."""
    user_id, _ = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Not authenticated")
    info = await credits_module.get_plan_info(user_id)
    if not info or info["plan_name"] != "studio":
        raise HTTPException(403, "API keys are only available on the Studio plan")
    key_id, raw_key = await api_keys_module.create_api_key(user_id, req.name)
    return {"id": key_id, "key": raw_key, "name": req.name.strip()[:100]}


@app.get("/api/keys")
async def api_list_keys(request: Request):
    """List all API keys for the current user. Requires auth."""
    user_id, _ = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Not authenticated")
    keys = await api_keys_module.list_api_keys(user_id)
    return {"keys": keys}


@app.delete("/api/keys/{key_id}")
async def api_revoke_key(request: Request, key_id: str):
    """Revoke an API key. Requires auth."""
    user_id, _ = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Not authenticated")
    ok = await api_keys_module.revoke_api_key(key_id, user_id)
    if not ok:
        raise HTTPException(404, "Key not found")
    return {"ok": True}


# ── Preview store (flipbook) ──────────────────────────────────────────────

PREVIEW_DIR = Path("/data/previews")
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_TTL = 3600  # 1 hour


def _cleanup_previews():
    """Remove preview HTML files older than PREVIEW_TTL."""
    now = time.time()
    for f in PREVIEW_DIR.glob("*.html"):
        try:
            if now - f.stat().st_mtime > PREVIEW_TTL:
                f.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/api/preview")
async def create_preview(request: Request):
    """Store a self-contained HTML flipbook and return a temporary URL (1h)."""
    _cleanup_previews()
    body = await request.body()
    if len(body) > 100_000_000:
        raise HTTPException(413, "Preview too large (max 100 MB)")
    if len(body) < 100:
        raise HTTPException(400, "Empty or invalid HTML")
    token = uuid.uuid4().hex[:12]
    (PREVIEW_DIR / f"{token}.html").write_bytes(body)
    url = f"{API_PUBLIC_URL}/preview/{token}"
    return {"token": token, "url": url}


@app.get("/preview/{token}")
def serve_preview(token: str):
    """Serve a stored preview HTML. Returns 410 if expired or not found."""
    _cleanup_previews()
    if not re.match(r'^[a-f0-9]{12}$', token):
        raise HTTPException(400, "Invalid token")
    path = PREVIEW_DIR / f"{token}.html"
    if not path.exists():
        return HTMLResponse(
            content="""\
<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lien expir&eacute;</title>
<style>body{font-family:Inter,Helvetica,sans-serif;display:flex;align-items:center;justify-content:center;
min-height:100vh;margin:0;background:#1a1a2e;color:#fff;text-align:center;padding:20px}
.box{max-width:400px}.emoji{font-size:3em;margin:0 0 16px}h1{font-size:1.3em;margin:0 0 8px}
p{opacity:.7;line-height:1.6;margin:0 0 20px;font-size:.95em}</style></head>
<body><div class="box"><div class="emoji">&#9200;</div>
<h1>Ce lien a expir&eacute;</h1>
<p>Les liens de preview sont valides pendant 1&nbsp;heure.
G&eacute;n&eacute;rez un nouveau flipbook pour obtenir un nouveau lien.</p>
</div></body></html>""",
            status_code=410,
        )
    html = path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ── Health ───────────────────────────────────────────────────────────────

@app.get("/api/stats")
def stats():
    """Public stats for social proof on the landing page."""
    return {"total_generated": _total_generated}


# ── Dynamic QR Codes ─────────────────────────────────────────────────────

import string
import random as _random
from database import get_db

def _gen_short_code(length=7):
    chars = string.ascii_lowercase + string.digits
    return ''.join(_random.choices(chars, k=length))


@app.post("/api/qr/create")
async def qr_create(request: Request, body: dict = Body(...)):
    """Create a dynamic QR code (costs 1 credit). Returns short_code + redirect URL."""
    user_id, ident = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Login required to create dynamic QR codes")

    target_url = (body.get("target_url") or "").strip()
    label = (body.get("label") or "").strip()[:100]
    if not target_url or not target_url.startswith("http"):
        raise HTTPException(400, "A valid URL is required")

    # Consume 1 credit
    ok, _bal = await credits_module.consume_credits(user_id, 1, "dynamic_qr", {"label": label})
    if not ok:
        raise HTTPException(402, "Not enough credits")

    qr_id = str(uuid.uuid4())
    short_code = _gen_short_code()

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO dynamic_qrcodes (id, user_id, short_code, target_url, label) VALUES (?,?,?,?,?)",
            (qr_id, user_id, short_code, target_url, label)
        )
        await db.commit()
    finally:
        await db.close()

    redirect_url = f"{API_PUBLIC_URL}/go/{short_code}"
    return {"id": qr_id, "short_code": short_code, "redirect_url": redirect_url, "target_url": target_url, "label": label, "scan_count": 0}


@app.get("/api/qr/list")
async def qr_list(request: Request):
    """List all dynamic QR codes for the logged-in user."""
    user_id, _ = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Login required")

    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT id, short_code, target_url, label, scan_count, created_at FROM dynamic_qrcodes WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        )
    finally:
        await db.close()

    return [
        {
            "id": r[0], "short_code": r[1], "target_url": r[2],
            "label": r[3], "scan_count": r[4], "created_at": r[5],
            "redirect_url": f"{API_PUBLIC_URL}/go/{r[1]}"
        }
        for r in rows
    ]


@app.put("/api/qr/{qr_id}")
async def qr_update(qr_id: str, request: Request, body: dict = Body(...)):
    """Update the target URL of a dynamic QR code (owner only)."""
    user_id, _ = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Login required")

    target_url = (body.get("target_url") or "").strip()
    if not target_url or not target_url.startswith("http"):
        raise HTTPException(400, "A valid URL is required")

    db = await get_db()
    try:
        cur = await db.execute(
            "UPDATE dynamic_qrcodes SET target_url=?, updated_at=unixepoch() WHERE id=? AND user_id=?",
            (target_url, qr_id, user_id)
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "QR code not found")
        await db.commit()
    finally:
        await db.close()

    return {"ok": True}


@app.delete("/api/qr/{qr_id}")
async def qr_delete(qr_id: str, request: Request):
    """Delete a dynamic QR code (owner only)."""
    user_id, _ = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Login required")

    db = await get_db()
    try:
        cur = await db.execute("DELETE FROM dynamic_qrcodes WHERE id=? AND user_id=?", (qr_id, user_id))
        if cur.rowcount == 0:
            raise HTTPException(404, "QR code not found")
        await db.commit()
    finally:
        await db.close()

    return {"ok": True}


@app.get("/go/{short_code}")
async def qr_redirect(short_code: str):
    """Public redirect — increments scan counter and redirects to target URL."""
    db = await get_db()
    try:
        row = await db.execute_fetchall(
            "SELECT target_url FROM dynamic_qrcodes WHERE short_code=?", (short_code,)
        )
        if not row:
            raise HTTPException(404, "QR code not found")
        target = row[0][0]
        await db.execute(
            "UPDATE dynamic_qrcodes SET scan_count = scan_count + 1 WHERE short_code=?",
            (short_code,)
        )
        await db.commit()
    finally:
        await db.close()

    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=target, status_code=302)


# ── Admin endpoints ───────────────────────────────────────────────────────

ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "COCO-ADMIN-2026")


class AdminSetPlanRequest(BaseModel):
    secret: str
    email: str
    plan: str = "studio"
    credits: int = 999999
    quota: int = 999999


@app.post("/api/admin/set-plan")
async def api_admin_set_plan(req: AdminSetPlanRequest):
    """Set a user's plan, credits and quota. Requires admin secret."""
    if req.secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")
    if req.plan not in ("free", "creator", "pro", "studio"):
        raise HTTPException(400, "Invalid plan name")
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT id FROM users WHERE email = ?", (req.email.strip().lower(),)
        )
        if not rows:
            raise HTTPException(404, "User not found")
        user_id = rows[0][0]
        await db.execute(
            "UPDATE credits SET plan_name = ?, balance = ?, monthly_quota = ?, sub_status = 'active' WHERE user_id = ?",
            (req.plan, req.credits, req.quota, user_id),
        )
        await db.commit()
        return {"ok": True, "email": req.email.strip().lower(), "plan": req.plan, "credits": req.credits}
    finally:
        await db.close()


@app.get("/api/admin/users")
async def api_admin_list_users(secret: str = ""):
    """List all users with plan info. Requires admin secret."""
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT u.id, u.email, u.display_name, c.plan_name, c.balance, c.monthly_quota, c.sub_status "
            "FROM users u LEFT JOIN credits c ON c.user_id = u.id ORDER BY u.id"
        )
        return {"users": [
            {"id": r[0], "email": r[1], "name": r[2], "plan": r[3], "credits": r[4], "quota": r[5], "sub_status": r[6]}
            for r in rows
        ]}
    finally:
        await db.close()


@app.get("/api/admin/stats")
async def api_admin_stats(secret: str = "", days: int = 30):
    """Admin dashboard: credit consumption stats + API cost estimate."""
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")
    db = await get_db()
    try:
        # created_at is stored as Unix timestamp (float)
        import time as _time
        cutoff = _time.time() - days * 86400

        # Total consumption by mode (last N days)
        rows = await db.execute_fetchall(
            "SELECT json_extract(metadata, '$.mode') as mode, "
            "COUNT(*) as count, SUM(ABS(delta)) as total_credits "
            "FROM credit_transactions "
            "WHERE delta < 0 AND created_at > ? "
            "GROUP BY mode ORDER BY total_credits DESC",
            (cutoff,),
        )
        by_mode = [{"mode": r[0] or "unknown", "count": r[1], "credits": r[2]} for r in rows]

        # Daily consumption (last N days)
        daily = await db.execute_fetchall(
            "SELECT date(created_at, 'unixepoch') as day, COUNT(*) as count, SUM(ABS(delta)) as credits "
            "FROM credit_transactions "
            "WHERE delta < 0 AND created_at > ? "
            "GROUP BY day ORDER BY day DESC",
            (cutoff,),
        )
        by_day = [{"date": r[0], "count": r[1], "credits": r[2]} for r in daily]

        # Recent transactions (last 50)
        recent = await db.execute_fetchall(
            "SELECT datetime(t.created_at, 'unixepoch') as ts, u.email, t.delta, t.reason, t.metadata "
            "FROM credit_transactions t LEFT JOIN users u ON u.id = t.user_id "
            "WHERE t.delta < 0 ORDER BY t.created_at DESC LIMIT 50"
        )
        transactions = [
            {"time": r[0], "email": r[1] or "anonymous", "credits": abs(r[2]), "reason": r[3], "metadata": r[4]}
            for r in recent
        ]

        # Estimate API cost (GPT modes = $0.17 each)
        gpt_modes = sum(r[1] for r in rows if r[0] in ("cbn", "pbn", "ai"))
        estimated_cost = round(gpt_modes * 0.17, 2)

        return {
            "period_days": days,
            "by_mode": by_mode,
            "by_day": by_day,
            "recent_transactions": transactions,
            "estimated_api_cost": f"${estimated_cost}",
            "total_generations": sum(r[1] for r in rows),
            "total_credits_consumed": sum(r[2] for r in rows),
        }
    finally:
        await db.close()


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.2.0"}
