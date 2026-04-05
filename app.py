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
    POST /api/auth/google       — sign in / sign up with Google
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
    generate_cbn_from_line_art, generate_smart_cbn, generate_from_preset, images_to_zip,
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
        "null",
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
    promo = _check_promo(promo_code)

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
        if promo:
            _apply_promo(request, promo_code)
        remaining = _check_quota(request, promo)
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
    elif not promo:
        _consume(request, cost)

    img_out = _add_watermark(ai_img) if (not user_id and not promo) else ai_img
    return _img_to_streaming(img_out, "ai-image.png")


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
                    try:
                        line_art_img = await _ai_coloring_page(data, hint=hint.strip(), style=cbn_style)
                        result = generate_cbn_from_line_art(
                            img,
                            line_art_img,
                            colors=colors,
                            page=page,
                            blur=2,
                            min_zone_pixels=150,
                        )
                        if result.zone_count <= 0:
                            raise ValueError("No closed zones from AI line art")
                        del line_art_img
                    except Exception:
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

class GoogleLoginRequest(BaseModel):
    credential: str
    ref: str = ""
    lang: str = "en"

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


@app.post("/api/auth/google")
async def api_google_login(req: GoogleLoginRequest):
    try:
        result = await auth_module.google_login(req.credential, req.ref, req.lang)
        return result
    except ValueError as e:
        raise HTTPException(401, str(e))


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

    # Check for expired trial — auto-downgrade to free
    import time as _time
    db = await get_db()
    try:
        trial_rows = await db.execute_fetchall(
            """SELECT tc.id FROM trial_codes tc
               JOIN credits c ON c.user_id = tc.used_by
               WHERE tc.used_by = ? AND tc.expires_at < ? AND c.sub_status = 'trial'""",
            (user_id, _time.time()),
        )
        if trial_rows:
            await db.execute(
                "UPDATE credits SET plan_name = 'free', balance = 3, monthly_quota = 3, sub_status = 'none' WHERE user_id = ?",
                (user_id,),
            )
            await db.execute(
                "INSERT INTO credit_transactions (user_id, delta, reason, metadata) VALUES (?, ?, ?, ?)",
                (user_id, 0, "trial_expired", "{}"),
            )
            await db.commit()
    finally:
        await db.close()

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


FREE_FEATURE_LIMIT = 3  # uses per tool for free plan

@app.post("/api/credits/consume")
async def api_credits_consume(request: Request, req: ConsumeRequest):
    """Consume 1 credit for a feature. Free plan: 3 uses per tool. Paid: deduct from balance."""
    user_id, _ = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Not authenticated")

    plan_info = await credits_module.get_plan_info(user_id)
    plan_name = (plan_info or {}).get("plan_name", "free")

    if plan_name == "free":
        feature = req.feature or "tool-download"
        used = await credits_module.count_feature_usage(user_id, feature)
        if used >= FREE_FEATURE_LIMIT:
            raise HTTPException(402, {
                "error": "Free limit reached",
                "remaining": 0,
                "upgrade_url": "https://univers.studio/pricing/",
            })
        await credits_module.record_feature_usage(user_id, feature, req.variant or "")
        return {"ok": True, "remaining": FREE_FEATURE_LIMIT - used - 1}

    success, remaining = await credits_module.consume_credits(
        user_id, 1, "generation",
        metadata={"feature": req.feature, "variant": req.variant},
    )
    if not success:
        raise HTTPException(402, {
            "error": "Insufficient credits",
            "remaining": remaining,
            "upgrade_url": "https://univers.studio/pricing/",
        })
    return {"ok": True, "remaining": remaining}


# ── Billing endpoints ─────────────────────────────────────────────────────

class CheckoutRequest(BaseModel):
    plan: str  # "creator" | "pro" | "studio" | "pack_20" | "pack_100"
    billing: str = "monthly"  # "monthly" | "annual" (ignored for packs)
    ref: str = ""
    currency: str = "usd"  # "usd" | "eur"


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
            user_id, profile["email"], req.plan, req.billing, req.ref, req.currency
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

    # Consume 3 credits
    ok, _bal = await credits_module.consume_credits(user_id, 3, "dynamic_qr", {"label": label})
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
AFFILIATE_NOTIFY_EMAIL = os.environ.get("AFFILIATE_NOTIFY_EMAIL", "4allsmilesllc@gmail.com")


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


# ── Trial codes (1 month Studio for influencers) ─────────────────────────

class TrialCodeGenerateRequest(BaseModel):
    secret: str
    name: str
    plan: str = "studio"
    credits: int = 1200
    duration_days: int = 30
    count: int = 1


@app.post("/api/admin/trial-codes")
async def api_admin_generate_trial_codes(req: TrialCodeGenerateRequest):
    """Generate trial codes for a specific influencer. Requires admin secret."""
    if req.secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")
    name = req.name.strip()[:100]
    if not name:
        raise HTTPException(400, "Name is required")

    import secrets as _secrets
    db = await get_db()
    codes = []
    try:
        for _ in range(req.count):
            prefix = name.split()[0].upper()[:5]
            suffix = _secrets.token_hex(3).upper()
            code = f"TRIAL-{prefix}-{suffix}"
            await db.execute(
                """INSERT INTO trial_codes (code, created_for, plan, credits, duration_days)
                   VALUES (?, ?, ?, ?, ?)""",
                (code, name, req.plan, req.credits, req.duration_days),
            )
            codes.append(code)
        await db.commit()
    finally:
        await db.close()
    return {"ok": True, "codes": codes, "for": name, "plan": req.plan, "duration_days": req.duration_days}


@app.get("/api/admin/trial-codes/list")
async def api_admin_list_trial_codes(secret: str = ""):
    """List all trial codes. Requires admin secret."""
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            """SELECT tc.code, tc.created_for, tc.plan, tc.credits, tc.duration_days,
                      tc.used_by, u.email AS used_by_email,
                      datetime(tc.used_at,'unixepoch') AS used_at,
                      datetime(tc.expires_at,'unixepoch') AS expires_at,
                      datetime(tc.created_at,'unixepoch') AS created_at
               FROM trial_codes tc
               LEFT JOIN users u ON u.id = tc.used_by
               ORDER BY tc.created_at DESC"""
        )
        return {"codes": [
            {"code": r[0], "for": r[1], "plan": r[2], "credits": r[3], "days": r[4],
             "used_by": r[6], "used_at": r[7], "expires_at": r[8], "created_at": r[9]}
            for r in rows
        ]}
    finally:
        await db.close()


class TrialRedeemRequest(BaseModel):
    code: str


@app.post("/api/trial/redeem")
async def api_trial_redeem(req: TrialRedeemRequest, request: Request):
    """Redeem a trial code. Requires JWT auth. Upgrades user to trial plan."""
    user_id, key = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Authentication required")

    code = req.code.strip().upper()
    if not code:
        raise HTTPException(400, "Code is required")

    import time as _time
    db = await get_db()
    try:
        # Check code exists and is unused
        rows = await db.execute_fetchall(
            "SELECT id, plan, credits, duration_days FROM trial_codes WHERE code = ? AND used_by IS NULL",
            (code,),
        )
        if not rows:
            raise HTTPException(400, "Invalid or already used code")

        tc_id, plan, credits, duration_days = rows[0][0], rows[0][1], rows[0][2], rows[0][3]
        now = _time.time()
        expires_at = now + duration_days * 86400

        # Check user doesn't already have an active paid subscription
        credit_rows = await db.execute_fetchall(
            "SELECT plan_name, sub_status FROM credits WHERE user_id = ?", (user_id,)
        )
        if credit_rows and credit_rows[0][1] == 'active' and credit_rows[0][0] != 'free':
            raise HTTPException(400, "You already have an active subscription")

        # Mark code as used
        await db.execute(
            "UPDATE trial_codes SET used_by = ?, used_at = ?, expires_at = ? WHERE id = ?",
            (user_id, now, expires_at, tc_id),
        )

        # Upgrade user plan
        await db.execute(
            "UPDATE credits SET plan_name = ?, balance = ?, monthly_quota = ?, sub_status = 'trial', updated_at = ? WHERE user_id = ?",
            (plan, credits, credits, now, user_id),
        )

        # Log transaction
        await db.execute(
            "INSERT INTO credit_transactions (user_id, delta, reason, metadata) VALUES (?, ?, ?, ?)",
            (user_id, credits, "trial_activation", json.dumps({"code": code, "plan": plan, "days": duration_days})),
        )

        await db.commit()
        return {"ok": True, "plan": plan, "credits": credits, "expires_at": expires_at, "days": duration_days}
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


@app.get("/api/admin/dashboard", response_class=HTMLResponse)
async def api_admin_dashboard(secret: str = "", days: int = 30):
    """Admin dashboard with readable HTML tables."""
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")
    stats = await api_admin_stats(secret=secret, days=days)
    bm = stats["by_mode"]
    bd = stats["by_day"]
    tx = stats["recent_transactions"]

    mode_rows = "".join(
        f"<tr><td>{m['mode']}</td><td>{m['count']}</td><td>{m['credits']}</td>"
        f"<td>{'$'+str(round(m['count']*0.17,2)) if m['mode'] in ('cbn','pbn','ai') else '-'}</td></tr>"
        for m in bm
    ) or "<tr><td colspan='4'>No data</td></tr>"

    day_rows = "".join(
        f"<tr><td>{d['date']}</td><td>{d['count']}</td><td>{d['credits']}</td></tr>"
        for d in bd
    ) or "<tr><td colspan='3'>No data</td></tr>"

    tx_rows = "".join(
        f"<tr><td>{t['time']}</td><td>{t['email']}</td><td>{t['credits']}</td><td>{t['reason']}</td>"
        f"<td>{t['metadata']}</td></tr>"
        for t in tx
    ) or "<tr><td colspan='5'>No data</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Admin Dashboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
h1 {{ color: #6a1b9a; margin-bottom: 5px; }}
.subtitle {{ color: #888; margin-bottom: 30px; }}
.cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 30px; }}
.card {{ background: #fff; border-radius: 12px; padding: 20px 28px; box-shadow: 0 2px 8px rgba(0,0,0,.08); min-width: 180px; }}
.card .value {{ font-size: 2em; font-weight: 700; color: #6a1b9a; }}
.card .label {{ font-size: .85em; color: #888; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-bottom: 30px; }}
th {{ background: #6a1b9a; color: #fff; padding: 12px 16px; text-align: left; font-weight: 600; }}
td {{ padding: 10px 16px; border-bottom: 1px solid #eee; }}
tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background: #faf5ff; }}
h2 {{ color: #6a1b9a; margin-top: 30px; }}
.period {{ display: inline-block; background: #ede7f6; color: #6a1b9a; padding: 4px 12px; border-radius: 20px; font-size: .85em; margin-left: 10px; }}
.period-btns {{ display: inline-flex; gap: 6px; margin-left: 16px; }}
.period-btns a {{ display: inline-block; padding: 5px 14px; border-radius: 20px; font-size: .85em; text-decoration: none; font-weight: 600; }}
.period-btns a.active {{ background: #6a1b9a; color: #fff; }}
.period-btns a:not(.active) {{ background: #e0e0e0; color: #555; }}
.period-btns a:not(.active):hover {{ background: #ce93d8; color: #fff; }}
</style></head><body>
<h1>Univers Studio — Admin Dashboard</h1>
<p class="subtitle">Credit consumption & API cost overview
<span class="period-btns">
<a href="?secret={secret}&days=7" class="{'active' if days == 7 else ''}">7 days</a>
<a href="?secret={secret}&days=30" class="{'active' if days == 30 else ''}">30 days</a>
<a href="?secret={secret}&days=90" class="{'active' if days == 90 else ''}">90 days</a>
</span></p>

<div class="cards">
  <div class="card"><div class="value">{stats['total_generations']}</div><div class="label">Generations</div></div>
  <div class="card"><div class="value">{stats['total_credits_consumed']}</div><div class="label">Credits consumed</div></div>
  <div class="card"><div class="value">{stats['estimated_api_cost']}</div><div class="label">Estimated API cost</div></div>
</div>

<h2>By Mode</h2>
<table>
<tr><th>Mode</th><th>Count</th><th>Credits</th><th>API Cost</th></tr>
{mode_rows}
</table>

<h2>By Day</h2>
<table>
<tr><th>Date</th><th>Count</th><th>Credits</th></tr>
{day_rows}
</table>

<h2>Recent Transactions (last 50)</h2>
<table>
<tr><th>Time</th><th>User</th><th>Credits</th><th>Reason</th><th>Details</th></tr>
{tx_rows}
</table>

</body></html>"""
    return HTMLResponse(content=html)


@app.get("/api/admin/affiliates", response_class=HTMLResponse)
async def api_admin_affiliates(secret: str = ""):
    """Affiliate dashboard — who brought whom, commissions owed."""
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")

    db = await get_db()
    try:
        # Summary per affiliate
        summary = await db.execute_fetchall("""
            SELECT
                ac.affiliate_email,
                ac.affiliate_name,
                ac.affiliate_code,
                COUNT(DISTINCT ac.customer_id) AS customers,
                SUM(ap.amount_usd) AS total_earned,
                SUM(CASE WHEN ap.status='pending' THEN ap.amount_usd ELSE 0 END) AS pending,
                SUM(CASE WHEN ap.status='paid' THEN ap.amount_usd ELSE 0 END) AS paid,
                ac.months_total,
                MIN(ac.months_paid) AS months_done,
                ac.status AS comm_status
            FROM affiliate_commissions ac
            LEFT JOIN affiliate_payouts ap ON ap.commission_id = ac.id
            GROUP BY ac.affiliate_email, ac.affiliate_code
            ORDER BY total_earned DESC NULLS LAST
        """)

        # Detail: who subscribed via each code
        customers = await db.execute_fetchall("""
            SELECT
                ac.affiliate_code,
                ac.customer_id,
                ac.plan,
                datetime(ac.started_at,'unixepoch') AS started,
                ac.months_paid,
                ac.months_total,
                ac.status,
                COALESCE(SUM(ap.amount_usd),0) AS earned
            FROM affiliate_commissions ac
            LEFT JOIN affiliate_payouts ap ON ap.commission_id = ac.id
            GROUP BY ac.id
            ORDER BY ac.started_at DESC
        """)

        # Pending applications
        apps = await db.execute_fetchall("""
            SELECT id, name, email, website, audience, datetime(created_at,'unixepoch'), status
            FROM affiliate_applications ORDER BY created_at DESC LIMIT 50
        """)
    finally:
        await db.close()

    def fmt(v):
        return f"${v:.2f}" if v else "$0.00"

    summary_rows = ""
    for r in summary:
        summary_rows += f"""<tr>
            <td><b>{r[1]}</b><br><small>{r[0]}</small></td>
            <td style="font-family:monospace;font-size:.9em">{r[2]}</td>
            <td style="text-align:center">{r[3]}</td>
            <td style="text-align:right;color:#16a34a"><b>{fmt(r[4])}</b></td>
            <td style="text-align:right;color:#d97706">{fmt(r[5])}</td>
            <td style="text-align:right;color:#6b7280">{fmt(r[6])}</td>
            <td style="text-align:center">{r[8]}/{r[7]} mois</td>
            <td><span style="background:{'#16a34a' if r[9]=='completed' else '#d97706'};color:#fff;padding:3px 10px;border-radius:4px;font-size:.8em;font-weight:600">{r[9]}</span></td>
        </tr>"""
    if not summary_rows:
        summary_rows = "<tr><td colspan='8' style='text-align:center;color:#9ca3af'>Aucune commission pour l'instant</td></tr>"

    # Group customers by code
    cust_by_code = {}
    for r in customers:
        cust_by_code.setdefault(r[0], []).append(r)

    cust_sections = ""
    for code, rows in cust_by_code.items():
        cust_sections += f"<h3 style='margin:24px 0 8px;font-size:.95em;color:#7c3aed'>Code : {code}</h3><table><tr><th>Customer ID</th><th>Plan</th><th>Début</th><th>Mois payés</th><th>Statut</th><th>Gagné</th></tr>"
        for r in rows:
            cust_sections += f"<tr><td style='font-size:.8em;font-family:monospace'>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td style='text-align:center'>{r[4]}/{r[5]}</td><td>{r[6]}</td><td style='color:#16a34a'>{fmt(r[7])}</td></tr>"
        cust_sections += "</table>"
    if not cust_sections:
        cust_sections = "<p style='color:#9ca3af'>Aucun client apporté pour l'instant.</p>"

    app_rows = ""
    for r in apps:
        if r[6] == "approved":
            badge = "<span style='background:#16a34a;color:#fff;padding:3px 10px;border-radius:4px;font-size:.8em;font-weight:600'>✓ approuvé</span>"
        else:
            badge = "<span style='background:#d97706;color:#fff;padding:3px 10px;border-radius:4px;font-size:.8em;font-weight:600'>en attente</span>"
        approve_btn = f' <a href="/api/admin/affiliate-applications/{r[0]}/approve?secret={secret}" style="background:#7c3aed;color:#fff;padding:3px 10px;border-radius:4px;text-decoration:none;font-size:.8em;font-weight:600">Approuver →</a>' if r[6]=="pending" else ""
        app_rows += f"<tr><td>{r[1]}<br><small style='color:#94a3b8'>{r[2]}</small></td><td><a href='{r[3]}'>{r[3]}</a></td><td>{r[4]}</td><td>{r[5]}</td><td>{badge}{approve_btn}</td></tr>"
    if not app_rows:
        app_rows = "<tr><td colspan='5' style='text-align:center;color:#9ca3af'>Aucune candidature</td></tr>"

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Affiliés — Admin</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:32px;max-width:1100px;margin:0 auto}}
h1{{color:#a78bfa;margin:0 0 8px}}
h2{{color:#c4b5fd;margin:32px 0 12px;font-size:1.1em;text-transform:uppercase;letter-spacing:.05em}}
h3{{color:#a78bfa}}
table{{width:100%;border-collapse:collapse;margin:0 0 16px;font-size:.88em}}
th{{background:#1e293b;padding:10px 12px;text-align:left;color:#94a3b8;font-weight:600;font-size:.78em;text-transform:uppercase}}
td{{padding:10px 12px;border-bottom:1px solid #1e293b}}
tr:hover td{{background:#1e293b}}
a{{color:#a78bfa}}
.nav{{margin:0 0 24px;font-size:.85em}}
.nav a{{color:#64748b;margin-right:16px;text-decoration:none}}
.nav a:hover{{color:#a78bfa}}
</style></head><body>
<h1>Programme Affiliés</h1>
<div class="nav">
  <a href="/api/admin/dashboard?secret={secret}">← Dashboard général</a>
  <a href="/api/admin/affiliates?secret={secret}">↺ Rafraîchir</a>
</div>

<h2>Résumé par affilié</h2>
<table>
<tr><th>Affilié</th><th>Code</th><th>Clients</th><th>Total gagné</th><th>En attente</th><th>Versé</th><th>Avancement</th><th>Statut</th></tr>
{summary_rows}
</table>

<h2>Clients apportés par code</h2>
{cust_sections}

<h2>Candidatures ({len(apps)})</h2>
<table>
<tr><th>Nom / Email</th><th>Site</th><th>Audience</th><th>Date</th><th>Statut</th></tr>
{app_rows}
</table>
</body></html>"""
    return HTMLResponse(content=html)


# ── Affiliate codes ──────────────────────────────────────────────────────

class AffiliateGenerateRequest(BaseModel):
    secret: str
    affiliate_name: str
    affiliate_email: str
    count: int = 10
    prefix: str = ""


@app.post("/api/admin/affiliate-codes")
async def api_admin_generate_affiliate_codes(req: AffiliateGenerateRequest):
    """Generate single-use affiliate codes for a partner. Requires admin secret."""
    if req.secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")
    if req.count < 1 or req.count > 500:
        raise HTTPException(400, "Count must be between 1 and 500")

    import string, random
    prefix = (req.prefix.strip().upper() or req.affiliate_name.strip().upper()[:4])
    codes = []
    db = await get_db()
    try:
        for _ in range(req.count):
            # Generate short unique code: PREFIX-XXXXXX
            suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
            code = f"{prefix}-{suffix}"
            try:
                await db.execute(
                    "INSERT INTO affiliate_codes (code, affiliate_name, affiliate_email) VALUES (?, ?, ?)",
                    (code, req.affiliate_name.strip(), req.affiliate_email.strip().lower()),
                )
                codes.append(code)
            except Exception:
                # Collision — retry with different suffix
                suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
                code = f"{prefix}-{suffix}"
                await db.execute(
                    "INSERT INTO affiliate_codes (code, affiliate_name, affiliate_email) VALUES (?, ?, ?)",
                    (code, req.affiliate_name.strip(), req.affiliate_email.strip().lower()),
                )
                codes.append(code)
        await db.commit()
        return {"ok": True, "affiliate": req.affiliate_name, "codes": codes, "count": len(codes)}
    finally:
        await db.close()


@app.get("/api/affiliate/check/{code}")
async def api_affiliate_check(code: str):
    """Check if an affiliate code is valid and unused. Public endpoint."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT code, affiliate_name, is_used FROM affiliate_codes WHERE code = ? COLLATE NOCASE",
            (code.strip(),),
        )
        if not rows:
            return {"valid": False, "reason": "Code not found"}
        row = rows[0]
        if row[2]:  # is_used
            return {"valid": False, "reason": "Code already used"}
        return {"valid": True, "affiliate": row[1]}
    finally:
        await db.close()


@app.get("/api/admin/affiliate-codes/list")
async def api_admin_list_affiliate_codes(secret: str = "", affiliate_email: str = ""):
    """List affiliate codes, optionally filtered by affiliate email. Requires admin secret."""
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")
    db = await get_db()
    try:
        if affiliate_email:
            rows = await db.execute_fetchall(
                "SELECT code, affiliate_name, affiliate_email, is_used, used_by_email, "
                "datetime(created_at, 'unixepoch') as created, datetime(used_at, 'unixepoch') as used "
                "FROM affiliate_codes WHERE affiliate_email = ? COLLATE NOCASE ORDER BY created_at DESC",
                (affiliate_email.strip().lower(),),
            )
        else:
            rows = await db.execute_fetchall(
                "SELECT code, affiliate_name, affiliate_email, is_used, used_by_email, "
                "datetime(created_at, 'unixepoch') as created, datetime(used_at, 'unixepoch') as used "
                "FROM affiliate_codes ORDER BY created_at DESC LIMIT 200"
            )
        return {"codes": [
            {"code": r[0], "affiliate": r[1], "email": r[2], "used": bool(r[3]),
             "used_by": r[4], "created": r[5], "used_at": r[6]}
            for r in rows
        ]}
    finally:
        await db.close()


@app.get("/api/admin/affiliate-commissions")
async def api_admin_affiliate_commissions(secret: str = "", affiliate_email: str = ""):
    """List affiliate commissions. Requires admin secret."""
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")
    db = await get_db()
    try:
        if affiliate_email:
            rows = await db.execute_fetchall(
                """SELECT id, affiliate_email, affiliate_name, affiliate_code, customer_id,
                   subscription_id, plan, datetime(started_at,'unixepoch') as started,
                   commission_rate, months_total, months_paid, status
                   FROM affiliate_commissions WHERE affiliate_email = ? COLLATE NOCASE
                   ORDER BY started_at DESC""",
                (affiliate_email.strip().lower(),),
            )
        else:
            rows = await db.execute_fetchall(
                """SELECT id, affiliate_email, affiliate_name, affiliate_code, customer_id,
                   subscription_id, plan, datetime(started_at,'unixepoch') as started,
                   commission_rate, months_total, months_paid, status
                   FROM affiliate_commissions ORDER BY started_at DESC LIMIT 200"""
            )
        return {"commissions": [
            {"id": r[0], "affiliate_email": r[1], "affiliate_name": r[2], "code": r[3],
             "customer_id": r[4], "subscription_id": r[5], "plan": r[6], "started": r[7],
             "rate": r[8], "months_total": r[9], "months_paid": r[10], "status": r[11]}
            for r in rows
        ]}
    finally:
        await db.close()


@app.get("/api/admin/affiliate-payouts")
async def api_admin_affiliate_payouts(secret: str = "", affiliate_email: str = "", status: str = ""):
    """List affiliate payouts. Requires admin secret."""
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")
    db = await get_db()
    try:
        conditions = []
        params = []
        if affiliate_email:
            conditions.append("p.affiliate_email = ? COLLATE NOCASE")
            params.append(affiliate_email.strip().lower())
        if status in ("pending", "paid"):
            conditions.append("p.status = ?")
            params.append(status)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        rows = await db.execute_fetchall(
            f"""SELECT p.id, p.affiliate_email, p.commission_id, p.month_number,
               p.amount_usd, p.invoice_id, p.status,
               datetime(p.created_at,'unixepoch') as created,
               datetime(p.paid_at,'unixepoch') as paid,
               c.affiliate_name
               FROM affiliate_payouts p
               LEFT JOIN affiliate_commissions c ON c.id = p.commission_id
               {where}
               ORDER BY p.created_at DESC LIMIT 500""",
            params,
        )
        return {"payouts": [
            {"id": r[0], "affiliate_email": r[1], "commission_id": r[2], "month": r[3],
             "amount_usd": r[4], "invoice_id": r[5], "status": r[6], "created": r[7],
             "paid_at": r[8], "affiliate_name": r[9]}
            for r in rows
        ]}
    finally:
        await db.close()


@app.post("/api/admin/affiliate-payouts/{payout_id}/mark-paid")
async def api_admin_mark_payout_paid(payout_id: int, secret: str = ""):
    """Mark a payout as paid. Requires admin secret."""
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")
    import time as _time
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT id, status FROM affiliate_payouts WHERE id = ?", (payout_id,)
        )
        if not rows:
            raise HTTPException(404, "Payout not found")
        if rows[0][1] == "paid":
            return {"ok": True, "message": "Already marked as paid"}
        await db.execute(
            "UPDATE affiliate_payouts SET status = 'paid', paid_at = ? WHERE id = ?",
            (_time.time(), payout_id),
        )
        await db.commit()
        return {"ok": True, "payout_id": payout_id}
    finally:
        await db.close()


# ── Affiliate application ────────────────────────────────────────────

class AffiliateApplyRequest(BaseModel):
    name: str
    email: str
    phone: str = ""
    website: str = ""
    audience: str = ""
    promotion: str = ""
    lang: str = "en"


# ── AI Writing Tool Models ──────────────────────────────────────────────

class TitleRequest(BaseModel):
    genre: str
    themes: str
    audience: str
    tone: str
    language: str = "en"
    count: int = 5

class StoryStructureRequest(BaseModel):
    genre: str
    premise: str
    protagonist: str
    setting: str = ""
    language: str = "en"

class BlurbRequest(BaseModel):
    title: str
    genre: str
    synopsis: str
    protagonist: str
    stakes: str = ""
    tone: str = "dramatic"
    language: str = "en"
    count: int = 3

# ── AI Writing Tool Rate Limiting ───────────────────────────────────────

_title_rate: dict[str, list] = {}   # ip -> [timestamps]
_story_rate: dict[str, list] = {}
_blurb_rate: dict[str, list] = {}

def _check_rate(store: dict, ip: str, max_per_hour: int):
    now = time.time()
    hits = store.get(ip, [])
    hits = [t for t in hits if now - t < 3600]
    if len(hits) >= max_per_hour:
        raise HTTPException(429, f"Rate limit: {max_per_hour} requests per hour")
    hits.append(now)
    store[ip] = hits


def _mask_email(email: str) -> str:
    """Mask email: j****n@g***l.com"""
    if not email or "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked_local = local[0] + "***"
    else:
        masked_local = local[0] + "*" * (len(local) - 2) + local[-1]
    parts = domain.rsplit(".", 1)
    if len(parts) == 2:
        d, ext = parts
        if len(d) <= 2:
            masked_domain = d[0] + "***." + ext
        else:
            masked_domain = d[0] + "*" * (len(d) - 2) + d[-1] + "." + ext
        return masked_local + "@" + masked_domain
    return masked_local + "@***"


@app.get("/api/affiliate/my-referrals")
async def api_affiliate_my_referrals(request: Request):
    """Affiliate dashboard data — returns referrals with masked emails. Requires JWT auth."""
    user_id, key = await get_user_or_ip(request)
    if not user_id:
        raise HTTPException(401, "Authentication required")

    db = await get_db()
    try:
        # Get current user's email
        user_rows = await db.execute_fetchall("SELECT email FROM users WHERE id = ?", (user_id,))
        if not user_rows:
            raise HTTPException(401, "User not found")
        user_email = user_rows[0][0]

        # Check if this user is an affiliate
        code_rows = await db.execute_fetchall(
            "SELECT code FROM affiliate_codes WHERE affiliate_email = ? COLLATE NOCASE LIMIT 1",
            (user_email,),
        )
        if not code_rows:
            return {"is_affiliate": False, "code": None, "referrals": [], "totals": {}}

        affiliate_code = code_rows[0][0]

        # Get commissions with customer emails (masked)
        referrals_raw = await db.execute_fetchall("""
            SELECT
                ac.id,
                u.email AS customer_email,
                ac.plan,
                datetime(ac.started_at, 'unixepoch') AS started,
                ac.months_paid,
                ac.months_total,
                ac.status,
                COALESCE(SUM(ap.amount_usd), 0) AS earned
            FROM affiliate_commissions ac
            LEFT JOIN users u ON u.stripe_customer_id = ac.customer_id
            LEFT JOIN affiliate_payouts ap ON ap.commission_id = ac.id
            WHERE ac.affiliate_code = ? COLLATE NOCASE
            GROUP BY ac.id
            ORDER BY ac.started_at DESC
        """, (affiliate_code,))

        referrals = []
        total_earned = 0.0
        total_pending = 0.0
        for r in referrals_raw:
            earned = r[7] or 0.0
            total_earned += earned
            referrals.append({
                "email": _mask_email(r[1]) if r[1] else "—",
                "plan": r[2],
                "started": r[3],
                "months_paid": r[4],
                "months_total": r[5],
                "status": r[6],
                "earned": round(earned, 2),
            })

        # Pending payouts
        payout_rows = await db.execute_fetchall("""
            SELECT COALESCE(SUM(amount_usd), 0)
            FROM affiliate_payouts
            WHERE affiliate_email = ? COLLATE NOCASE AND status = 'pending'
        """, (user_email,))
        total_pending = payout_rows[0][0] if payout_rows else 0.0

        return {
            "is_affiliate": True,
            "code": affiliate_code,
            "referrals": referrals,
            "totals": {
                "customers": len(referrals),
                "earned": round(total_earned, 2),
                "pending": round(total_pending, 2),
            },
        }
    finally:
        await db.close()


@app.post("/api/affiliate/apply")
async def api_affiliate_apply(req: AffiliateApplyRequest):
    """Store affiliate application in DB and notify admin by email."""
    name = req.name.strip()[:100]
    email = req.email.strip()[:200]
    if not name or not email:
        raise HTTPException(400, "Name and email are required.")

    # Store in database
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO affiliate_applications (name, email, phone, website, audience, promotion, lang)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, email, req.phone.strip()[:50], req.website.strip()[:300],
             req.audience.strip()[:300], req.promotion.strip()[:1000], req.lang or 'en'),
        )
        await db.commit()
        rows = await db.execute_fetchall("SELECT last_insert_rowid()")
        app_id = rows[0][0]
    finally:
        await db.close()

    # Notify admin
    approve_url = f"https://mosaic-api.fly.dev/api/admin/affiliate-applications/{app_id}/approve?secret={ADMIN_SECRET}"
    html = f"""
<h2>New Affiliate Application — Univers Studio</h2>
<table cellpadding="6" style="border-collapse:collapse;font-family:sans-serif;font-size:14px">
  <tr><td><b>ID</b></td><td>#{app_id}</td></tr>
  <tr><td><b>Name</b></td><td>{name}</td></tr>
  <tr><td><b>Email</b></td><td>{email}</td></tr>
  <tr><td><b>Phone</b></td><td>{req.phone.strip() or '—'}</td></tr>
  <tr><td><b>Website</b></td><td>{req.website.strip() or '—'}</td></tr>
  <tr><td><b>Audience</b></td><td>{req.audience.strip() or '—'}</td></tr>
  <tr><td><b>Promotion plan</b></td><td style="white-space:pre-wrap">{req.promotion.strip() or '—'}</td></tr>
</table>
<p style="margin-top:20px">
  <a href="{approve_url}" style="background:#7c3aed;color:#fff;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:bold;font-family:sans-serif">
    ✅ Approve &amp; Send Code
  </a>
</p>
<p style="font-size:12px;color:#666">Clicking approve will auto-generate an affiliate code and email it to {email}.</p>
"""
    try:
        payload = json.dumps({
            "sender": {"name": SENDER_NAME, "email": SENDER_EMAIL},
            "to": [{"email": AFFILIATE_NOTIFY_EMAIL}],
            "replyTo": {"email": email, "name": name},
            "subject": f"New Affiliate Application – {name}",
            "htmlContent": html,
        }, ensure_ascii=False).encode("utf-8")
        api_req = urllib.request.Request(
            "https://api.brevo.com/v3/smtp/email",
            data=payload,
            headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(api_req, timeout=10)
    except Exception:
        pass  # Don't fail if email fails — application is already stored

    return {"ok": True}


@app.get("/api/admin/affiliate-applications")
async def api_admin_list_applications(secret: str = "", status: str = "pending"):
    """List affiliate applications. Requires admin secret."""
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            """SELECT id, name, email, phone, website, audience, promotion, lang, status,
               datetime(created_at,'unixepoch') as created
               FROM affiliate_applications WHERE status = ? ORDER BY created_at DESC""",
            (status,),
        )
        return {"applications": [
            {"id": r[0], "name": r[1], "email": r[2], "phone": r[3], "website": r[4],
             "audience": r[5], "promotion": r[6], "lang": r[7], "status": r[8], "created": r[9]}
            for r in rows
        ]}
    finally:
        await db.close()


@app.get("/api/admin/affiliate-applications/{app_id}/approve")
async def api_admin_approve_application(app_id: int, secret: str = ""):
    """Approve an affiliate application: generate a code and email it to the applicant."""
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")

    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT id, name, email, lang, status FROM affiliate_applications WHERE id = ?",
            (app_id,),
        )
        if not rows:
            raise HTTPException(404, "Application not found")
        row = rows[0]
        if row[4] == 'approved':
            return HTMLResponse("<h2>Already approved.</h2>", status_code=200)

        name, email, lang = row[1], row[2], row[3]

        # Generate affiliate code
        import string, random as _random
        prefix = "".join(c for c in name.upper().split()[0][:6] if c.isalpha()) or "AFF"
        suffix = "".join(_random.choices(string.ascii_uppercase + string.digits, k=6))
        code = f"{prefix}-{suffix}"

        # Insert code into affiliate_codes
        await db.execute(
            "INSERT INTO affiliate_codes (code, affiliate_name, affiliate_email) VALUES (?, ?, ?)",
            (code, name, email.lower()),
        )
        await db.execute(
            "UPDATE affiliate_applications SET status = 'approved' WHERE id = ?",
            (app_id,),
        )
        await db.commit()
    finally:
        await db.close()

    # Email the code to the applicant
    if lang == 'fr':
        subject = "Votre code affilié Univers Studio"
        html_body = f"""
<h2>Bienvenue dans le programme affilié Univers Studio !</h2>
<p>Bonjour {name},</p>
<p>Votre candidature a été acceptée. Voici votre lien affilié personnel — <strong>c'est ce lien que vous partagez avec votre audience</strong> :</p>
<p style="text-align:center;margin:24px 0">
  <a href="https://univers.studio/pricing/?ref={code}" style="background:#7c3aed;color:#fff;padding:14px 28px;border-radius:10px;text-decoration:none;font-weight:bold;font-size:1.1em;font-family:sans-serif">
    👉 https://univers.studio/pricing/?ref={code}
  </a>
</p>
<p>Quand quelqu'un clique sur ce lien et souscrit un abonnement, le code <strong>{code}</strong> est automatiquement appliqué et vous touchez <strong>30% de commission</strong> sur ses 6 premiers mois.</p>
<p style="background:#f3f0ff;padding:16px;border-radius:8px;font-size:.9em">
  <strong>Comment partager ?</strong><br>
  Copiez ce lien et partagez-le dans vos vidéos, articles, newsletters, réseaux sociaux — partout où vous conseillez des outils à votre audience.<br><br>
  Vos abonnés n'ont rien à saisir : le code est appliqué automatiquement dès qu'ils arrivent sur la page via votre lien.
</p>
<p>Pour toute question : <a href="mailto:{AFFILIATE_NOTIFY_EMAIL}">{AFFILIATE_NOTIFY_EMAIL}</a></p>
<p>Bonne chance !<br>L'équipe Univers Studio</p>
"""
    else:
        subject = "Your Univers Studio Affiliate Code"
        html_body = f"""
<h2>Welcome to the Univers Studio Affiliate Program!</h2>
<p>Hi {name},</p>
<p>Your application has been approved. Here is your personal affiliate link — <strong>this is what you share with your audience</strong>:</p>
<p style="text-align:center;margin:24px 0">
  <a href="https://univers.studio/pricing/?ref={code}" style="background:#7c3aed;color:#fff;padding:14px 28px;border-radius:10px;text-decoration:none;font-weight:bold;font-size:1.1em;font-family:sans-serif">
    👉 https://univers.studio/pricing/?ref={code}
  </a>
</p>
<p>When someone clicks your link and subscribes, the code <strong>{code}</strong> is applied automatically and you earn <strong>30% commission</strong> on their first 6 months.</p>
<p style="background:#f3f0ff;padding:16px;border-radius:8px;font-size:.9em">
  <strong>How to share?</strong><br>
  Copy this link and share it in your videos, blog posts, newsletters, social media — anywhere you recommend tools to your audience.<br><br>
  Your followers don't need to type anything: the code is applied automatically as soon as they land on the page via your link.
</p>
<p>Questions? <a href="mailto:{AFFILIATE_NOTIFY_EMAIL}">{AFFILIATE_NOTIFY_EMAIL}</a></p>
<p>Good luck!<br>The Univers Studio Team</p>
"""
    try:
        payload = json.dumps({
            "sender": {"name": SENDER_NAME, "email": SENDER_EMAIL},
            "to": [{"email": email, "name": name}],
            "subject": subject,
            "htmlContent": html_body,
        }, ensure_ascii=False).encode("utf-8")
        api_req = urllib.request.Request(
            "https://api.brevo.com/v3/smtp/email",
            data=payload,
            headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(api_req, timeout=10)
    except Exception as e:
        raise HTTPException(500, f"Code generated ({code}) but failed to send email: {e}")

    return HTMLResponse(f"""
<html><body style="font-family:sans-serif;padding:40px;max-width:500px;margin:0 auto">
<h2 style="color:#16a34a">✅ Approved!</h2>
<p><b>{name}</b> ({email}) has been approved.</p>
<p>Affiliate code: <b style="color:#7c3aed;font-size:1.3em">{code}</b></p>
<p>An email with the code has been sent to {email}.</p>
</body></html>
""")


# ── Book Translation ─────────────────────────────────────────────────

@app.post("/api/translate-book")
async def api_translate_book(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form(...),
):
    """Translate a book file (PPTX, PDF, EPUB) to target language."""
    import translate as translate_module

    user_id, identifier = await get_user_or_ip(request)

    # Validate file type
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    type_map = {".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ".pdf": "application/pdf", ".epub": "application/epub+zip",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    if ext not in type_map:
        raise HTTPException(400, f"Unsupported file type: {ext}. Use .pptx, .pdf, .epub, or .docx")

    content_type = type_map[ext]

    # Validate language
    if language not in translate_module.LANGUAGES:
        raise HTTPException(400, f"Unsupported language: {language}")

    # Read file
    data = await file.read()
    if len(data) > 200_000_000:
        raise HTTPException(400, "File too large (max 200 MB)")

    # Translate (with semaphore to prevent OOM)
    async with _gen_semaphore:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, translate_module.translate_book, data, language, content_type
            )
        except Exception as e:
            raise HTTPException(500, f"Translation failed: {str(e)[:200]}")

    # Return translated file
    out_name = f"{Path(file.filename).stem}-{language}{ext}"
    return StreamingResponse(
        io.BytesIO(result),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


# ── AI Writing Tool Endpoints ───────────────────────────────────────────

@app.post("/api/title-generator")
async def api_title_generator(req: TitleRequest, request: Request):
    if not OPENAI_API_KEY:
        raise HTTPException(503, "AI service not configured")

    ip = request.client.host
    _check_rate(_title_rate, ip, 10)

    lang_instruction = "Respond entirely in French." if req.language == "fr" else "Respond entirely in English."

    prompt = f"""Generate {req.count} book title suggestions.
Genre: {req.genre}
Key themes: {req.themes}
Target audience: {req.audience}
Tone: {req.tone}
{lang_instruction}

Return a JSON object with a "titles" key containing an array of objects with "title", "subtitle", and "reasoning" fields.
Each title should be:
- Memorable and marketable
- Optimized for Amazon KDP search
- Appropriate for the genre and audience
- The subtitle should clarify the book's value proposition"""

    oai_req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a professional book title consultant specializing in Amazon KDP bestsellers. Generate compelling, marketable book titles with subtitles. Each title should be optimized for Amazon search discoverability while being memorable and genre-appropriate. Always respond with valid JSON only, no markdown."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.9,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"}
        }).encode(),
    )

    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(oai_req, timeout=30))
        body = json.loads(resp.read())
    except Exception as e:
        raise HTTPException(502, f"AI service error: {str(e)[:200]}")

    content = body["choices"][0]["message"]["content"]
    data = json.loads(content)
    tokens = body.get("usage", {}).get("total_tokens", 0)

    return {"titles": data.get("titles", data.get("suggestions", [])), "tokens_used": tokens}


@app.post("/api/story-structure")
async def api_story_structure(req: StoryStructureRequest, request: Request):
    if not OPENAI_API_KEY:
        raise HTTPException(503, "AI service not configured")

    ip = request.client.host
    _check_rate(_story_rate, ip, 5)

    lang_instruction = "Respond entirely in French." if req.language == "fr" else "Respond entirely in English."
    setting_line = f"Setting: {req.setting}" if req.setting else ""

    prompt = f"""Create a detailed Save the Cat beat sheet for the following novel concept.

Genre: {req.genre}
Premise: {req.premise}
Protagonist: {req.protagonist}
{setting_line}
{lang_instruction}

Return a JSON object with:
- "logline": a compelling one-sentence logline
- "beats": an array of 15 beat objects, each with:
  - "beat_name": the Save the Cat beat name
  - "description": brief description of what happens at this beat
  - "page_range": approximate page range (for a 300-page novel)
  - "details": 2-3 sentences of specific story detail for this beat

The 15 beats in order: Opening Image, Theme Stated, Setup, Catalyst, Debate, Break into Two, B Story, Fun and Games, Midpoint, Bad Guys Close In, All Is Lost, Dark Night of the Soul, Break into Three, Finale, Final Image."""

    oai_req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an expert story structure consultant specializing in the Save the Cat beat sheet method by Blake Snyder. You help authors plan compelling novels with strong narrative structure. Always respond with valid JSON only, no markdown."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.8,
            "max_tokens": 3000,
            "response_format": {"type": "json_object"}
        }).encode(),
    )

    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(oai_req, timeout=45))
        body = json.loads(resp.read())
    except Exception as e:
        raise HTTPException(502, f"AI service error: {str(e)[:200]}")

    content = body["choices"][0]["message"]["content"]
    data = json.loads(content)
    tokens = body.get("usage", {}).get("total_tokens", 0)

    return {"beats": data.get("beats", []), "logline": data.get("logline", ""), "tokens_used": tokens}


@app.post("/api/blurb-generator")
async def api_blurb_generator(req: BlurbRequest, request: Request):
    if not OPENAI_API_KEY:
        raise HTTPException(503, "AI service not configured")

    ip = request.client.host
    _check_rate(_blurb_rate, ip, 10)

    lang_instruction = "Respond entirely in French." if req.language == "fr" else "Respond entirely in English."
    stakes_line = f"Stakes: {req.stakes}" if req.stakes else ""

    prompt = f"""Generate {req.count} compelling back cover blurbs for the following fiction book.

Title: {req.title}
Genre: {req.genre}
Synopsis: {req.synopsis}
Protagonist: {req.protagonist}
{stakes_line}
Tone: {req.tone}
{lang_instruction}

Return a JSON object with a "blurbs" key containing an array of objects with:
- "text": the full blurb text (100-180 words each)
- "hook_type": the type of hook used (e.g. "question", "action", "mystery", "emotional", "provocative")

Each blurb should:
- Open with an irresistible hook
- Build intrigue and emotional tension
- End with a cliffhanger or compelling question that makes readers want to buy
- Be 100-180 words
- Use varied hook types across the different blurbs"""

    oai_req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an expert copywriter specializing in fiction back cover blurbs and book marketing copy. You craft irresistible blurbs that hook readers instantly and drive purchases. Always respond with valid JSON only, no markdown."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.9,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}
        }).encode(),
    )

    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(oai_req, timeout=30))
        body = json.loads(resp.read())
    except Exception as e:
        raise HTTPException(502, f"AI service error: {str(e)[:200]}")

    content = body["choices"][0]["message"]["content"]
    data = json.loads(content)
    tokens = body.get("usage", {}).get("total_tokens", 0)

    return {"blurbs": data.get("blurbs", []), "tokens_used": tokens}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.4.0"}
