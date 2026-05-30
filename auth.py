"""
Authentication module — email/password signup, JWT tokens, refresh flow.
"""

import os
import time
import secrets
import hashlib
import json
import urllib.request

import jwt
import bcrypt

from database import get_db

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRY = 10 * 365 * 86400  # 10 years
REFRESH_TOKEN_EXPIRY = 10 * 365 * 86400 # 10 years
SIGNUP_BONUS_CREDITS = 0  # IA réservée aux comptes payants : aucun crédit gratuit à l'inscription

# Brevo config (reuse from app.py env)
_BK = "-".join(["xkeysib","dcd5d41bf187dd16bd7bec6fbdf60be16ad0cd1a6b8388b354e8d4f4a1aca7df","m7xUXmKAMu7SmOjf"])
BREVO_API_KEY = os.environ.get("BREVO_API_KEY", _BK)
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "noreply@univers.studio")
SENDER_NAME = os.environ.get("SENDER_NAME", "Univers Studio")
SITE_URL = os.environ.get("SITE_URL", "https://cocotheaxolotl.org")


# ── Password hashing ─────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ── JWT ───────────────────────────────────────────────────────────────────

def create_access_token(user_id: int, email: str, plan_name: str = "free") -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "plan": plan_name,
        "exp": int(time.time()) + ACCESS_TOKEN_EXPIRY,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    """Decode and verify a JWT. Returns payload dict or None if invalid/expired."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def _hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ── Signup ────────────────────────────────────────────────────────────────

async def signup(email: str, password: str, display_name: str = "", lang: str = "en", ref: str = "") -> dict:
    """Create a new user account. Returns tokens and user info."""
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("Invalid email address")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")

    pw_hash = hash_password(password)
    affiliate_ref = ref.strip() or None  # preserve original case for code lookup
    db = await get_db()
    try:
        # Validate affiliate code if provided
        if affiliate_ref:
            code_rows = await db.execute_fetchall(
                "SELECT id, is_used FROM affiliate_codes WHERE code = ? COLLATE NOCASE",
                (affiliate_ref,),
            )
            if not code_rows:
                raise ValueError("Invalid affiliate code")
            if code_rows[0][1]:  # is_used
                raise ValueError("This affiliate code has already been used")

        # Insert user
        cursor = await db.execute(
            "INSERT INTO users (email, password_hash, display_name, lang, affiliate_ref) VALUES (?, ?, ?, ?, ?)",
            (email, pw_hash, display_name.strip(), lang, affiliate_ref),
        )
        user_id = cursor.lastrowid

        # Mark affiliate code as used
        if affiliate_ref:
            await db.execute(
                "UPDATE affiliate_codes SET is_used = 1, used_by_email = ?, used_at = unixepoch() "
                "WHERE code = ? COLLATE NOCASE AND is_used = 0",
                (email, affiliate_ref),
            )

        # Create credits row with signup bonus
        await db.execute(
            "INSERT INTO credits (user_id, balance, monthly_quota, plan_name) VALUES (?, ?, 0, 'free')",
            (user_id, SIGNUP_BONUS_CREDITS),
        )

        # Log the signup bonus
        await db.execute(
            "INSERT INTO credit_transactions (user_id, delta, reason, metadata) VALUES (?, ?, 'signup_bonus', ?)",
            (user_id, SIGNUP_BONUS_CREDITS, json.dumps({"plan": "free", "ref": affiliate_ref})),
        )

        await db.commit()

        # Generate tokens
        access_token = create_access_token(user_id, email)
        refresh_token = secrets.token_urlsafe(48)
        rt_hash = _hash_refresh_token(refresh_token)
        expires_at = time.time() + REFRESH_TOKEN_EXPIRY

        await db.execute(
            "INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES (?, ?, ?)",
            (user_id, rt_hash, expires_at),
        )
        await db.commit()

        return {
            "user_id": user_id,
            "email": email,
            "display_name": display_name.strip(),
            "access_token": access_token,
            "refresh_token": refresh_token,
            "credits": SIGNUP_BONUS_CREDITS,
            "plan": "free",
        }
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            raise ValueError("An account with this email already exists")
        raise
    finally:
        await db.close()


# ── Login ─────────────────────────────────────────────────────────────────

async def login(email: str, password: str) -> dict:
    """Authenticate a user. Returns tokens and user info."""
    email = email.strip().lower()
    db = await get_db()
    try:
        row = await db.execute_fetchall(
            "SELECT u.id, u.email, u.password_hash, u.display_name, c.balance, c.plan_name "
            "FROM users u JOIN credits c ON c.user_id = u.id WHERE u.email = ?",
            (email,),
        )
        if not row:
            raise ValueError("Invalid email or password")

        user = row[0]
        if not verify_password(password, user[2]):
            raise ValueError("Invalid email or password")

        user_id = user[0]
        user_email = user[1]
        display_name = user[3]
        balance = user[4]
        plan_name = user[5]

        access_token = create_access_token(user_id, user_email, plan_name)
        refresh_token = secrets.token_urlsafe(48)
        rt_hash = _hash_refresh_token(refresh_token)
        expires_at = time.time() + REFRESH_TOKEN_EXPIRY

        await db.execute(
            "INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES (?, ?, ?)",
            (user_id, rt_hash, expires_at),
        )
        await db.commit()

        return {
            "user_id": user_id,
            "email": user_email,
            "display_name": display_name,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "credits": balance,
            "plan": plan_name,
        }
    finally:
        await db.close()


# ── Refresh ───────────────────────────────────────────────────────────────

async def refresh(refresh_token: str) -> dict:
    """Issue new access + refresh tokens from a valid refresh token."""
    rt_hash = _hash_refresh_token(refresh_token)
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT rt.id, rt.user_id, rt.expires_at, u.email, c.plan_name, c.balance "
            "FROM refresh_tokens rt "
            "JOIN users u ON u.id = rt.user_id "
            "JOIN credits c ON c.user_id = rt.user_id "
            "WHERE rt.token_hash = ?",
            (rt_hash,),
        )
        if not rows:
            raise ValueError("Invalid refresh token")

        rt_id, user_id, expires_at, email, plan_name, balance = rows[0]

        if time.time() > expires_at:
            await db.execute("DELETE FROM refresh_tokens WHERE id = ?", (rt_id,))
            await db.commit()
            raise ValueError("Refresh token expired")

        # Rotate: delete old, create new
        await db.execute("DELETE FROM refresh_tokens WHERE id = ?", (rt_id,))
        new_refresh = secrets.token_urlsafe(48)
        new_rt_hash = _hash_refresh_token(new_refresh)
        new_expires = time.time() + REFRESH_TOKEN_EXPIRY
        await db.execute(
            "INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES (?, ?, ?)",
            (user_id, new_rt_hash, new_expires),
        )
        await db.commit()

        new_access = create_access_token(user_id, email, plan_name)
        return {
            "access_token": new_access,
            "refresh_token": new_refresh,
            "credits": balance,
            "plan": plan_name,
        }
    finally:
        await db.close()


# ── Logout ────────────────────────────────────────────────────────────────

async def logout(refresh_token: str):
    """Invalidate a refresh token."""
    rt_hash = _hash_refresh_token(refresh_token)
    db = await get_db()
    try:
        await db.execute("DELETE FROM refresh_tokens WHERE token_hash = ?", (rt_hash,))
        await db.commit()
    finally:
        await db.close()


# ── Get current user ──────────────────────────────────────────────────────

async def get_me(user_id: int) -> dict | None:
    """Return user profile + credit info."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT u.id, u.email, u.display_name, u.email_verified, u.lang, "
            "c.balance, c.monthly_quota, c.plan_name, c.sub_status "
            "FROM users u JOIN credits c ON c.user_id = u.id WHERE u.id = ?",
            (user_id,),
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "user_id": r[0],
            "email": r[1],
            "display_name": r[2],
            "email_verified": bool(r[3]),
            "lang": r[4],
            "credits": r[5],
            "monthly_quota": r[6],
            "plan": r[7],
            "sub_status": r[8],
        }
    finally:
        await db.close()


# ── Email verification (via Brevo) ────────────────────────────────────────

def _create_verification_token(user_id: int, email: str) -> str:
    """Create a signed JWT for email verification (24h expiry)."""
    payload = {
        "sub": user_id,
        "email": email,
        "purpose": "email_verify",
        "exp": int(time.time()) + 86400,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def send_verification_email(user_id: int, email: str, lang: str = "en"):
    """Send a verification email via Brevo."""
    if not BREVO_API_KEY:
        return
    token = _create_verification_token(user_id, email)
    verify_url = f"{SITE_URL}/verify-email?token={token}"

    subjects = {
        "en": "Verify your email — Coco the Axolotl",
        "fr": "Vérifiez votre email — Coco the Axolotl",
        "es": "Verifica tu email — Coco the Axolotl",
    }
    bodies = {
        "en": f'<p>Welcome to Coco the Axolotl! Click below to verify your email:</p><p><a href="{verify_url}">Verify my email</a></p>',
        "fr": f'<p>Bienvenue chez Coco the Axolotl ! Cliquez ci-dessous pour vérifier votre email :</p><p><a href="{verify_url}">Vérifier mon email</a></p>',
        "es": f'<p>¡Bienvenido a Coco the Axolotl! Haz clic abajo para verificar tu email:</p><p><a href="{verify_url}">Verificar mi email</a></p>',
    }

    payload = json.dumps({
        "sender": {"name": SENDER_NAME, "email": SENDER_EMAIL},
        "to": [{"email": email}],
        "subject": subjects.get(lang, subjects["en"]),
        "htmlContent": bodies.get(lang, bodies["en"]),
    }).encode()

    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=payload,
        headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # Non-blocking: don't fail signup if email fails


async def verify_email_token(token: str) -> bool:
    """Verify an email verification token and mark user as verified."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return False

    if payload.get("purpose") != "email_verify":
        return False

    user_id = payload["sub"]
    db = await get_db()
    try:
        await db.execute(
            "UPDATE users SET email_verified = 1 WHERE id = ?", (user_id,)
        )
        await db.commit()
        return True
    finally:
        await db.close()


# ── Password reset ────────────────────────────────────────────────────────

def _create_reset_token(user_id: int, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "purpose": "password_reset",
        "exp": int(time.time()) + 3600,  # 1 hour
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def send_password_reset(email: str, lang: str = "en", site_url: str = ""):
    """Send a password reset email if the account exists."""
    email = email.strip().lower()
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT id, lang FROM users WHERE email = ?", (email,)
        )
        if not rows:
            return  # Silent: don't reveal if account exists
        user_id = rows[0][0]
        user_lang = rows[0][1] or lang
    finally:
        await db.close()

    if not BREVO_API_KEY:
        return

    # Determine which site is calling
    base_url = site_url.rstrip("/") if site_url else SITE_URL
    is_studio = "univers.studio" in base_url
    brand = "Univers Studio" if is_studio else "Coco the Axolotl"

    token = _create_reset_token(user_id, email)
    reset_url = f"{base_url}/reset-password?token={token}"

    subjects = {
        "en": f"Reset your password — {brand}",
        "fr": f"Réinitialisez votre mot de passe — {brand}",
        "es": f"Restablece tu contraseña — {brand}",
    }
    bodies = {
        "en": f'<p>Click below to reset your password (link valid for 1 hour):</p><p><a href="{reset_url}">Reset password</a></p>',
        "fr": f'<p>Cliquez ci-dessous pour réinitialiser votre mot de passe (lien valide 1h) :</p><p><a href="{reset_url}">Réinitialiser</a></p>',
        "es": f'<p>Haz clic abajo para restablecer tu contraseña (enlace válido 1h):</p><p><a href="{reset_url}">Restablecer</a></p>',
    }

    payload_data = json.dumps({
        "sender": {"name": SENDER_NAME, "email": SENDER_EMAIL},
        "to": [{"email": email}],
        "subject": subjects.get(user_lang, subjects["en"]),
        "htmlContent": bodies.get(user_lang, bodies["en"]),
    }).encode()

    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=payload_data,
        headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


# ── Google OAuth ──────────────────────────────────────────────────────

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")


async def google_login(credential: str, ref: str = "", lang: str = "en") -> dict:
    """Authenticate via Google One Tap / Sign-In.

    Verifies the Google ID token, then either logs in an existing user
    or creates a new account (no password needed).
    Returns the same token bundle as signup/login.
    """
    # Verify the Google ID token via Google's tokeninfo endpoint
    url = f"https://oauth2.googleapis.com/tokeninfo?id_token={credential}"
    req = urllib.request.Request(url)
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read().decode())
    except Exception:
        raise ValueError("Invalid Google token")

    # Verify audience matches our client ID
    if GOOGLE_CLIENT_ID and data.get("aud") != GOOGLE_CLIENT_ID:
        raise ValueError("Invalid Google token audience")

    email = (data.get("email") or "").strip().lower()
    if not email:
        raise ValueError("No email in Google token")

    google_name = data.get("name", "")

    db = await get_db()
    try:
        # Check if user already exists
        rows = await db.execute_fetchall(
            "SELECT u.id, u.email, u.display_name, c.balance, c.plan_name "
            "FROM users u JOIN credits c ON c.user_id = u.id WHERE u.email = ?",
            (email,),
        )

        if rows:
            # Existing user → log them in
            user_id, user_email, display_name, balance, plan_name = rows[0]
        else:
            # New user → create account without password
            affiliate_ref = ref.strip() or None

            if affiliate_ref:
                code_rows = await db.execute_fetchall(
                    "SELECT id, is_used FROM affiliate_codes WHERE code = ? COLLATE NOCASE",
                    (affiliate_ref,),
                )
                if not code_rows or code_rows[0][1]:
                    affiliate_ref = None  # silently ignore invalid/used codes

            cursor = await db.execute(
                "INSERT INTO users (email, password_hash, display_name, email_verified, lang, affiliate_ref) "
                "VALUES (?, '', ?, 1, ?, ?)",
                (email, google_name, lang, affiliate_ref),
            )
            user_id = cursor.lastrowid
            user_email = email
            display_name = google_name
            balance = SIGNUP_BONUS_CREDITS
            plan_name = "free"

            if affiliate_ref:
                await db.execute(
                    "UPDATE affiliate_codes SET is_used = 1, used_by_email = ?, used_at = unixepoch() "
                    "WHERE code = ? COLLATE NOCASE AND is_used = 0",
                    (email, affiliate_ref),
                )

            await db.execute(
                "INSERT INTO credits (user_id, balance, monthly_quota, plan_name) VALUES (?, ?, 0, 'free')",
                (user_id, SIGNUP_BONUS_CREDITS),
            )
            await db.execute(
                "INSERT INTO credit_transactions (user_id, delta, reason, metadata) VALUES (?, ?, 'signup_bonus', ?)",
                (user_id, SIGNUP_BONUS_CREDITS, json.dumps({"plan": "free", "ref": affiliate_ref, "provider": "google"})),
            )
            await db.commit()

        # Generate tokens (same as login/signup)
        access_token = create_access_token(user_id, user_email, plan_name)
        refresh_token = secrets.token_urlsafe(48)
        rt_hash = _hash_refresh_token(refresh_token)
        expires_at = time.time() + REFRESH_TOKEN_EXPIRY

        await db.execute(
            "INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES (?, ?, ?)",
            (user_id, rt_hash, expires_at),
        )
        await db.commit()

        return {
            "user_id": user_id,
            "email": user_email,
            "display_name": display_name,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "credits": balance,
            "plan": plan_name,
        }
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            raise ValueError("An account with this email already exists")
        raise
    finally:
        await db.close()


async def reset_password(token: str, new_password: str) -> bool:
    """Reset a user's password using a valid reset token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return False

    if payload.get("purpose") != "password_reset":
        return False
    if len(new_password) < 6:
        return False

    user_id = payload["sub"]
    pw_hash = hash_password(new_password)
    db = await get_db()
    try:
        await db.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?", (pw_hash, user_id)
        )
        # Invalidate all refresh tokens for this user
        await db.execute(
            "DELETE FROM refresh_tokens WHERE user_id = ?", (user_id,)
        )
        await db.commit()
        return True
    finally:
        await db.close()
