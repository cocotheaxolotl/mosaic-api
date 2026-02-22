"""
API Keys module — create, verify, list, and revoke API keys.

Keys use the format: us_live_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
(prefix "us_live_" + 40 hex chars). Only the SHA-256 hash is stored.
"""

import hashlib
import secrets
import time
import uuid

from database import get_db

API_KEY_PREFIX = "us_live_"


def _generate_raw_key() -> str:
    """Generate a new raw API key string."""
    return API_KEY_PREFIX + secrets.token_hex(20)


def _hash_key(raw_key: str) -> str:
    """SHA-256 hash of the raw key for storage."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def create_api_key(user_id: int, name: str = "") -> tuple[str, str]:
    """
    Create a new API key for a user.
    Returns (key_id, raw_key). The raw key is shown only once.
    Caller must verify that the user is on the commercial plan.
    """
    raw_key = _generate_raw_key()
    key_hash = _hash_key(raw_key)
    key_prefix = raw_key[:16]  # "us_live_" + 8 hex chars
    key_id = uuid.uuid4().hex

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO api_keys (id, user_id, key_hash, key_prefix, name) "
            "VALUES (?, ?, ?, ?, ?)",
            (key_id, user_id, key_hash, key_prefix, name.strip()[:100]),
        )
        await db.commit()
        return key_id, raw_key
    finally:
        await db.close()


async def verify_api_key(raw_key: str) -> int | None:
    """
    Verify an API key. Returns user_id if valid and active, None otherwise.
    Also updates last_used_at.
    """
    key_hash = _hash_key(raw_key)
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT id, user_id FROM api_keys WHERE key_hash = ? AND is_active = 1",
            (key_hash,),
        )
        if not rows:
            return None
        key_id, user_id = rows[0][0], rows[0][1]
        await db.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
            (time.time(), key_id),
        )
        await db.commit()
        return user_id
    finally:
        await db.close()


async def list_api_keys(user_id: int) -> list[dict]:
    """List all API keys for a user (does not reveal the full key)."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT id, key_prefix, name, created_at, last_used_at, is_active "
            "FROM api_keys WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        return [
            {
                "id": r[0],
                "key_prefix": r[1],
                "name": r[2],
                "created_at": r[3],
                "last_used_at": r[4],
                "is_active": bool(r[5]),
            }
            for r in rows
        ]
    finally:
        await db.close()


async def revoke_api_key(key_id: str, user_id: int) -> bool:
    """Revoke an API key. Returns True if found and revoked."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "UPDATE api_keys SET is_active = 0 WHERE id = ? AND user_id = ?",
            (key_id, user_id),
        )
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()
