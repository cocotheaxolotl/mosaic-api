"""
Credits module — balance management, consumption, and transaction logging.
"""

import json
import time

from database import get_db


async def get_balance(user_id: int) -> int:
    """Return the current credit balance for a user."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT balance FROM credits WHERE user_id = ?", (user_id,)
        )
        return rows[0][0] if rows else 0
    finally:
        await db.close()


async def get_plan_info(user_id: int) -> dict | None:
    """Return full plan info for a user."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT balance, monthly_quota, plan_name, sub_status, stripe_sub_id "
            "FROM credits WHERE user_id = ?",
            (user_id,),
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "balance": r[0],
            "monthly_quota": r[1],
            "plan_name": r[2],
            "sub_status": r[3],
            "stripe_sub_id": r[4],
        }
    finally:
        await db.close()


async def consume_credits(
    user_id: int, amount: int, reason: str, metadata: dict | None = None
) -> tuple[bool, int]:
    """
    Consume credits atomically.
    Returns (success, remaining_balance).
    If insufficient credits, returns (False, current_balance) without consuming.
    """
    db = await get_db()
    try:
        # Check balance
        rows = await db.execute_fetchall(
            "SELECT balance FROM credits WHERE user_id = ?", (user_id,)
        )
        if not rows:
            return False, 0

        current = rows[0][0]
        if current < amount:
            return False, current

        new_balance = current - amount

        # Deduct
        await db.execute(
            "UPDATE credits SET balance = ?, updated_at = ? WHERE user_id = ?",
            (new_balance, time.time(), user_id),
        )

        # Log transaction
        await db.execute(
            "INSERT INTO credit_transactions (user_id, delta, reason, metadata) VALUES (?, ?, ?, ?)",
            (user_id, -amount, reason, json.dumps(metadata) if metadata else None),
        )
        await db.commit()
        return True, new_balance
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()


async def add_credits(
    user_id: int, amount: int, reason: str, metadata: dict | None = None
) -> int:
    """Add credits to a user's balance. Returns new balance."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT balance FROM credits WHERE user_id = ?", (user_id,)
        )
        current = rows[0][0] if rows else 0
        new_balance = current + amount

        await db.execute(
            "UPDATE credits SET balance = ?, updated_at = ? WHERE user_id = ?",
            (new_balance, time.time(), user_id),
        )
        await db.execute(
            "INSERT INTO credit_transactions (user_id, delta, reason, metadata) VALUES (?, ?, ?, ?)",
            (user_id, amount, reason, json.dumps(metadata) if metadata else None),
        )
        await db.commit()
        return new_balance
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()


async def set_plan(
    user_id: int,
    plan_name: str,
    monthly_quota: int,
    stripe_sub_id: str | None = None,
    sub_status: str = "active",
    current_period_end: float | None = None,
) -> int:
    """
    Set a user's plan and credit their account with the monthly quota.
    Returns new balance.
    """
    db = await get_db()
    try:
        await db.execute(
            "UPDATE credits SET plan_name = ?, monthly_quota = ?, balance = ?, "
            "stripe_sub_id = ?, sub_status = ?, current_period_end = ?, updated_at = ? "
            "WHERE user_id = ?",
            (plan_name, monthly_quota, monthly_quota, stripe_sub_id,
             sub_status, current_period_end, time.time(), user_id),
        )
        await db.execute(
            "INSERT INTO credit_transactions (user_id, delta, reason, metadata) VALUES (?, ?, ?, ?)",
            (user_id, monthly_quota, "subscription_start",
             json.dumps({"plan": plan_name, "stripe_sub_id": stripe_sub_id})),
        )
        await db.commit()
        return monthly_quota
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()


async def renew_credits(user_id: int) -> int:
    """Reset credits to monthly quota on subscription renewal. Returns new balance."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT monthly_quota, plan_name FROM credits WHERE user_id = ?",
            (user_id,),
        )
        if not rows:
            return 0
        quota = rows[0][0]
        plan = rows[0][1]

        await db.execute(
            "UPDATE credits SET balance = ?, updated_at = ? WHERE user_id = ?",
            (quota, time.time(), user_id),
        )
        await db.execute(
            "INSERT INTO credit_transactions (user_id, delta, reason, metadata) VALUES (?, ?, ?, ?)",
            (user_id, quota, "subscription_renewal", json.dumps({"plan": plan})),
        )
        await db.commit()
        return quota
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()


async def cancel_plan(user_id: int):
    """Mark a subscription as canceled. Don't remove remaining credits."""
    db = await get_db()
    try:
        await db.execute(
            "UPDATE credits SET sub_status = 'canceled', updated_at = ? WHERE user_id = ?",
            (time.time(), user_id),
        )
        await db.commit()
    finally:
        await db.close()


async def count_feature_usage(user_id: int, feature: str) -> int:
    """Count how many times a free user has used a specific feature."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT COUNT(*) FROM credit_transactions "
            "WHERE user_id = ? AND reason = 'generation' AND metadata LIKE ?",
            (user_id, f'%"feature": "{feature}"%'),
        )
        return rows[0][0] if rows else 0
    finally:
        await db.close()


async def record_feature_usage(user_id: int, feature: str, variant: str = "") -> None:
    """Record a feature use for free plan (no balance deduction)."""
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO credit_transactions (user_id, delta, reason, metadata) VALUES (?, 0, 'generation', ?)",
            (user_id, json.dumps({"feature": feature, "variant": variant})),
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()


async def get_transaction_history(user_id: int, limit: int = 50) -> list[dict]:
    """Return recent credit transactions for a user."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT delta, reason, metadata, created_at "
            "FROM credit_transactions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        return [
            {
                "delta": r[0],
                "reason": r[1],
                "metadata": json.loads(r[2]) if r[2] else None,
                "created_at": r[3],
            }
            for r in rows
        ]
    finally:
        await db.close()
