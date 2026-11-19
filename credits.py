"""
Credits module — balance management, consumption, and transaction logging.
"""

import json
import time
from datetime import datetime, timezone

from database import get_db

FREE_DAILY_QUOTA = 3


def _is_current_day(timestamp: float | None) -> bool:
    if not timestamp:
        return False
    current = datetime.now(timezone.utc)
    updated = datetime.fromtimestamp(timestamp, timezone.utc)
    return updated.date() == current.date()


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
    await reset_free_daily_quota_if_needed(user_id)
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


async def reset_free_daily_quota_if_needed(user_id: int) -> int | None:
    """Reset/repair Free daily credits at the first use of a new UTC day."""
    db = await get_db()
    try:
        await db.execute("BEGIN IMMEDIATE")
        rows = await db.execute_fetchall(
            "SELECT balance, monthly_quota, plan_name, updated_at FROM credits WHERE user_id = ?",
            (user_id,),
        )
        if not rows:
            return None

        balance, monthly_quota, plan_name, updated_at = rows[0]
        if plan_name != "free":
            return balance

        quota = monthly_quota if monthly_quota and monthly_quota > 0 else FREE_DAILY_QUOTA
        should_repair_quota = quota != monthly_quota
        should_reset_balance = not _is_current_day(updated_at)

        if not should_repair_quota and not should_reset_balance:
            return balance

        new_balance = quota if should_reset_balance else balance
        await db.execute(
            "UPDATE credits SET balance = ?, monthly_quota = ?, updated_at = ? WHERE user_id = ?",
            (new_balance, quota, time.time(), user_id),
        )
        if should_reset_balance:
            await db.execute(
                "INSERT INTO credit_transactions (user_id, delta, reason, metadata) VALUES (?, ?, ?, ?)",
                (user_id, quota, "free_daily_reset", json.dumps({"plan": "free", "quota": quota})),
            )
        await db.commit()
        return new_balance
    except Exception:
        await db.rollback()
        raise
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
        await db.execute("BEGIN IMMEDIATE")
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


async def _operation_seen(db, operation_key: str | None) -> bool:
    if not operation_key:
        return False
    return bool(await db.execute_fetchall(
        "SELECT 1 FROM credit_operations WHERE operation_key=?", (operation_key,)))


async def _record_operation(db, operation_key: str | None, user_id: int, kind: str):
    if operation_key:
        await db.execute("INSERT INTO credit_operations(operation_key,user_id,kind) VALUES(?,?,?)",
                         (operation_key,user_id,kind))


async def protected_extra_credits(db, user_id: int, balance: int) -> int:
    # Subscription credits are spent before purchased packs. Never expire packs
    # at a renewal; clamp to the real balance so legacy losses aren't backfilled.
    rows = await db.execute_fetchall(
        "SELECT delta, reason, metadata FROM credit_transactions WHERE user_id=? ORDER BY id",
        (user_id,),
    )
    subscription, packs = 0, 0
    for delta, reason, metadata in rows:
        if reason in ("credit_pack_purchase", "customer_service_gift"):
            packs += max(delta, 0)
        elif reason in ("subscription_start", "subscription_renewal", "free_daily_reset", "trial_activation", "signup_bonus"):
            try:
                info = json.loads(metadata or "{}")
            except (ValueError, TypeError):
                info = {}
            subscription = max(info.get("quota", delta), 0) if isinstance(info, dict) else max(delta, 0)
        elif delta < 0:
            used = min(subscription, -delta)
            subscription -= used
            packs = max(0, packs - (-delta - used))
    return min(packs, max(balance, 0))


async def add_credits(
    user_id: int, amount: int, reason: str, metadata: dict | None = None,
    operation_key: str | None = None,
) -> int:
    """Add credits atomically; a paid Checkout session can be fulfilled once."""
    db = await get_db()
    try:
        await db.execute("BEGIN IMMEDIATE")
        rows = await db.execute_fetchall("SELECT balance FROM credits WHERE user_id=?",(user_id,))
        if not rows:
            raise ValueError("Credit account not found")
        current = rows[0][0]
        if await _operation_seen(db,operation_key):
            return current
        new_balance = current + amount
        await db.execute("UPDATE credits SET balance=?,updated_at=? WHERE user_id=?",
                         (new_balance,time.time(),user_id))
        await db.execute("INSERT INTO credit_transactions(user_id,delta,reason,metadata) VALUES(?,?,?,?)",
                         (user_id,amount,reason,json.dumps(metadata) if metadata else None))
        await _record_operation(db,operation_key,user_id,reason)
        await db.commit()
        return new_balance
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()


async def set_plan(
    user_id: int, plan_name: str, monthly_quota: int,
    stripe_sub_id: str | None = None, sub_status: str = "active",
    current_period_end: float | None = None, operation_key: str | None = None,
) -> int:
    """Activate once, retaining purchased packs and approved service gifts."""
    db = await get_db()
    try:
        await db.execute("BEGIN IMMEDIATE")
        rows = await db.execute_fetchall("SELECT balance FROM credits WHERE user_id=?",(user_id,))
        if not rows:
            raise ValueError("Credit account not found")
        before=rows[0][0]
        if await _operation_seen(db,operation_key):
            return before
        # Legacy paid starts predate the operation ledger. A replay must not
        # overwrite credits already consumed, purchased or gifted since then.
        if operation_key and stripe_sub_id:
            starts=await db.execute_fetchall(
                "SELECT metadata FROM credit_transactions WHERE user_id=? AND reason='subscription_start'",(user_id,))
            for raw, in starts:
                try: info=json.loads(raw or "{}")
                except (TypeError,ValueError): continue
                if isinstance(info,dict) and info.get("stripe_sub_id")==stripe_sub_id:
                    await _record_operation(db,operation_key,user_id,"legacy_subscription_start")
                    await db.commit()
                    return before
        balance=monthly_quota+await protected_extra_credits(db,user_id,before)
        await db.execute(
            "UPDATE credits SET plan_name=?,monthly_quota=?,balance=?,stripe_sub_id=?,sub_status=?,current_period_end=?,updated_at=? WHERE user_id=?",
            (plan_name,monthly_quota,balance,stripe_sub_id,sub_status,current_period_end,time.time(),user_id))
        metadata={"plan":plan_name,"stripe_sub_id":stripe_sub_id,"quota":monthly_quota,
                  "balance_before":before,"balance_after":balance,"operation_key":operation_key}
        await db.execute("INSERT INTO credit_transactions(user_id,delta,reason,metadata) VALUES(?,?,?,?)",
                         (user_id,balance-before,"subscription_start",json.dumps(metadata)))
        await _record_operation(db,operation_key,user_id,"subscription_start")
        await db.commit()
        return balance
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()


async def renew_credits(
    user_id: int, operation_key: str | None = None,
    stripe_sub_id: str | None = None, period_start: float | None = None,
    period_end: float | None = None,
) -> int:
    """Renew once per paid invoice, preserving packs, gifts and Studio rollover."""
    db = await get_db()
    try:
        await db.execute("BEGIN IMMEDIATE")
        rows=await db.execute_fetchall(
            "SELECT monthly_quota,plan_name,balance,stripe_sub_id FROM credits WHERE user_id=?",(user_id,))
        if not rows:
            return 0
        quota,plan,before,current_sub=rows[0]
        if stripe_sub_id and stripe_sub_id!=current_sub:
            return before
        if await _operation_seen(db,operation_key):
            return before
        if operation_key and period_start and period_end:
            # Bootstrap invoices already fulfilled before this release, so a
            # delayed retry cannot reset the same paid month's balance again.
            previous=await db.execute_fetchall(
                "SELECT metadata FROM credit_transactions WHERE user_id=? AND reason='subscription_renewal' AND created_at>=? AND created_at<?",
                (user_id,period_start,period_end))
            for raw, in previous:
                try: info=json.loads(raw or "{}")
                except (TypeError,ValueError): continue
                if isinstance(info,dict) and info.get("stripe_sub_id",current_sub)==current_sub:
                    await _record_operation(db,operation_key,user_id,"legacy_subscription_renewal")
                    await db.commit()
                    return before
        balance=before+quota if plan=="studio" else quota+await protected_extra_credits(db,user_id,before)
        await db.execute("UPDATE credits SET balance=?,updated_at=? WHERE user_id=?",(balance,time.time(),user_id))
        metadata={"plan":plan,"stripe_sub_id":current_sub,"quota":quota,"balance_before":before,
                  "balance_after":balance,"operation_key":operation_key}
        if period_start:metadata["period_start"]=period_start
        if period_end:metadata["period_end"]=period_end
        await db.execute("INSERT INTO credit_transactions(user_id,delta,reason,metadata) VALUES(?,?,?,?)",
                         (user_id,balance-before,"subscription_renewal",json.dumps(metadata)))
        await _record_operation(db,operation_key,user_id,"subscription_renewal")
        await db.commit()
        return balance
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
