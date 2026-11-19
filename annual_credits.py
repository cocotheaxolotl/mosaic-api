"""Monthly entitlements for paid annual subscriptions, independent of invoices.

The first deployment records a permanent cutoff. Earlier missing periods are
reported as pending_review and NEVER credited by this automatic worker.
"""
import asyncio
import calendar
import json
import logging
import time
from datetime import datetime, timezone

from database import get_db

log = logging.getLogger(__name__)
POLL_SECONDS = 300


def month_after(anchor: float, months: int) -> int:
    dt = datetime.fromtimestamp(anchor, timezone.utc)
    year, month = divmod(dt.year * 12 + dt.month - 1 + months, 12)
    month += 1
    return int(dt.replace(year=year, month=month,
                          day=min(dt.day, calendar.monthrange(year, month)[1])).timestamp())


def annual_term(subscription: dict):
    items = (subscription.get("items") or {}).get("data") or []
    if len(items) != 1 or subscription.get("status") != "active":
        return None
    item = items[0]
    recurring = (item.get("price") or {}).get("recurring") or {}
    if recurring.get("interval") != "year" or recurring.get("interval_count", 1) != 1:
        return None
    invoice = subscription.get("latest_invoice")
    if not isinstance(invoice, dict) or invoice.get("status") != "paid":
        return None
    start = subscription.get("current_period_start") or item.get("current_period_start")
    end = subscription.get("current_period_end") or item.get("current_period_end")
    if not start or not end or end <= start:
        return None
    return int(start), int(end)


async def _protected_packs(db, user_id: int, balance: int) -> int:
    # Subscription credits are spent before purchased packs. Never expire packs
    # at a renewal; clamp to the real balance so legacy losses aren't backfilled.
    rows = await db.execute_fetchall(
        "SELECT delta, reason, metadata FROM credit_transactions WHERE user_id=? ORDER BY id",
        (user_id,),
    )
    subscription, packs = 0, 0
    for delta, reason, metadata in rows:
        if reason == "credit_pack_purchase":
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


async def process_subscription(user_id: int, subscription: dict, now: float | None = None) -> list[dict]:
    """Apply each future period once; quarantine pre-deployment missing periods."""
    now = time.time() if now is None else now
    term = annual_term(subscription)
    if term is None or not term[0] <= now < term[1]:
        return []
    sub_id = subscription["id"]
    start, end = term
    results = []
    db = await get_db()
    try:
        await db.execute("BEGIN IMMEDIATE")
        policy = await db.execute_fetchall("SELECT live_from FROM annual_credit_policy WHERE id=1")
        if not policy:
            raise RuntimeError("Annual credit policy not initialized")
        cutoff = policy[0][0]
        rows = await db.execute_fetchall(
            "SELECT c.balance,c.monthly_quota,c.plan_name,c.stripe_sub_id,c.sub_status,u.stripe_customer_id "
            "FROM credits c JOIN users u ON u.id=c.user_id WHERE c.user_id=?", (user_id,))
        if not rows:
            return []
        balance, quota, plan, current_sub, status, customer = rows[0]
        if (current_sub != sub_id or customer != subscription.get("customer")
                or status != "active" or plan not in ("creator", "pro", "studio") or quota <= 0):
            return []
        grants = await db.execute_fetchall(
            "SELECT created_at,metadata FROM credit_transactions WHERE user_id=? "
            "AND reason IN ('subscription_start','subscription_renewal')", (user_id,))
        for index in range(12):
            period_start = month_after(start, index)
            period_end = min(month_after(start, index + 1), end)
            if period_start > now or period_start >= end:
                break
            seen = await db.execute_fetchall(
                "SELECT status FROM annual_credit_periods WHERE stripe_sub_id=? AND period_start=?",
                (sub_id, period_start))
            if seen:
                continue
            allocated = False
            for created, raw in grants:
                try:
                    metadata = json.loads(raw or "{}")
                except (ValueError, TypeError):
                    metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}
                matches_period = (metadata["period_start"] == period_start if "period_start" in metadata
                                  else period_start <= created < period_end)
                if matches_period and metadata.get("stripe_sub_id", sub_id) == sub_id:
                    allocated = True
                    break
            before = balance
            if allocated:
                action = "already_granted"
            elif period_start < cutoff:
                action = "pending_review"
            elif period_end <= now and plan != "studio":
                action = "expired"
            else:
                action = "granted"
                balance = balance + quota if plan == "studio" else quota + await _protected_packs(db, user_id, balance)
                metadata = {"plan": plan, "stripe_sub_id": sub_id, "period_start": period_start,
                            "period_end": period_end, "source": "annual_monthly", "quota": quota,
                            "balance_before": before, "balance_after": balance}
                await db.execute("UPDATE credits SET balance=?,updated_at=? WHERE user_id=?",
                                 (balance, now, user_id))
                await db.execute(
                    "INSERT INTO credit_transactions(user_id,delta,reason,metadata,created_at) VALUES(?,?,?,?,?)",
                    (user_id, balance-before, "subscription_renewal", json.dumps(metadata), now))
            await db.execute(
                "INSERT INTO annual_credit_periods(stripe_sub_id,period_start,period_end,user_id,quota,status,balance_before,balance_after,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (sub_id,period_start,period_end,user_id,quota,action,before,balance,now))
            results.append({"user_id":user_id,"period_start":period_start,"status":action})
        await db.commit()
        return results
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()


async def run_once() -> list[dict]:
    # Import lazily to avoid a circular dependency with webhook handling.
    import stripe_integration
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT user_id,stripe_sub_id FROM credits WHERE sub_status='active' "
            "AND plan_name IN ('creator','pro','studio') AND stripe_sub_id IS NOT NULL")
    finally:
        await db.close()
    results = []
    for user_id, sub_id in rows:
        try:
            subscription = await asyncio.to_thread(
                stripe_integration.stripe.Subscription.retrieve, sub_id, expand=["latest_invoice"])
            results.extend(await process_subscription(user_id, subscription))
        except Exception:
            log.exception("Annual credit check failed for user_id=%s", user_id)
    return results


async def run_forever():
    while True:
        try:
            results = await run_once()
            if results:
                log.info("Annual credit periods: %s", results)
        except Exception:
            log.exception("Annual credit worker failed")
        await asyncio.sleep(POLL_SECONDS)
