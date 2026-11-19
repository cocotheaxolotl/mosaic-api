"""Owner-approved December gifts. No recipient is enrolled automatically."""
import json
from time import time as _now
from database import get_db
import credits
import tool_credits

CAMPAIGN_ID = "christmas-2026"
# Europe/Paris: December 1 00:00 through January 1 00:00 (exclusive).
STARTS_AT = 1796079600
ENDS_AT = 1798758000

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS credit_campaigns (
    id TEXT PRIMARY KEY,
    starts_at REAL NOT NULL,
    ends_at REAL NOT NULL CHECK(ends_at > starts_at),
    enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0,1)),
    approval_reference TEXT
);
CREATE TABLE IF NOT EXISTS credit_campaign_recipients (
    campaign_id TEXT NOT NULL REFERENCES credit_campaigns(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    email TEXT NOT NULL COLLATE NOCASE,
    plan_name TEXT NOT NULL,
    amount INTEGER NOT NULL CHECK(amount IN (15,25,100,300)),
    approved INTEGER NOT NULL DEFAULT 0 CHECK(approved IN (0,1)),
    claimed_at REAL,
    PRIMARY KEY(campaign_id,user_id)
);
INSERT OR IGNORE INTO credit_campaigns(id,starts_at,ends_at)
VALUES('christmas-2026',1796079600,1798758000);
"""


class GiftUnavailable(ValueError):
    pass


async def _status(db, user_id, now):
    rows = await db.execute_fetchall(
        "SELECT starts_at,ends_at,enabled,approval_reference FROM credit_campaigns WHERE id=?",
        (CAMPAIGN_ID,))
    if not rows:
        return {"active": False}
    start, end, enabled, approval = rows[0]
    active = bool(enabled and approval and start <= now < end)
    result = {"campaign": CAMPAIGN_ID, "active": active,
              "starts_at": start, "ends_at": end, "credit_type":"non_api"}
    if not active or user_id is None:
        return result
    rows = await db.execute_fetchall(
        "SELECT r.amount,r.claimed_at FROM credit_campaign_recipients r "
        "JOIN users u ON u.id=r.user_id WHERE r.campaign_id=? AND r.user_id=? "
        "AND r.approved=1 AND r.email=u.email COLLATE NOCASE",
        (CAMPAIGN_ID,user_id))
    result["eligible"] = bool(rows)
    if rows:
        result.update(amount=rows[0][0], claimed=rows[0][1] is not None)
    return result


async def get_status(user_id=None):
    db = await get_db()
    try:
        return await _status(db,user_id,_now())
    finally:
        await db.close()


async def claim(user_id):
    # Refuse unapproved/inactive requests before even resetting a daily quota.
    status = await get_status(user_id)
    if not status.get("eligible"):
        raise GiftUnavailable("Gift unavailable for this account")
    db = await get_db()
    try:
        await db.execute("BEGIN IMMEDIATE")
        now = _now()
        status = await _status(db,user_id,now)
        if not status.get("eligible"):
            raise GiftUnavailable("Gift unavailable for this account")
        rows = await db.execute_fetchall("SELECT 1 FROM credits WHERE user_id=?",(user_id,))
        if not rows:
            raise GiftUnavailable("Credit account not found")
        wallet = await db.execute_fetchall("SELECT balance FROM tool_credit_balances WHERE user_id=?",(user_id,))
        before = wallet[0][0] if wallet else 0
        operation = f"{CAMPAIGN_ID}:{user_id}"
        seen = await credits._operation_seen(db,operation)
        if status["claimed"] or seen:
            return {"ok":True,"already_claimed":True,"amount":status["amount"],"tool_credits":before,"credit_type":"non_api"}
        amount = status["amount"]
        after = before + amount
        await db.execute("INSERT INTO tool_credit_balances(user_id,balance) VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET balance=excluded.balance",(user_id,after))
        await db.execute(
            "INSERT INTO tool_credit_transactions(user_id,delta,reason,metadata) VALUES(?,?,'christmas_non_api_gift',?)",
            (user_id,amount,json.dumps({"campaign":CAMPAIGN_ID,"operation_key":operation,
                                       "balance_before":before,"balance_after":after})))
        await credits._record_operation(db,operation,user_id,"christmas_non_api_gift")
        await db.execute("UPDATE credit_campaign_recipients SET claimed_at=? WHERE campaign_id=? AND user_id=?",
                         (now,CAMPAIGN_ID,user_id))
        await db.commit()
        return {"ok":True,"already_claimed":False,"amount":amount,"tool_credits":after,"credit_type":"non_api"}
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()
