"""Separate promotional wallet for browser tools. Never funds an AI provider."""
import json
import time
from database import get_db
import credits

# These routes export files locally. Paid generation endpoints never read this wallet.
BROWSER_TOOLS = frozenset({"maze","sudoku","wordsearch","crossword","bingo","cryptogram",
                           "handwriting","mathdrills","back-matter","book-template","qr-code"})
LOCALES = frozenset({"en","fr","es","de","it","pt","ja","ar","pl"})
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tool_credit_balances (
    user_id INTEGER PRIMARY KEY REFERENCES users(id),
    balance INTEGER NOT NULL DEFAULT 0 CHECK(balance>=0)
);
CREATE TABLE IF NOT EXISTS tool_credit_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    delta INTEGER NOT NULL,
    reason TEXT NOT NULL,
    metadata TEXT,
    created_at REAL NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_tool_txn_user ON tool_credit_transactions(user_id,id);
"""


def browser_tool(path):
    parts=str(path or "").strip('/').split('/')
    if parts and parts[0] in LOCALES: parts=parts[1:]
    return parts[0] if len(parts)==1 and parts[0] in BROWSER_TOOLS else None


async def get_balance(user_id):
    db=await get_db()
    try:
        rows=await db.execute_fetchall("SELECT balance FROM tool_credit_balances WHERE user_id=?",(user_id,))
        return rows[0][0] if rows else 0
    finally:
        await db.close()


async def consume_download(user_id,tool_path,feature="tool-download",variant=""):
    await credits.reset_free_daily_quota_if_needed(user_id)
    db=await get_db()
    try:
        await db.execute("BEGIN IMMEDIATE")
        rows=await db.execute_fetchall("SELECT balance,plan_name FROM credits WHERE user_id=?",(user_id,))
        if not rows: return {"ok":False,"remaining":0,"balance":0,"tool_credits":0}
        normal,plan=rows[0]
        rows=await db.execute_fetchall("SELECT balance FROM tool_credit_balances WHERE user_id=?",(user_id,))
        tools=rows[0][0] if rows else 0
        eligible=browser_tool(tool_path)
        # Spend today's free allowance before the lasting gift. Paid customers
        # keep their normal credits for paid features by using tool gifts first.
        daily=normal-await credits.protected_extra_credits(db,user_id,normal) if plan=="free" else 0
        use_gift=bool(eligible and tools>0 and daily<=0)
        if not use_gift and normal<1:
            return {"ok":False,"remaining":normal+(tools if eligible else 0),"balance":normal,"tool_credits":tools}
        metadata=json.dumps({"feature":feature,"variant":variant,"tool":eligible or str(tool_path)})
        if use_gift:
            tools-=1
            await db.execute("UPDATE tool_credit_balances SET balance=? WHERE user_id=?",(tools,user_id))
            await db.execute("INSERT INTO tool_credit_transactions(user_id,delta,reason,metadata) VALUES(?,-1,'browser_tool_export',?)",(user_id,metadata))
        else:
            normal-=1
            await db.execute("UPDATE credits SET balance=?,updated_at=? WHERE user_id=?",(normal,time.time(),user_id))
            await db.execute("INSERT INTO credit_transactions(user_id,delta,reason,metadata) VALUES(?,-1,'generation',?)",(user_id,metadata))
        await db.commit()
        return {"ok":True,"remaining":normal+(tools if eligible else 0),"balance":normal,"tool_credits":tools,"used_tool_credit":use_gift}
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()
