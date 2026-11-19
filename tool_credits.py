"""Separate promotional wallet for browser tools. Never funds an AI provider."""
import json
import time
import re
import uuid
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
CREATE TABLE IF NOT EXISTS tool_credit_artifacts (
    user_id INTEGER NOT NULL REFERENCES users(id),
    operation_id TEXT NOT NULL,
    artifact_hash TEXT NOT NULL,
    cost INTEGER NOT NULL CHECK(cost=10),
    funding_source TEXT NOT NULL CHECK(funding_source IN ('tool','ordinary')),
    created_at REAL NOT NULL DEFAULT (unixepoch()),
    PRIMARY KEY(user_id,operation_id),
    UNIQUE(user_id,artifact_hash)
);
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


FLIPBOOK_COST = 10
FLIPBOOK_PLANS = frozenset({'creator','pro','studio'})


class FlipbookCreditError(ValueError):
    def __init__(self,status,message):
        super().__init__(message)
        self.status=status


async def _flipbook_access(db,user_id):
    rows=await db.execute_fetchall("SELECT balance,plan_name FROM credits WHERE user_id=?",(user_id,))
    if not rows:raise FlipbookCreditError(404,'Credit account not found')
    normal,plan=rows[0]
    rows=await db.execute_fetchall("SELECT balance FROM tool_credit_balances WHERE user_id=?",(user_id,))
    tools=rows[0][0] if rows else 0
    paid=plan in FLIPBOOK_PLANS
    previous=bool(await db.execute_fetchall("SELECT 1 FROM tool_credit_artifacts WHERE user_id=? LIMIT 1",(user_id,)))
    return {'user_id':user_id,'cost':FLIPBOOK_COST,'balance':normal,'tool_credits':tools,
            'is_paid':paid,'can_create':tools>=FLIPBOOK_COST or (paid and normal>=FLIPBOOK_COST),
            'has_previous':previous,'allowed_modes':['basic','audio','video'] if paid else ['basic']}


async def flipbook_access(user_id):
    db=await get_db()
    try:return await _flipbook_access(db,user_id)
    finally:await db.close()


async def consume_flipbook(user_id,operation_id,artifact_hash):
    try:
        if str(uuid.UUID(operation_id))!=operation_id:raise ValueError()
    except (ValueError,TypeError,AttributeError):
        raise FlipbookCreditError(400,'Invalid operation identifier')
    if not re.fullmatch(r'[a-f0-9]{64}',artifact_hash or ''):
        raise FlipbookCreditError(400,'Invalid artifact fingerprint')
    db=await get_db()
    try:
        await db.execute('BEGIN IMMEDIATE')
        access=await _flipbook_access(db,user_id)
        rows=await db.execute_fetchall('SELECT artifact_hash FROM tool_credit_artifacts WHERE user_id=? AND operation_id=?',(user_id,operation_id))
        if rows and rows[0][0]!=artifact_hash:
            raise FlipbookCreditError(409,'Operation already used for a different flipbook')
        rows=await db.execute_fetchall('SELECT operation_id,funding_source FROM tool_credit_artifacts WHERE user_id=? AND artifact_hash=?',(user_id,artifact_hash))
        if rows:
            return {**access,'ok':True,'already_charged':True,'operation_id':rows[0][0],'funding_source':rows[0][1]}
        if not access['can_create']:
            raise FlipbookCreditError(402,'A basic flipbook requires 10 tool credits, or 10 ordinary credits on a paid account')
        source='tool' if access['tool_credits']>=FLIPBOOK_COST else 'ordinary'
        metadata=json.dumps({'feature':'flipbook','variant':'basic','operation_id':operation_id,'artifact_hash':artifact_hash,'cost':FLIPBOOK_COST})
        if source=='tool':
            await db.execute('UPDATE tool_credit_balances SET balance=balance-? WHERE user_id=?',(FLIPBOOK_COST,user_id))
            await db.execute("INSERT INTO tool_credit_transactions(user_id,delta,reason,metadata) VALUES(?,?,'flipbook_basic',?)",(user_id,-FLIPBOOK_COST,metadata))
        else:
            await db.execute('UPDATE credits SET balance=balance-?,updated_at=? WHERE user_id=?',(FLIPBOOK_COST,time.time(),user_id))
            await db.execute("INSERT INTO credit_transactions(user_id,delta,reason,metadata) VALUES(?,?,'generation',?)",(user_id,-FLIPBOOK_COST,metadata))
        await db.execute('INSERT INTO tool_credit_artifacts(user_id,operation_id,artifact_hash,cost,funding_source) VALUES(?,?,?,?,?)',(user_id,operation_id,artifact_hash,FLIPBOOK_COST,source))
        result=await _flipbook_access(db,user_id)
        await db.commit()
        return {**result,'ok':True,'already_charged':False,'operation_id':operation_id,'funding_source':source}
    except Exception:
        await db.rollback()
        raise
    finally:await db.close()
