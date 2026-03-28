"""
Database module — SQLite via aiosqlite.
Stores users, credits, transactions, and refresh tokens.
"""

import os
import aiosqlite

DB_PATH = os.environ.get("DB_PATH", "/data/mosaic.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    email           TEXT UNIQUE NOT NULL COLLATE NOCASE,
    password_hash   TEXT NOT NULL,
    display_name    TEXT DEFAULT '',
    created_at      REAL NOT NULL DEFAULT (unixepoch()),
    email_verified  INTEGER DEFAULT 0,
    stripe_customer_id TEXT UNIQUE,
    lang            TEXT DEFAULT 'en',
    affiliate_ref   TEXT DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_stripe ON users(stripe_customer_id);

CREATE TABLE IF NOT EXISTS credits (
    user_id          INTEGER PRIMARY KEY REFERENCES users(id),
    balance          INTEGER NOT NULL DEFAULT 3,
    monthly_quota    INTEGER NOT NULL DEFAULT 3,
    plan_name        TEXT NOT NULL DEFAULT 'free',
    stripe_sub_id    TEXT,
    sub_status       TEXT DEFAULT 'none',
    current_period_end REAL,
    updated_at       REAL NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS credit_transactions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    delta      INTEGER NOT NULL,
    reason     TEXT NOT NULL,
    metadata   TEXT,
    created_at REAL NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_txn_user ON credit_transactions(user_id, created_at);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    token_hash TEXT UNIQUE NOT NULL,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id);

CREATE TABLE IF NOT EXISTS api_keys (
    id         TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    key_hash   TEXT UNIQUE NOT NULL,
    key_prefix TEXT NOT NULL,
    name       TEXT DEFAULT '',
    created_at REAL NOT NULL DEFAULT (unixepoch()),
    last_used_at REAL,
    is_active  INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_apikeys_user ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_apikeys_hash ON api_keys(key_hash);

CREATE TABLE IF NOT EXISTS dynamic_qrcodes (
    id          TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    short_code  TEXT UNIQUE NOT NULL,
    target_url  TEXT NOT NULL,
    label       TEXT DEFAULT '',
    scan_count  INTEGER DEFAULT 0,
    created_at  REAL NOT NULL DEFAULT (unixepoch()),
    updated_at  REAL NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_qr_user ON dynamic_qrcodes(user_id);
CREATE INDEX IF NOT EXISTS idx_qr_short ON dynamic_qrcodes(short_code);

CREATE TABLE IF NOT EXISTS affiliate_codes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    code            TEXT UNIQUE NOT NULL COLLATE NOCASE,
    affiliate_name  TEXT NOT NULL,
    affiliate_email TEXT NOT NULL COLLATE NOCASE,
    is_used         INTEGER DEFAULT 0,
    used_by_email   TEXT DEFAULT NULL,
    used_at         REAL DEFAULT NULL,
    created_at      REAL NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_aff_code ON affiliate_codes(code);
CREATE INDEX IF NOT EXISTS idx_aff_email ON affiliate_codes(affiliate_email);

CREATE TABLE IF NOT EXISTS affiliate_applications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    email           TEXT NOT NULL COLLATE NOCASE,
    phone           TEXT DEFAULT '',
    website         TEXT DEFAULT '',
    audience        TEXT DEFAULT '',
    promotion       TEXT DEFAULT '',
    lang            TEXT DEFAULT 'en',
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      REAL NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_aff_app_email ON affiliate_applications(email);
CREATE INDEX IF NOT EXISTS idx_aff_app_status ON affiliate_applications(status);

CREATE TABLE IF NOT EXISTS affiliate_commissions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    affiliate_email TEXT NOT NULL COLLATE NOCASE,
    affiliate_name  TEXT NOT NULL,
    affiliate_code  TEXT NOT NULL,
    customer_id     TEXT NOT NULL,
    subscription_id TEXT,
    plan            TEXT NOT NULL DEFAULT '',
    started_at      REAL NOT NULL DEFAULT (unixepoch()),
    commission_rate REAL NOT NULL DEFAULT 0.30,
    months_total    INTEGER NOT NULL DEFAULT 6,
    months_paid     INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_aff_comm_email ON affiliate_commissions(affiliate_email);
CREATE INDEX IF NOT EXISTS idx_aff_comm_customer ON affiliate_commissions(customer_id);

CREATE TABLE IF NOT EXISTS affiliate_payouts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    affiliate_email TEXT NOT NULL COLLATE NOCASE,
    commission_id   INTEGER REFERENCES affiliate_commissions(id),
    month_number    INTEGER NOT NULL,
    amount_usd      REAL NOT NULL,
    invoice_id      TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      REAL NOT NULL DEFAULT (unixepoch()),
    paid_at         REAL
);
CREATE INDEX IF NOT EXISTS idx_aff_pay_email ON affiliate_payouts(affiliate_email);
CREATE INDEX IF NOT EXISTS idx_aff_pay_status ON affiliate_payouts(status);

CREATE TABLE IF NOT EXISTS trial_codes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    code            TEXT UNIQUE NOT NULL COLLATE NOCASE,
    created_for     TEXT NOT NULL,
    plan            TEXT NOT NULL DEFAULT 'studio',
    credits         INTEGER NOT NULL DEFAULT 1200,
    duration_days   INTEGER NOT NULL DEFAULT 30,
    used_by         INTEGER REFERENCES users(id),
    used_at         REAL,
    expires_at      REAL,
    created_at      REAL NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_trial_code ON trial_codes(code);
"""


async def get_db() -> aiosqlite.Connection:
    """Open a connection with WAL mode and foreign keys enabled."""
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    """Create all tables if they don't exist. Called once at app startup."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = await get_db()
    try:
        await db.executescript(SCHEMA_SQL)
        # Migration: add affiliate_ref column if missing (existing databases)
        try:
            await db.execute("ALTER TABLE users ADD COLUMN affiliate_ref TEXT DEFAULT NULL")
        except Exception:
            pass  # column already exists
        # Migration: affiliate_commissions and affiliate_payouts may not exist on old DBs
        # (handled by CREATE TABLE IF NOT EXISTS in SCHEMA_SQL above)
        await db.commit()
    finally:
        await db.close()
