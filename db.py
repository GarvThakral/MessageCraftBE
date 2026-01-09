from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set.")
    return url


def get_connection() -> psycopg.Connection:
    return psycopg.connect(_database_url(), row_factory=dict_row)


@lru_cache(maxsize=1)
def ensure_schema() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    tier TEXT NOT NULL DEFAULT 'FREE',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email TEXT;")
            cur.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN NOT NULL DEFAULT FALSE;"
            )
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMPTZ;")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS tier_expires_at TIMESTAMPTZ;")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS tier_paid_at TIMESTAMPTZ;")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_payment_id TEXT;")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_reminder_at TIMESTAMPTZ;")
            cur.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_reminder_for_expires_at TIMESTAMPTZ;"
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS users_email_unique
                ON users (LOWER(email))
                WHERE email IS NOT NULL;
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS usage (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    week_start DATE NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (session_id, week_start)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_entries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    contact TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    output_text TEXT NOT NULL,
                    tone_key TEXT NOT NULL,
                    tone_scores JSONB NOT NULL,
                    impact_prediction INTEGER NOT NULL,
                    clarity_before INTEGER NOT NULL,
                    clarity_after INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS conversation_entries_session_contact_idx
                ON conversation_entries (session_id, contact, created_at DESC);
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limits (
                    key TEXT NOT NULL,
                    window_start TIMESTAMPTZ NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (key, window_start)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS payments (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    user_id TEXT NOT NULL REFERENCES users(id),
                    tier TEXT NOT NULL,
                    status TEXT NOT NULL,
                    raw JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            cur.execute("ALTER TABLE payments ADD COLUMN IF NOT EXISTS payment_date TIMESTAMPTZ;")
            cur.execute("ALTER TABLE payments ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;")
            cur.execute("ALTER TABLE payments ADD COLUMN IF NOT EXISTS email TEXT;")
            cur.execute("ALTER TABLE payments ADD COLUMN IF NOT EXISTS event_id TEXT;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS email_verifications (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    used_at TIMESTAMPTZ
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS password_resets (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    used_at TIMESTAMPTZ
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    revoked_at TIMESTAMPTZ,
                    last_used_at TIMESTAMPTZ
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS dodo_metadata_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    tier TEXT NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    used_at TIMESTAMPTZ
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS webhook_events (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    event_type TEXT,
                    status TEXT NOT NULL,
                    raw JSONB NOT NULL,
                    error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    processed_at TIMESTAMPTZ
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS unmatched_payments (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    payment_id TEXT,
                    email TEXT,
                    status TEXT,
                    raw JSONB NOT NULL,
                    reason TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS rate_limits_window_idx
                ON rate_limits (window_start);
                """
            )


def get_week_start(now: datetime | None = None) -> date:
    if now is None:
        now = datetime.now(timezone.utc)
    monday = now - timedelta(days=now.weekday())
    return date(monday.year, monday.month, monday.day)


def get_next_week_start(week_start: date) -> datetime:
    return datetime.combine(week_start, datetime.min.time(), tzinfo=timezone.utc) + timedelta(
        days=7
    )


def get_day_start(now: datetime | None = None) -> date:
    if now is None:
        now = datetime.now(timezone.utc)
    return date(now.year, now.month, now.day)


def get_next_day_start(day_start: date) -> datetime:
    return datetime.combine(day_start, datetime.min.time(), tzinfo=timezone.utc) + timedelta(
        days=1
    )


def touch_session(session_id: str) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (id)
                VALUES (%s)
                ON CONFLICT (id) DO UPDATE SET last_seen_at = now()
                """,
                (session_id,),
            )


def create_user(
    *,
    user_id: str,
    username: str,
    email: str | None,
    password_hash: str,
    tier: str,
) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (id, username, email, password_hash, tier)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_id, username, email, password_hash, tier),
            )


def get_user_by_username(username: str) -> dict[str, Any] | None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_user_by_email(email: str) -> dict[str, Any] | None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(%s)", (email,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def update_user_tier(
    user_id: str,
    tier: str,
    *,
    expires_at: datetime | None = None,
    paid_at: datetime | None = None,
    payment_id: str | None = None,
) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET tier = %s,
                    tier_expires_at = %s,
                    tier_paid_at = %s,
                    last_payment_id = %s
                WHERE id = %s
                """,
                (tier, expires_at, paid_at, payment_id, user_id),
            )


def mark_user_email_verified(user_id: str, verified_at: datetime) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET email_verified = TRUE,
                    email_verified_at = %s
                WHERE id = %s
                """,
                (verified_at, user_id),
            )


def update_user_password(user_id: str, password_hash: str) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (password_hash, user_id),
            )


def record_expiry_reminder(user_id: str, expires_at: datetime, sent_at: datetime) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET last_reminder_at = %s,
                    last_reminder_for_expires_at = %s
                WHERE id = %s
                """,
                (sent_at, expires_at, user_id),
            )


def complete_email_verification(user_id: str, token_hash: str, verified_at: datetime) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET email_verified = TRUE,
                    email_verified_at = %s
                WHERE id = %s
                """,
                (verified_at, user_id),
            )
            cur.execute(
                """
                UPDATE email_verifications
                SET used_at = %s
                WHERE token_hash = %s
                """,
                (verified_at, token_hash),
            )


def complete_password_reset(
    user_id: str, token_hash: str, password_hash: str, used_at: datetime
) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (password_hash, user_id),
            )
            cur.execute(
                """
                UPDATE password_resets
                SET used_at = %s
                WHERE token_hash = %s
                """,
                (used_at, token_hash),
            )
            cur.execute(
                """
                UPDATE refresh_tokens
                SET revoked_at = %s
                WHERE user_id = %s AND revoked_at IS NULL
                """,
                (used_at, user_id),
            )


def get_usage_count(session_id: str, week_start: date) -> int:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count FROM usage WHERE session_id = %s AND week_start = %s",
                (session_id, week_start),
            )
            row = cur.fetchone()
            return int(row["count"]) if row else 0


def increment_usage(session_id: str, week_start: date, amount: int) -> int:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO usage (session_id, week_start, count)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id, week_start)
                DO UPDATE SET count = usage.count + EXCLUDED.count
                RETURNING count
                """,
                (session_id, week_start, amount),
            )
            row = cur.fetchone()
            return int(row["count"]) if row else amount


def count_contacts(session_id: str) -> int:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(DISTINCT contact) AS total FROM conversation_entries WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            return int(row["total"]) if row else 0


def contact_exists(session_id: str, contact: str) -> bool:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM conversation_entries WHERE session_id = %s AND contact = %s LIMIT 1",
                (session_id, contact),
            )
            return cur.fetchone() is not None


def insert_conversation_entry(
    *,
    entry_id: str,
    session_id: str,
    contact: str,
    input_text: str,
    output_text: str,
    tone_key: str,
    tone_scores: dict[str, Any],
    impact_prediction: int,
    clarity_before: int,
    clarity_after: int,
) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversation_entries (
                    id,
                    session_id,
                    contact,
                    input_text,
                    output_text,
                    tone_key,
                    tone_scores,
                    impact_prediction,
                    clarity_before,
                    clarity_after
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    entry_id,
                    session_id,
                    contact,
                    input_text,
                    output_text,
                    tone_key,
                    Jsonb(tone_scores),
                    impact_prediction,
                    clarity_before,
                    clarity_after,
                ),
            )


def fetch_conversation_entries(
    session_id: str,
    contact: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            if contact:
                cur.execute(
                    """
                    SELECT *
                    FROM conversation_entries
                    WHERE session_id = %s AND contact = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (session_id, contact, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT *
                    FROM conversation_entries
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (session_id, limit),
                )
            rows = cur.fetchall()
            return [dict(row) for row in rows]


def increment_rate_limit(key: str, window_start: datetime) -> int:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rate_limits (key, window_start, count)
                VALUES (%s, %s, 1)
                ON CONFLICT (key, window_start)
                DO UPDATE SET count = rate_limits.count + 1
                RETURNING count
                """,
                (key, window_start),
            )
            row = cur.fetchone()
            return int(row["count"]) if row else 1


def upsert_payment(
    *,
    payment_id: str,
    provider: str,
    user_id: str,
    tier: str,
    status: str,
    raw: dict[str, Any],
    payment_date: datetime | None = None,
    expires_at: datetime | None = None,
    email: str | None = None,
    event_id: str | None = None,
) -> bool:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO payments (
                    id,
                    provider,
                    user_id,
                    tier,
                    status,
                    raw,
                    payment_date,
                    expires_at,
                    email,
                    event_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET
                    status = EXCLUDED.status,
                    raw = EXCLUDED.raw,
                    payment_date = EXCLUDED.payment_date,
                    expires_at = EXCLUDED.expires_at,
                    email = EXCLUDED.email,
                    event_id = EXCLUDED.event_id
                RETURNING id
                """,
                (
                    payment_id,
                    provider,
                    user_id,
                    tier,
                    status,
                    Jsonb(raw),
                    payment_date,
                    expires_at,
                    email,
                    event_id,
                ),
            )
            return cur.fetchone() is not None


def insert_email_verification(
    *, token_hash: str, user_id: str, expires_at: datetime
) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO email_verifications (token_hash, user_id, expires_at)
                VALUES (%s, %s, %s)
                """,
                (token_hash, user_id, expires_at),
            )


def get_email_verification(token_hash: str) -> dict[str, Any] | None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM email_verifications WHERE token_hash = %s",
                (token_hash,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def mark_email_verification_used(token_hash: str, used_at: datetime) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE email_verifications
                SET used_at = %s
                WHERE token_hash = %s
                """,
                (used_at, token_hash),
            )


def insert_password_reset(*, token_hash: str, user_id: str, expires_at: datetime) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO password_resets (token_hash, user_id, expires_at)
                VALUES (%s, %s, %s)
                """,
                (token_hash, user_id, expires_at),
            )


def get_password_reset(token_hash: str) -> dict[str, Any] | None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM password_resets WHERE token_hash = %s",
                (token_hash,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def mark_password_reset_used(token_hash: str, used_at: datetime) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE password_resets
                SET used_at = %s
                WHERE token_hash = %s
                """,
                (used_at, token_hash),
            )


def insert_refresh_token(
    *, token_hash: str, user_id: str, expires_at: datetime
) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO refresh_tokens (token_hash, user_id, expires_at)
                VALUES (%s, %s, %s)
                """,
                (token_hash, user_id, expires_at),
            )


def get_refresh_token(token_hash: str) -> dict[str, Any] | None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM refresh_tokens WHERE token_hash = %s",
                (token_hash,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def mark_refresh_token_used(token_hash: str, used_at: datetime) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE refresh_tokens
                SET last_used_at = %s
                WHERE token_hash = %s
                """,
                (used_at, token_hash),
            )


def revoke_refresh_token(token_hash: str, revoked_at: datetime) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE refresh_tokens
                SET revoked_at = %s
                WHERE token_hash = %s
                """,
                (revoked_at, token_hash),
            )


def revoke_refresh_tokens_for_user(user_id: str, revoked_at: datetime) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE refresh_tokens
                SET revoked_at = %s
                WHERE user_id = %s AND revoked_at IS NULL
                """,
                (revoked_at, user_id),
            )


def insert_metadata_token(
    *,
    token_hash: str,
    user_id: str,
    tier: str,
    expires_at: datetime,
) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dodo_metadata_tokens (token_hash, user_id, tier, expires_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (token_hash) DO NOTHING
                """,
                (token_hash, user_id, tier, expires_at),
            )


def get_metadata_token(token_hash: str) -> dict[str, Any] | None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM dodo_metadata_tokens WHERE token_hash = %s",
                (token_hash,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def mark_metadata_token_used(token_hash: str, used_at: datetime) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dodo_metadata_tokens
                SET used_at = %s
                WHERE token_hash = %s
                """,
                (used_at, token_hash),
            )


def cleanup_metadata_tokens(cutoff: datetime) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM dodo_metadata_tokens WHERE expires_at < %s",
                (cutoff,),
            )


def insert_webhook_event(
    *,
    event_id: str,
    provider: str,
    event_type: str | None,
    status: str,
    raw: dict[str, Any],
    error: str | None,
) -> bool:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO webhook_events (id, provider, event_type, status, raw, error)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                RETURNING id
                """,
                (event_id, provider, event_type, status, Jsonb(raw), error),
            )
            return cur.fetchone() is not None


def mark_webhook_event_processed(event_id: str, processed_at: datetime) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE webhook_events
                SET processed_at = %s
                WHERE id = %s
                """,
                (processed_at, event_id),
            )


def insert_unmatched_payment(
    *,
    entry_id: str,
    provider: str,
    payment_id: str | None,
    email: str | None,
    status: str | None,
    raw: dict[str, Any],
    reason: str,
) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO unmatched_payments (
                    id,
                    provider,
                    payment_id,
                    email,
                    status,
                    raw,
                    reason
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (entry_id, provider, payment_id, email, status, Jsonb(raw), reason),
            )


def apply_payment_update(
    *,
    user_id: str,
    tier: str,
    status: str,
    payment_id: str,
    raw: dict[str, Any],
    payment_date: datetime | None,
    expires_at: datetime | None,
    email: str | None,
    event_id: str | None,
    metadata_token_hash: str | None,
) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET tier = %s,
                    tier_expires_at = CASE WHEN %s = 'FREE' THEN NULL ELSE %s END,
                    tier_paid_at = CASE WHEN %s = 'FREE' THEN NULL ELSE %s END,
                    last_payment_id = %s
                WHERE id = %s
                """,
                (tier, tier, expires_at, tier, payment_date, payment_id, user_id),
            )
            cur.execute(
                """
                INSERT INTO payments (
                    id,
                    provider,
                    user_id,
                    tier,
                    status,
                    raw,
                    payment_date,
                    expires_at,
                    email,
                    event_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET
                    status = EXCLUDED.status,
                    raw = EXCLUDED.raw,
                    payment_date = EXCLUDED.payment_date,
                    expires_at = EXCLUDED.expires_at,
                    email = EXCLUDED.email,
                    event_id = EXCLUDED.event_id
                """,
                (
                    payment_id,
                    "dodo",
                    user_id,
                    tier,
                    status,
                    Jsonb(raw),
                    payment_date,
                    expires_at,
                    email,
                    event_id,
                ),
            )
            if metadata_token_hash:
                cur.execute(
                    """
                    UPDATE dodo_metadata_tokens
                    SET used_at = now()
                    WHERE token_hash = %s
                    """,
                    (metadata_token_hash,),
                )
