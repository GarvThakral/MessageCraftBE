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


def get_connection():
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


def create_user(*, user_id: str, username: str, password_hash: str, tier: str) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (id, username, password_hash, tier)
                VALUES (%s, %s, %s, %s)
                """,
                (user_id, username, password_hash, tier),
            )


def get_user_by_username(username: str) -> dict[str, Any] | None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def update_user_tier(user_id: str, tier: str) -> None:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET tier = %s WHERE id = %s", (tier, user_id))


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


def insert_payment(
    *,
    payment_id: str,
    provider: str,
    user_id: str,
    tier: str,
    status: str,
    raw: dict[str, Any],
) -> bool:
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO payments (id, provider, user_id, tier, status, raw)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                RETURNING id
                """,
                (payment_id, provider, user_id, tier, status, Jsonb(raw)),
            )
            return cur.fetchone() is not None
