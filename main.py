from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import logging
import os
import re
import smtplib
import uuid
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from functools import lru_cache
from typing import Any, TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from standardwebhooks import Webhook
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError
import psycopg

from auth import (
    create_token,
    generate_refresh_token,
    hash_password,
    hash_token,
    verify_password,
    verify_token,
)

from db import (
    contact_exists,
    count_contacts,
    create_user,
    complete_email_verification,
    complete_password_reset,
    fetch_conversation_entries,
    get_day_start,
    get_email_verification,
    get_metadata_token,
    get_next_day_start,
    get_next_week_start,
    get_password_reset,
    get_refresh_token,
    get_usage_count,
    get_user_by_email,
    get_user_by_id,
    get_user_by_username,
    get_week_start,
    insert_email_verification,
    insert_metadata_token,
    insert_password_reset,
    insert_refresh_token,
    insert_unmatched_payment,
    insert_webhook_event,
    increment_usage,
    increment_rate_limit,
    insert_conversation_entry,
    mark_refresh_token_used,
    mark_webhook_event_processed,
    record_expiry_reminder,
    revoke_refresh_token,
    revoke_refresh_tokens_for_user,
    touch_session,
    update_user_tier,
    apply_payment_update,
    cleanup_metadata_tokens,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("messagecraft")

if os.getenv("OPENROUTER_API_KEY"):
    os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENROUTER_API_KEY"))
if os.getenv("OPENROUTER_BASE_URL"):
    os.environ.setdefault("OPENAI_BASE_URL", os.getenv("OPENROUTER_BASE_URL"))
    os.environ.setdefault("OPENAI_API_BASE", os.getenv("OPENROUTER_BASE_URL"))

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


class MessageCraftRequest(BaseModel):
    text: str
    audience_style: str | None = None
    tone_balance: int | None = Field(default=None, ge=0, le=100)
    user_goal: str | None = None


class ContextInfo(BaseModel):
    detected_type: str
    confidence: float
    rationale: str


class ToneVersions(BaseModel):
    professional_formal: str
    empathetic_warm: str
    direct_assertive: str
    diplomatic_tactful: str
    casual_friendly: str


class ToneScores(BaseModel):
    emotion: int
    formality: int
    assertiveness: int
    clarity: int


class AnalysisPanel(BaseModel):
    tone_scores: ToneScores
    tone_summary: str
    misinterpretation_warnings: list[str]
    power_dynamics: str
    urgency: str
    framework_tags: list[str]


class TacticalEnhancements(BaseModel):
    add_emotional_validation: str
    remove_emotional_validation: str
    include_action_items: str
    exclude_action_items: str
    add_softeners: str
    add_strengtheners: str
    insert_boundaries: str
    frame_as_question: str
    frame_as_statement: str


class RedFlags(BaseModel):
    manipulative_patterns: list[str]
    defensive_phrases: list[str]
    assumptions: list[str]
    escalation_triggers: list[str]


class OneClickScenarios(BaseModel):
    boundary_setting: str
    clarifying_question: str
    accountability_without_blame: str
    collaborative_proposal: str
    non_defensive_response: str


class QuickActions(BaseModel):
    condense: str
    expand: str
    cool_down: str
    add_assertiveness: str


class UniqueSellingPoints(BaseModel):
    before_clarity_score: int
    after_clarity_score: int
    relationship_impact_prediction: int
    cultural_style: str
    generation_notes: str


class Explanations(BaseModel):
    why_it_works: list[str]
    change_log: list[str]


class UsageMeta(BaseModel):
    tier: str
    count: int
    limit: int | None
    reset_at: str


class MessageCraftResponse(BaseModel):
    context: ContextInfo
    tone_versions: ToneVersions
    analysis: AnalysisPanel
    tactical_enhancements: TacticalEnhancements
    red_flags: RedFlags
    one_click_scenarios: OneClickScenarios
    quick_actions: QuickActions
    usp: UniqueSellingPoints
    explanations: Explanations
    meta: UsageMeta | None = None


class TranslateRequest(BaseModel):
    text: str
    mode: str


class TranslateResponse(BaseModel):
    output: str


class UsageResponse(BaseModel):
    tier: str
    count: int
    limit: int | None
    reset_at: str


class ConversationEntryRequest(BaseModel):
    contact: str
    input_text: str
    output_text: str
    tone_key: str
    tone_scores: ToneScores
    impact_prediction: int
    clarity_before: int
    clarity_after: int


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str | None = None


class VerifyEmailRequest(BaseModel):
    token: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    tier: str
    email: str | None = None
    email_verified: bool = False
    tier_expires_at: str | None = None


class AuthResponse(BaseModel):
    token: str
    refresh_token: str
    user: UserResponse


class TierUpdateRequest(BaseModel):
    tier: str


class AdminTierUpdateRequest(BaseModel):
    user_id: str
    tier: str
    days: int | None = None


class TierStatusResponse(BaseModel):
    current_tier: str
    tier_expires_at: str | None
    days_remaining: int | None
    is_expired: bool


class ResyncPaymentRequest(BaseModel):
    payment_id: str | None = None


class SupportRequest(BaseModel):
    name: str
    email: str
    subject: str
    message: str
    category: str | None = None
    order_id: str | None = None


class DodoCheckoutRequest(BaseModel):
    tier: str


class DodoCheckoutResponse(BaseModel):
    checkout_url: str

app = FastAPI(title="MessageCraft Pro Backend")

TIER_FREE = "FREE"
TIER_STARTER = "STARTER"
TIER_PRO = "PRO"
TIERS = (TIER_FREE, TIER_STARTER, TIER_PRO)

TIER_LIMITS = {
    TIER_FREE: 1,
    TIER_STARTER: 25,
    TIER_PRO: None,
}

TIER_TONE_LIMITS = {
    TIER_FREE: 3,
    TIER_STARTER: 5,
    TIER_PRO: 5,
}

CONTACT_LIMITS = {
    TIER_FREE: 0,
    TIER_STARTER: 5,
    TIER_PRO: None,
}


def _parse_origins(value: str | None) -> list[str]:
    if not value:
        return []
    origins: list[str] = []
    for origin in value.split(","):
        origin = origin.strip()
        if not origin or origin == "*":
            continue
        origins.append(origin.rstrip("/"))
    return origins


def _frontend_url() -> str:
    url = _env("FRONTEND_URL")
    if not url:
        raise RuntimeError("FRONTEND_URL is not set.")
    return url.rstrip("/")


def _cors_origins() -> list[str]:
    origins = [_frontend_url()]
    extra = _parse_origins(_env("CORS_ORIGINS"))
    for origin in extra:
        if origin not in origins:
            origins.append(origin)
    return origins


def _validate_env() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    strict = _strict_env()
    payments_enabled = _payments_enabled()
    auth_secret = _env("AUTH_SECRET")
    if not auth_secret:
        errors.append("AUTH_SECRET is required.")
    elif strict and len(auth_secret) < 32:
        errors.append("AUTH_SECRET must be at least 32 characters.")
    admin_secret = _env("ADMIN_SECRET")
    if not admin_secret:
        if strict:
            errors.append("ADMIN_SECRET is required.")
        else:
            warnings.append("ADMIN_SECRET is not set; admin endpoints disabled.")
    elif strict and len(admin_secret) < 16:
        errors.append("ADMIN_SECRET must be at least 16 characters.")

    if payments_enabled:
        dodo_webhook_secret = _env("DODO_WEBHOOK_SECRET") or _env("DODO_WEBHOOK_KEY")
        if not dodo_webhook_secret:
            errors.append("DODO_WEBHOOK_SECRET is required.")
        if not _env("DODO_STARTER_LINK") or not _env("DODO_PRO_LINK"):
            errors.append("DODO_STARTER_LINK and DODO_PRO_LINK are required.")
        dodo_env = _env("DODO_ENVIRONMENT")
        if dodo_env and dodo_env not in {"test_mode", "live_mode"}:
            errors.append('DODO_ENVIRONMENT must be "test_mode" or "live_mode".')
        if not _env("DODO_API_KEY"):
            warnings.append("DODO_API_KEY not set; manual payment resync disabled.")
    else:
        warnings.append("Payments disabled; Dodo keys not required.")
    try:
        _ = _frontend_url()
    except RuntimeError as exc:
        errors.append(str(exc))

    if payments_enabled:
        return_url = _env("DODO_RETURN_URL")
        if not return_url:
            errors.append("DODO_RETURN_URL is required.")
        else:
            frontend_origin = urlparse(_frontend_url()).netloc
            return_origin = urlparse(return_url).netloc
            dodo_env = _env("DODO_ENVIRONMENT") or _dodo_environment()
            if dodo_env == "live_mode" and frontend_origin and return_origin != frontend_origin:
                errors.append("DODO_RETURN_URL must match FRONTEND_URL in live_mode.")

    try:
        int(_env("TIER_EXPIRY_DAYS", "30") or 30)
    except ValueError:
        errors.append("TIER_EXPIRY_DAYS must be an integer.")
    try:
        int(_env("EMAIL_REMINDER_DAYS_BEFORE_EXPIRY", "3") or 3)
    except ValueError:
        errors.append("EMAIL_REMINDER_DAYS_BEFORE_EXPIRY must be an integer.")

    if errors:
        raise RuntimeError("Config errors: " + "; ".join(errors))
    for warning in warnings:
        logger.warning(warning)


def _normalize_tier(value: str | None) -> str:
    if not value:
        return TIER_FREE
    normalized = value.strip().upper()
    if normalized in TIER_LIMITS:
        return normalized
    return TIER_FREE




def _build_usage_meta(tier: str, count: int, period_start: date) -> UsageMeta:
    limit = TIER_LIMITS[tier]
    if tier == TIER_FREE:
        reset_at = get_next_day_start(period_start).isoformat()
    else:
        reset_at = get_next_week_start(period_start).isoformat()
    return UsageMeta(tier=tier, count=count, limit=limit, reset_at=reset_at)


MESSAGECRAFT_SCHEMA = """
Return JSON with this schema:
{
  "context": {
    "detected_type": "personal relationship | professional email | customer service | social media | negotiation | apology | request | other",
    "confidence": 0.0,
    "rationale": "..."
  },
  "tone_versions": {
    "professional_formal": "...",
    "empathetic_warm": "...",
    "direct_assertive": "...",
    "diplomatic_tactful": "...",
    "casual_friendly": "..."
  },
  "analysis": {
    "tone_scores": {"emotion": 0, "formality": 0, "assertiveness": 0, "clarity": 0},
    "tone_summary": "...",
    "misinterpretation_warnings": ["..."],
    "power_dynamics": "speaker_has_leverage | balanced | other_has_leverage",
    "urgency": "low | medium | high",
    "framework_tags": ["NVC", "Active Listening", "Assertiveness", "Boundary-Setting", "Collaborative"]
  },
  "tactical_enhancements": {
    "add_emotional_validation": "...",
    "remove_emotional_validation": "...",
    "include_action_items": "...",
    "exclude_action_items": "...",
    "add_softeners": "...",
    "add_strengtheners": "...",
    "insert_boundaries": "...",
    "frame_as_question": "...",
    "frame_as_statement": "..."
  },
  "red_flags": {
    "manipulative_patterns": ["..."],
    "defensive_phrases": ["..."],
    "assumptions": ["..."],
    "escalation_triggers": ["..."]
  },
  "one_click_scenarios": {
    "boundary_setting": "...",
    "clarifying_question": "...",
    "accountability_without_blame": "...",
    "collaborative_proposal": "...",
    "non_defensive_response": "..."
  },
  "quick_actions": {
    "condense": "...",
    "expand": "...",
    "cool_down": "...",
    "add_assertiveness": "..."
  },
  "usp": {
    "before_clarity_score": 0,
    "after_clarity_score": 0,
    "relationship_impact_prediction": 0,
    "cultural_style": "neutral | Gen Z | Millennial | Boomer",
    "generation_notes": "..."
  },
  "explanations": {
    "why_it_works": ["..."],
    "change_log": ["..."]
  }
}
""".strip()


def _build_messagecraft_prompt(request: MessageCraftRequest) -> str:
    audience = request.audience_style or "Gen Z"
    tone_balance = "balanced" if request.tone_balance is None else str(request.tone_balance)
    goal = request.user_goal or "general clarity and respect"
    style_notes: list[str] = []
    audience_lower = audience.lower()
    if "gen" in audience_lower and "z" in audience_lower:
        style_notes.append(
            "Style: casual Gen Z, short lines, contractions, no corporate/therapy phrasing."
        )
    if "de-escalat" in goal.lower():
        style_notes.append(
            "De-escalate: calm, non-accusatory, keep emotional truth, remove insults."
        )
    style_hint = " ".join(style_notes)

    return (
        "Return ONE JSON object that matches the schema exactly. No markdown or extra text. "
        "Rewrite the input as if sent by the original speaker. Never reply or acknowledge it. "
        "Preserve intent and point of view. Do not add new facts, decisions, or requests. "
        "Every output string must be a rewrite (same speaker). No acknowledgments like "
        "'Understood', 'Okay', 'Sure', 'I will', 'I won't', or 'Got it' unless those words "
        "appear in the original input. "
        "If the input is ambiguous, keep it ambiguous. "
        "Keep length close to the original (about +/- 30%). Use short sentences. "
        "Avoid corporate or therapy jargon unless the input uses it. "
        f"Audience style: {audience}. "
        f"Tone balance (0-100, low=emotional high=logical): {tone_balance}. "
        f"Primary goal: {goal}. "
        f"{style_hint} "
        "Provide all fields. Use empty arrays when none. "
        "Scores are integers 0-100; after_clarity_score >= before_clarity_score. "
        "relationship_impact_prediction is 0-100. "
        "Use plain text; avoid emojis unless the original text includes them.\n\n"
        + MESSAGECRAFT_SCHEMA
    )


def _build_prompt(mode: str) -> str:
    if mode == "logic":
        return (
            "Convert emotional text into calm, logical, NVC-style communication. "
            "Rewrite as the original speaker. Never reply or acknowledge the message. "
            "No acknowledgment phrases like 'Understood', 'Okay', 'Sure', 'I will', or 'Got it'. "
            "Keep it concise, respectful, and specific."
        )
    if mode == "emotion":
        return (
            "Convert overly logical or dismissive text into warm, emotionally "
            "validating language. Rewrite as the original speaker. "
            "Never reply or acknowledge the message. "
            "No acknowledgment phrases like 'Understood', 'Okay', 'Sure', 'I will', or 'Got it'. "
            "Keep it concise, genuine, and non-blaming."
        )
    raise HTTPException(status_code=400, detail="Unsupported mode.")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _bool_env(name: str, default: str | None = None) -> bool:
    raw = _env(name, default)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes"}


def _environment() -> str:
    return (_env("ENVIRONMENT", "development") or "development").strip().lower()


def _strict_env() -> bool:
    if os.getenv("STRICT_ENV_VALIDATION") is not None:
        return _bool_env("STRICT_ENV_VALIDATION", "true")
    return _environment() in {"production", "prod"}


def _payments_enabled() -> bool:
    if os.getenv("PAYMENTS_ENABLED") is not None:
        return _bool_env("PAYMENTS_ENABLED", "true")
    return _environment() in {"production", "prod"}


RATE_LIMITS = {
    TIER_FREE: int(_env("RATE_LIMIT_FREE_PER_MINUTE", "10") or 10),
    TIER_STARTER: int(_env("RATE_LIMIT_STARTER_PER_MINUTE", "30") or 30),
    TIER_PRO: int(_env("RATE_LIMIT_PRO_PER_MINUTE", "60") or 60),
}

RATE_LIMIT_IP = int(_env("RATE_LIMIT_IP_PER_MINUTE", "120") or 120)
RATE_LIMIT_AUTH = int(_env("RATE_LIMIT_AUTH_PER_MINUTE", "20") or 20)


def _auth_secret() -> str:
    """Load and validate the shared auth secret used for signing tokens."""
    secret = _env("AUTH_SECRET")
    if not secret:
        raise RuntimeError("AUTH_SECRET is not set.")
    return secret


def _token_ttl_seconds() -> int:
    return int(_env("AUTH_ACCESS_TOKEN_TTL_SECONDS") or _env("AUTH_TOKEN_TTL_SECONDS", "3600") or 3600)


def _refresh_token_ttl_seconds() -> int:
    return int(_env("AUTH_REFRESH_TOKEN_TTL_SECONDS", "2592000") or 2592000)


def _admin_secret() -> str:
    secret = _env("ADMIN_SECRET")
    if not secret:
        raise RuntimeError("ADMIN_SECRET is not set.")
    return secret


def _tier_expiry_days() -> int:
    return int(_env("TIER_EXPIRY_DAYS", "30") or 30)


def _email_reminder_days() -> int:
    return int(_env("EMAIL_REMINDER_DAYS_BEFORE_EXPIRY", "3") or 3)


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    if request.client:
        return request.client.host
    return "unknown"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _wants_cookie_auth(request: Request) -> bool:
    header_flag = (request.headers.get("x-use-cookie") or "").lower() in {"1", "true", "yes"}
    has_cookie = bool(request.cookies.get("mc_refresh") or request.cookies.get("mc_access"))
    return header_flag or has_cookie


def _cookie_secure() -> bool:
    try:
        return urlparse(_frontend_url()).scheme == "https"
    except RuntimeError:
        return False


def _read_access_token(request: Request) -> str | None:
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    if header and header.lower().startswith("bearer "):
        return header.split(" ", 1)[1].strip()
    return request.cookies.get("mc_access")


def _read_refresh_token(request: Request, payload: RefreshRequest | None = None) -> str | None:
    if payload and payload.refresh_token:
        return payload.refresh_token.strip()
    return request.cookies.get("mc_refresh")


def _smtp_settings() -> dict[str, Any]:
    host = _env("SMTP_HOST")
    sender = _env("SMTP_FROM")
    if not host or not sender:
        raise RuntimeError("SMTP is not configured.")
    port = int(_env("SMTP_PORT", "587") or 587)
    user = _env("SMTP_USER")
    password = _env("SMTP_PASSWORD")
    use_tls = (_env("SMTP_USE_TLS", "true") or "true").lower() in {"1", "true", "yes"}
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "sender": sender,
        "use_tls": use_tls,
    }


def _send_email(
    to_address: str,
    subject: str,
    body: str,
    reply_to: str | None = None,
) -> None:
    settings = _smtp_settings()
    message = EmailMessage()
    message["From"] = settings["sender"]
    message["To"] = to_address
    message["Subject"] = subject
    if reply_to:
        message["Reply-To"] = reply_to
    message.set_content(body)
    with smtplib.SMTP(settings["host"], settings["port"]) as smtp:
        if settings["use_tls"]:
            smtp.starttls()
        if settings["user"] and settings["password"]:
            smtp.login(settings["user"], settings["password"])
        smtp.send_message(message)


def _email_verification_ttl_seconds() -> int:
    return int(_env("EMAIL_VERIFICATION_TTL_SECONDS", "86400") or 86400)


def _password_reset_ttl_seconds() -> int:
    return int(_env("PASSWORD_RESET_TTL_SECONDS", "3600") or 3600)


def _verification_link(token: str) -> str:
    return f"{_frontend_url()}/verify-email?token={token}"


def _password_reset_link(token: str) -> str:
    return f"{_frontend_url()}/reset-password?token={token}"


def _send_verification_email(to_address: str, token: str) -> None:
    _send_email(
        to_address,
        "Verify your MessageCraft Pro email",
        f"Click to verify your email: {_verification_link(token)}",
    )


def _send_password_reset_email(to_address: str, token: str) -> None:
    _send_email(
        to_address,
        "Reset your MessageCraft Pro password",
        f"Reset your password here: {_password_reset_link(token)}",
    )


def _support_email_to() -> str:
    address = _env("SUPPORT_EMAIL_TO")
    if not address:
        raise RuntimeError("SUPPORT_EMAIL_TO is not set.")
    return address


def _alert_admin(subject: str, body: str) -> None:
    alert_email = _env("ALERT_EMAIL_TO")
    if not alert_email:
        logger.error("Webhook failure: %s", body)
        return
    try:
        _send_email(alert_email, subject, body)
    except Exception:
        logger.exception("Failed to send admin alert.")


def _maybe_send_expiry_reminder(user: dict[str, Any]) -> None:
    if not user.get("email_verified"):
        return
    email = user.get("email")
    if not email:
        return
    tier = _normalize_tier(user.get("tier"))
    if tier == TIER_FREE:
        return
    expires_at = user.get("tier_expires_at")
    if not isinstance(expires_at, datetime):
        return
    now = _now_utc()
    reminder_threshold = expires_at - timedelta(days=_email_reminder_days())
    if now < reminder_threshold:
        return
    if user.get("last_reminder_for_expires_at") == expires_at:
        return
    try:
        _send_email(
            email,
            f"Your {tier} access expires soon",
            (
                f"Your {tier} access expires on {expires_at.date().isoformat()}.\n"
                "Pay again to continue without interruption."
            ),
        )
        record_expiry_reminder(user["id"], expires_at, now)
    except Exception:
        logger.exception("Failed to send expiry reminder.")


def _issue_tokens(user_id: str) -> tuple[str, str]:
    access = create_token(
        {"sub": user_id, "typ": "access"},
        _auth_secret(),
        _token_ttl_seconds(),
    )
    refresh = generate_refresh_token()
    token_hash = hash_token(refresh)
    expires_at = _now_utc() + timedelta(seconds=_refresh_token_ttl_seconds())
    insert_refresh_token(token_hash=token_hash, user_id=user_id, expires_at=expires_at)
    return access, refresh


def _set_auth_cookies(response: Response, access: str, refresh: str) -> None:
    secure = _cookie_secure()
    response.set_cookie(
        "mc_access",
        access,
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=_token_ttl_seconds(),
        path="/",
    )
    response.set_cookie(
        "mc_refresh",
        refresh,
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=_refresh_token_ttl_seconds(),
        path="/",
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie("mc_access", path="/")
    response.delete_cookie("mc_refresh", path="/")


def _user_response_from_row(user: dict[str, Any]) -> UserResponse:
    expires_at = user.get("tier_expires_at")
    return UserResponse(
        id=user["id"],
        username=user["username"],
        tier=_normalize_tier(user.get("tier")),
        email=user.get("email"),
        email_verified=bool(user.get("email_verified")),
        tier_expires_at=expires_at.isoformat() if isinstance(expires_at, datetime) else None,
    )


def _ensure_user_active_tier(user: dict[str, Any]) -> dict[str, Any]:
    """Downgrade expired tiers and send renewal reminders if needed."""
    tier = _normalize_tier(user.get("tier"))
    expires_at = user.get("tier_expires_at")
    now = _now_utc()
    if tier != TIER_FREE and isinstance(expires_at, datetime) and now > expires_at:
        update_user_tier(user["id"], TIER_FREE, expires_at=None, paid_at=None, payment_id=None)
        logger.info("Tier expired; user downgraded to FREE: %s", user["id"])
        user = dict(user)
        user["tier"] = TIER_FREE
        user["tier_expires_at"] = None
        user["tier_paid_at"] = None
        user["last_payment_id"] = None
    _maybe_send_expiry_reminder(user)
    return user


def _require_user(request: Request, require_verified: bool = True) -> dict[str, Any]:
    token = _read_access_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token.")
    try:
        payload = verify_token(token, _auth_secret())
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    token_type = payload.get("typ")
    if token_type not in {None, "access"}:
        raise HTTPException(status_code=401, detail="Invalid auth token.")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid auth token.")
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    user = _ensure_user_active_tier(user)
    if require_verified and not user.get("email_verified"):
        raise HTTPException(status_code=403, detail="Email verification required.")
    return user


def _effective_tier(request: Request, user: dict[str, Any]) -> str:
    return _normalize_tier(user.get("tier"))


def _rate_limit(request: Request, tier: str, user_id: str) -> None:
    window_start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    user_key = f"user:{user_id}"
    user_count = increment_rate_limit(user_key, window_start)
    limit = RATE_LIMITS.get(tier, RATE_LIMITS[TIER_FREE])
    if user_count > limit:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down and try again.",
        )

    ip = _get_client_ip(request)
    if ip:
        ip_count = increment_rate_limit(f"ip:{ip}", window_start)
        if ip_count > RATE_LIMIT_IP:
            raise HTTPException(
                status_code=429,
                detail="Too many requests from this network. Please try again later.",
            )


def _rate_limit_auth(request: Request) -> None:
    window_start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    ip = _get_client_ip(request)
    if not ip:
        return
    ip_count = increment_rate_limit(f"auth:{ip}", window_start)
    if ip_count > RATE_LIMIT_AUTH:
        raise HTTPException(
            status_code=429,
            detail="Too many authentication attempts. Please try again shortly.",
        )


def _window_start(minutes: int) -> datetime:
    now = _now_utc()
    bucket = (now.minute // minutes) * minutes
    return now.replace(minute=bucket, second=0, microsecond=0)


def _rate_limit_signup(request: Request) -> None:
    ip = _get_client_ip(request)
    if not ip:
        return
    window_start = _window_start(60)
    count = increment_rate_limit(f"signup:{ip}", window_start)
    if count > 5:
        raise HTTPException(
            status_code=429,
            detail="Too many signups from this network. Please try again later.",
        )


def _rate_limit_login_email(request: Request, email: str) -> None:
    window_start = _window_start(15)
    key = f"login:{email.lower()}"
    count = increment_rate_limit(key, window_start)
    if count > 5:
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Please wait before trying again.",
        )


def _rate_limit_support(request: Request) -> None:
    ip = _get_client_ip(request)
    if not ip:
        return
    window_start = _window_start(60)
    count = increment_rate_limit(f"support:{ip}", window_start)
    if count > 5:
        raise HTTPException(
            status_code=429,
            detail="Too many support requests. Please try again later.",
        )


def _dodo_return_url() -> str:
    return_url = _env("DODO_RETURN_URL")
    if not return_url:
        raise RuntimeError("DODO_RETURN_URL is not set.")
    return return_url


def _dodo_environment() -> str:
    env = _env("DODO_ENVIRONMENT")
    if env:
        return env
    starter_link = _env("DODO_STARTER_LINK") or ""
    if "test.checkout.dodopayments.com" in starter_link:
        return "test_mode"
    if "checkout.dodopayments.com" in starter_link:
        return "live_mode"
    return "live_mode" if _payments_enabled() else "test_mode"


def _dodo_api_key() -> str:
    api_key = _env("DODO_API_KEY")
    if not api_key:
        raise RuntimeError("DODO_API_KEY is not set.")
    return api_key


def _dodo_webhook_secret() -> str:
    secret = _env("DODO_WEBHOOK_SECRET") or _env("DODO_WEBHOOK_KEY")
    if not secret:
        raise RuntimeError("DODO_WEBHOOK_SECRET is not set.")
    return secret


def _dodo_api_base() -> str:
    override = _env("DODO_API_BASE_URL")
    if override:
        return override.rstrip("/")
    env = _dodo_environment()
    if env == "test_mode":
        return "https://test.dodopayments.com"
    return "https://live.dodopayments.com"


_validate_env()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _dodo_link_for_tier(tier: str) -> str:
    if tier == "STARTER":
        link = _env("DODO_STARTER_LINK")
    elif tier == "PRO":
        link = _env("DODO_PRO_LINK")
    else:
        link = None
    if not link:
        raise RuntimeError("Dodo payment link is not configured.")
    return link


def _dodo_metadata_token_ttl_seconds() -> int:
    return int(_env("DODO_METADATA_TOKEN_TTL_SECONDS", "86400") or 86400)


def _build_dodo_checkout_url(link: str, user_id: str, tier: str) -> str:
    parsed = urlparse(link)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params.setdefault("quantity", "1")
    params["redirect_url"] = _dodo_return_url()
    params["metadata_user_id"] = user_id
    params["metadata_tier"] = tier
    token = create_token(
        {"sub": user_id, "tier": tier, "jti": str(uuid.uuid4())},
        _auth_secret(),
        _dodo_metadata_token_ttl_seconds(),
    )
    token_hash = hash_token(token)
    expires_at = _now_utc() + timedelta(seconds=_dodo_metadata_token_ttl_seconds())
    cleanup_metadata_tokens(_now_utc())
    insert_metadata_token(token_hash=token_hash, user_id=user_id, tier=tier, expires_at=expires_at)
    params["metadata_token"] = token
    query = urlencode(params)
    return urlunparse(parsed._replace(query=query))


def _metadata_value(metadata: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return str(metadata[key])
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _extract_email(data: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    customer = data.get("customer") or {}
    email = customer.get("email") or data.get("email") or metadata.get("email")
    if not email:
        billing = data.get("billing") or {}
        email = billing.get("email")
    return str(email).strip().lower() if email else None


def _verify_dodo_webhook(raw_body: bytes, request: Request) -> None:
    """Verify webhook authenticity using standardwebhooks or HMAC SHA256."""
    secret = _dodo_webhook_secret()
    headers = {
        "webhook-id": request.headers.get("webhook-id", ""),
        "webhook-signature": request.headers.get("webhook-signature", ""),
        "webhook-timestamp": request.headers.get("webhook-timestamp", ""),
    }
    if headers["webhook-signature"]:
        Webhook(secret).verify(raw_body.decode("utf-8"), headers)
        return
    signature_header = (
        request.headers.get("x-dodo-signature")
        or request.headers.get("dodo-signature")
        or request.headers.get("x-signature")
    )
    if not signature_header:
        raise ValueError("Missing webhook signature header.")
    signature = signature_header.replace("sha256=", "").strip()
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("Invalid webhook signature.")


def _webhook_event_id(request: Request, payload: dict[str, Any], payment_id: str | None) -> str:
    return (
        request.headers.get("webhook-id")
        or payload.get("id")
        or payload.get("event_id")
        or payment_id
        or str(uuid.uuid4())
    )


def _product_id_from_link(link: str) -> str | None:
    path = urlparse(link).path.strip("/")
    parts = path.split("/")
    if "buy" in parts:
        idx = parts.index("buy")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return parts[-1] if parts else None


def _tier_from_product_id(product_id: str | None) -> str | None:
    if not product_id:
        return None
    starter_id = _product_id_from_link(_dodo_link_for_tier(TIER_STARTER))
    pro_id = _product_id_from_link(_dodo_link_for_tier(TIER_PRO))
    if starter_id and product_id == starter_id:
        return TIER_STARTER
    if pro_id and product_id == pro_id:
        return TIER_PRO
    return None


def _dodo_success(status: str, event_type: str) -> bool:
    if status in {"succeeded", "success", "paid", "completed"}:
        return True
    return "payment.succeeded" in event_type or "payment.completed" in event_type


def _dodo_refund(status: str, event_type: str) -> bool:
    if status in {"refunded", "refund", "chargeback"}:
        return True
    return "refunded" in event_type or "refund" in event_type


def _extract_product_id(data: dict[str, Any]) -> str | None:
    if "product_id" in data:
        return str(data["product_id"])
    cart = data.get("product_cart") or data.get("products") or []
    if isinstance(cart, list) and cart:
        first = cart[0]
        if isinstance(first, dict) and first.get("product_id"):
            return str(first["product_id"])
    return None


def _apply_payment_from_gateway(
    payload: dict[str, Any],
    user: dict[str, Any] | None,
    *,
    event_id: str | None,
    allow_email_match: bool = False,
) -> bool:
    data = payload.get("data") if "data" in payload else payload
    if not isinstance(data, dict):
        return False
    metadata = data.get("metadata") or payload.get("metadata") or {}
    status = str(
        (data.get("status") or data.get("payment_status") or payload.get("status") or "")
    ).lower()
    event_type = str(
        payload.get("type") or payload.get("event_type") or payload.get("event") or ""
    ).lower()
    payment_id = (
        data.get("payment_id")
        or data.get("id")
        or payload.get("payment_id")
        or payload.get("id")
    )
    email = _extract_email(data, metadata)
    paid_at = _parse_timestamp(data.get("paid_at") or data.get("created_at") or payload.get("created_at")) or _now_utc()

    tier = _metadata_value(metadata, "tier", "metadata_tier")
    if not tier:
        tier = _tier_from_product_id(_extract_product_id(data))
    tier = _normalize_tier(tier) if tier else None

    if allow_email_match and not user and email:
        user = get_user_by_email(email)

    if not user or not tier or tier == TIER_FREE:
        insert_unmatched_payment(
            entry_id=str(uuid.uuid4()),
            provider="dodo",
            payment_id=str(payment_id) if payment_id else None,
            email=email,
            status=status,
            raw=payload,
            reason="Unable to match payment to user/tier.",
        )
        logger.warning("Unmatched payment received from Dodo.")
        return False

    expiry = None
    if tier != TIER_FREE and _dodo_success(status, event_type):
        expiry = paid_at + timedelta(days=_tier_expiry_days())
    if _dodo_refund(status, event_type):
        tier = TIER_FREE
        expiry = None

    apply_payment_update(
        user_id=user["id"],
        tier=tier,
        status=status,
        payment_id=f"dodo:{payment_id or uuid.uuid4()}",
        raw=payload,
        payment_date=paid_at,
        expires_at=expiry,
        email=email,
        event_id=event_id,
        metadata_token_hash=None,
    )
    return True


def _fetch_recent_payment(email: str, payment_id: str | None) -> dict[str, Any] | None:
    if not _env("DODO_API_KEY"):
        raise HTTPException(status_code=501, detail="Manual resync requires DODO_API_KEY.")
    headers = {"Authorization": f"Bearer {_dodo_api_key()}"}
    base_url = _dodo_api_base()
    try:
        if payment_id:
            resp = requests.get(f"{base_url}/payments/{payment_id}", headers=headers, timeout=15)
        else:
            resp = requests.get(
                f"{base_url}/payments",
                headers=headers,
                params={"email": email, "limit": 10},
                timeout=15,
            )
    except requests.RequestException as exc:
        logger.exception("Failed to reach Dodo API.")
        raise HTTPException(status_code=502, detail="Payment provider unavailable.") from exc

    if resp.status_code >= 400:
        logger.error("Dodo API error: %s", resp.text)
        raise HTTPException(status_code=resp.status_code, detail="Payment lookup failed.")

    payload = resp.json()
    if payment_id:
        return payload.get("data") if isinstance(payload, dict) else payload
    items = []
    if isinstance(payload, dict):
        items = payload.get("data") or payload.get("payments") or payload.get("items") or []
    if isinstance(items, list):
        for item in items:
            status = str(item.get("status") or item.get("payment_status") or "").lower()
            if status in {"succeeded", "success", "paid", "completed"}:
                return item
        return items[0] if items else None
    return None


def _llm_kwargs(
    api_key: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int | None,
    headers: dict[str, str],
) -> dict:
    from langchain_openai import ChatOpenAI

    signature = inspect.signature(ChatOpenAI.__init__)
    params = signature.parameters
    kwargs: dict[str, object] = {"model": model}

    if "api_key" in params:
        kwargs["api_key"] = api_key
    if "openai_api_key" in params:
        kwargs["openai_api_key"] = api_key
    if "base_url" in params:
        kwargs["base_url"] = base_url
    if "openai_api_base" in params:
        kwargs["openai_api_base"] = base_url
    if "temperature" in params:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        if "max_tokens" in params:
            kwargs["max_tokens"] = max_tokens
        elif "max_completion_tokens" in params:
            kwargs["max_completion_tokens"] = max_tokens
    if headers and "default_headers" in params:
        kwargs["default_headers"] = headers

    return kwargs


@lru_cache(maxsize=1)
def _get_llm() -> "ChatOpenAI":
    from langchain_openai import ChatOpenAI

    api_key = _env("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    model = _env("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    base_url = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    temperature = float(_env("OPENROUTER_TEMPERATURE", "0.6") or 0.6)

    max_tokens = None
    max_tokens_raw = _env("OPENROUTER_MAX_TOKENS")
    if max_tokens_raw:
        max_tokens = int(max_tokens_raw)

    headers: dict[str, str] = {}
    app_url = _env("OPENROUTER_APP_URL")
    app_name = _env("OPENROUTER_APP_NAME")
    if app_url:
        headers["HTTP-Referer"] = app_url
    if app_name:
        headers["X-Title"] = app_name

    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_BASE_URL", base_url)
    os.environ.setdefault("OPENAI_API_BASE", base_url)

    return ChatOpenAI(
        **_llm_kwargs(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            headers=headers,
        )
    )


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise ValueError("Unable to parse JSON from model response.")


def _empty_tactical() -> TacticalEnhancements:
    return TacticalEnhancements(
        add_emotional_validation="",
        remove_emotional_validation="",
        include_action_items="",
        exclude_action_items="",
        add_softeners="",
        add_strengtheners="",
        insert_boundaries="",
        frame_as_question="",
        frame_as_statement="",
    )


def _empty_scenarios() -> OneClickScenarios:
    return OneClickScenarios(
        boundary_setting="",
        clarifying_question="",
        accountability_without_blame="",
        collaborative_proposal="",
        non_defensive_response="",
    )


def _empty_quick_actions() -> QuickActions:
    return QuickActions(
        condense="",
        expand="",
        cool_down="",
        add_assertiveness="",
    )


def _empty_red_flags() -> RedFlags:
    return RedFlags(
        manipulative_patterns=[],
        defensive_phrases=[],
        assumptions=[],
        escalation_triggers=[],
    )


def _apply_tier_limits(response: MessageCraftResponse, tier: str) -> MessageCraftResponse:
    if tier == TIER_FREE:
        response.tone_versions.diplomatic_tactful = ""
        response.tone_versions.casual_friendly = ""
        response.analysis.misinterpretation_warnings = []
        response.analysis.power_dynamics = "unknown"
        response.analysis.urgency = "low"
        response.analysis.framework_tags = []
        response.tactical_enhancements = _empty_tactical()
        response.quick_actions = _empty_quick_actions()
        response.one_click_scenarios = _empty_scenarios()
        response.red_flags = _empty_red_flags()

    if tier == TIER_STARTER:
        response.one_click_scenarios = _empty_scenarios()

    return response


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/signup", response_model=AuthResponse)
def signup(payload: SignupRequest, raw_request: Request, response: Response) -> AuthResponse:
    _rate_limit_auth(raw_request)
    _rate_limit_signup(raw_request)
    username = payload.username.strip().lower()
    email = payload.email.strip().lower()
    if not re.fullmatch(r"[a-z0-9._-]{3,32}", username):
        raise HTTPException(status_code=400, detail="Username must be 3-32 characters.")
    if not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
        raise HTTPException(status_code=400, detail="Email address is invalid.")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Email already in use.")

    user_id = str(uuid.uuid4())
    try:
        create_user(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=hash_password(payload.password),
            tier=TIER_FREE,
        )
    except psycopg.errors.UniqueViolation as exc:
        raise HTTPException(status_code=409, detail="Username already taken.") from exc

    verification_token = generate_refresh_token()
    verification_hash = hash_token(verification_token)
    expires_at = _now_utc() + timedelta(seconds=_email_verification_ttl_seconds())
    insert_email_verification(token_hash=verification_hash, user_id=user_id, expires_at=expires_at)
    try:
        _send_verification_email(email, verification_token)
    except Exception as exc:
        logger.exception("Failed to send verification email.")
        raise HTTPException(status_code=500, detail="Unable to send verification email.") from exc

    touch_session(user_id)
    access_token, refresh_token = _issue_tokens(user_id)
    if _wants_cookie_auth(raw_request):
        _set_auth_cookies(response, access_token, refresh_token)
    user = _user_response_from_row(
        {
            "id": user_id,
            "username": username,
            "tier": TIER_FREE,
            "email": email,
            "email_verified": False,
            "tier_expires_at": None,
        }
    )
    return AuthResponse(token=access_token, refresh_token=refresh_token, user=user)


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, raw_request: Request, response: Response) -> AuthResponse:
    _rate_limit_auth(raw_request)
    identifier = payload.username.strip().lower()
    user = get_user_by_email(identifier) if "@" in identifier else get_user_by_username(identifier)
    rate_key = user.get("email") if user and user.get("email") else identifier
    _rate_limit_login_email(raw_request, rate_key)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    touch_session(user["id"])
    access_token, refresh_token = _issue_tokens(user["id"])
    if _wants_cookie_auth(raw_request):
        _set_auth_cookies(response, access_token, refresh_token)
    user_response = _user_response_from_row(user)
    return AuthResponse(token=access_token, refresh_token=refresh_token, user=user_response)


@app.get("/api/auth/me", response_model=UserResponse)
def me(raw_request: Request) -> UserResponse:
    user = _require_user(raw_request, require_verified=False)
    return _user_response_from_row(user)


@app.post("/api/auth/refresh", response_model=AuthResponse)
def refresh_session(
    payload: RefreshRequest,
    raw_request: Request,
    response: Response,
) -> AuthResponse:
    refresh_token = _read_refresh_token(raw_request, payload)
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token.")
    token_hash = hash_token(refresh_token)
    token_row = get_refresh_token(token_hash)
    now = _now_utc()
    if (
        not token_row
        or token_row.get("revoked_at")
        or (token_row.get("expires_at") and token_row["expires_at"] < now)
    ):
        raise HTTPException(status_code=401, detail="Refresh token expired.")

    user_id = token_row["user_id"]
    revoke_refresh_token(token_hash, now)
    access_token, new_refresh_token = _issue_tokens(user_id)
    mark_refresh_token_used(token_hash, now)
    if _wants_cookie_auth(raw_request):
        _set_auth_cookies(response, access_token, new_refresh_token)
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return AuthResponse(
        token=access_token,
        refresh_token=new_refresh_token,
        user=_user_response_from_row(user),
    )


@app.post("/api/auth/logout")
def logout(raw_request: Request, response: Response) -> dict[str, bool]:
    refresh_token = _read_refresh_token(raw_request)
    if refresh_token:
        revoke_refresh_token(hash_token(refresh_token), _now_utc())
    _clear_auth_cookies(response)
    return {"ok": True}


@app.post("/api/auth/verify-email")
def verify_email(payload: VerifyEmailRequest) -> dict[str, bool]:
    token_hash = hash_token(payload.token.strip())
    record = get_email_verification(token_hash)
    now = _now_utc()
    if not record:
        raise HTTPException(status_code=400, detail="Invalid verification token.")
    if record.get("used_at"):
        return {"verified": True}
    expires_at = record.get("expires_at")
    if isinstance(expires_at, datetime) and now > expires_at:
        raise HTTPException(status_code=400, detail="Verification token expired.")
    complete_email_verification(record["user_id"], token_hash, now)
    return {"verified": True}


@app.post("/api/auth/resend-verification")
def resend_verification(raw_request: Request) -> dict[str, bool]:
    user = _require_user(raw_request, require_verified=False)
    if user.get("email_verified"):
        return {"sent": True}
    email = user.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email address is missing.")
    verification_token = generate_refresh_token()
    verification_hash = hash_token(verification_token)
    expires_at = _now_utc() + timedelta(seconds=_email_verification_ttl_seconds())
    insert_email_verification(token_hash=verification_hash, user_id=user["id"], expires_at=expires_at)
    _send_verification_email(email, verification_token)
    return {"sent": True}


@app.post("/api/auth/forgot-password")
def forgot_password(payload: ForgotPasswordRequest, raw_request: Request) -> dict[str, bool]:
    _rate_limit_auth(raw_request)
    email = payload.email.strip().lower()
    _rate_limit_login_email(raw_request, email)
    user = get_user_by_email(email)
    if user:
        reset_token = generate_refresh_token()
        reset_hash = hash_token(reset_token)
        expires_at = _now_utc() + timedelta(seconds=_password_reset_ttl_seconds())
        insert_password_reset(token_hash=reset_hash, user_id=user["id"], expires_at=expires_at)
        _send_password_reset_email(email, reset_token)
    return {"sent": True}


@app.post("/api/auth/reset-password")
def reset_password(payload: ResetPasswordRequest) -> dict[str, bool]:
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    token_hash = hash_token(payload.token.strip())
    record = get_password_reset(token_hash)
    now = _now_utc()
    if not record:
        raise HTTPException(status_code=400, detail="Invalid reset token.")
    if record.get("used_at"):
        raise HTTPException(status_code=400, detail="Reset token already used.")
    expires_at = record.get("expires_at")
    if isinstance(expires_at, datetime) and now > expires_at:
        raise HTTPException(status_code=400, detail="Reset token expired.")
    complete_password_reset(record["user_id"], token_hash, hash_password(payload.password), now)
    return {"reset": True}


@app.post("/api/support/contact")
def support_contact(payload: SupportRequest, raw_request: Request) -> dict[str, bool]:
    _rate_limit_support(raw_request)
    name = payload.name.strip()
    email = payload.email.strip().lower()
    subject = payload.subject.strip()
    message = payload.message.strip()
    if not name or not subject or not message:
        raise HTTPException(status_code=400, detail="Name, subject, and message are required.")
    if not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
        raise HTTPException(status_code=400, detail="Email address is invalid.")
    if len(message) > 5000:
        raise HTTPException(status_code=400, detail="Message is too long.")
    category = payload.category.strip() if payload.category else "general"
    order_id = payload.order_id.strip() if payload.order_id else "n/a"
    body = (
        "New support request\n"
        f"Name: {name}\n"
        f"Email: {email}\n"
        f"Category: {category}\n"
        f"Order ID: {order_id}\n\n"
        "Message:\n"
        f"{message}\n"
    )
    try:
        _send_email(_support_email_to(), f"[Support] {subject}", body, reply_to=email)
    except Exception as exc:
        logger.exception("Failed to send support email.")
        raise HTTPException(status_code=500, detail="Support email failed.") from exc
    return {"sent": True}


@app.post("/api/account/tier")
def update_tier(payload: TierUpdateRequest, raw_request: Request) -> dict[str, str]:
    _require_user(raw_request)
    raise HTTPException(status_code=403, detail="Tier changes are not allowed here.")


@app.post("/api/admin/account/tier")
def admin_update_tier(payload: AdminTierUpdateRequest, raw_request: Request) -> dict[str, str]:
    admin_secret = raw_request.headers.get("x-admin-secret")
    if not admin_secret or admin_secret != _admin_secret():
        raise HTTPException(status_code=403, detail="Forbidden.")
    tier = _normalize_tier(payload.tier)
    now = _now_utc()
    expires_at = None
    paid_at = None
    payment_id = None
    if tier != TIER_FREE:
        days = payload.days or _tier_expiry_days()
        expires_at = now + timedelta(days=days)
        paid_at = now
        payment_id = f"admin:{uuid.uuid4()}"
    update_user_tier(payload.user_id, tier, expires_at=expires_at, paid_at=paid_at, payment_id=payment_id)
    return {"tier": tier}


@app.get("/api/account/tier-status", response_model=TierStatusResponse)
def tier_status(raw_request: Request) -> TierStatusResponse:
    user = _require_user(raw_request, require_verified=False)
    expires_at = user.get("tier_expires_at")
    now = _now_utc()
    is_expired = isinstance(expires_at, datetime) and now > expires_at
    days_remaining = None
    if isinstance(expires_at, datetime) and not is_expired:
        days_remaining = max((expires_at - now).days, 0)
    return TierStatusResponse(
        current_tier=_normalize_tier(user.get("tier")),
        tier_expires_at=expires_at.isoformat() if isinstance(expires_at, datetime) else None,
        days_remaining=days_remaining,
        is_expired=is_expired,
    )


@app.post("/api/account/resync-payment")
def resync_payment(payload: ResyncPaymentRequest, raw_request: Request) -> dict[str, str]:
    if not _payments_enabled():
        raise HTTPException(status_code=503, detail="Payments are disabled.")
    if not _env("DODO_API_KEY"):
        raise HTTPException(status_code=501, detail="Manual resync requires DODO_API_KEY.")
    user = _require_user(raw_request)
    email = user.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required to verify payments.")
    payment = _fetch_recent_payment(email, payload.payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail="No recent payments found.")
    result = _apply_payment_from_gateway(payment, user, event_id="manual_resync")
    if not result:
        raise HTTPException(status_code=400, detail="Payment could not be verified.")
    return {"status": "resynced"}


@app.post("/api/payments/dodo/checkout-link", response_model=DodoCheckoutResponse)
def create_dodo_checkout_link(
    payload: DodoCheckoutRequest,
    raw_request: Request,
) -> DodoCheckoutResponse:
    if not _payments_enabled():
        raise HTTPException(status_code=503, detail="Payments are disabled.")
    user = _require_user(raw_request)
    tier = _normalize_tier(payload.tier)
    if tier not in {"STARTER", "PRO"}:
        raise HTTPException(status_code=400, detail="Unsupported tier.")
    current_tier = _normalize_tier(user.get("tier"))
    if current_tier == TIER_PRO and tier == TIER_STARTER:
        raise HTTPException(status_code=403, detail="Already on Pro.")
    checkout_url = _build_dodo_checkout_url(_dodo_link_for_tier(tier), user["id"], tier)
    return DodoCheckoutResponse(checkout_url=checkout_url)


@app.post("/api/payments/dodo/webhook")
async def dodo_webhook(request: Request) -> dict[str, bool]:
    if not _payments_enabled():
        return {"received": False}
    raw_body = await request.body()
    try:
        _verify_dodo_webhook(raw_body, request)
    except Exception as exc:
        insert_webhook_event(
            event_id=str(uuid.uuid4()),
            provider="dodo",
            event_type=None,
            status="invalid_signature",
            raw={"raw": raw_body.decode("utf-8", errors="replace")},
            error=str(exc),
        )
        _alert_admin("Dodo webhook verification failed", str(exc))
        raise HTTPException(status_code=400, detail="Invalid webhook signature.") from exc

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        insert_webhook_event(
            event_id=str(uuid.uuid4()),
            provider="dodo",
            event_type=None,
            status="invalid_payload",
            raw={"raw": raw_body.decode("utf-8", errors="replace")},
            error="Invalid JSON payload.",
        )
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    data = payload.get("data") if isinstance(payload, dict) else {}
    metadata = data.get("metadata") or payload.get("metadata") or {}
    status = str(
        (data.get("status") or data.get("payment_status") or payload.get("status") or "")
    ).lower()
    event_type = str(payload.get("type") or payload.get("event_type") or payload.get("event") or "").lower()
    payment_id = (
        data.get("payment_id")
        or data.get("id")
        or payload.get("payment_id")
        or payload.get("id")
    )
    event_id = _webhook_event_id(request, payload, str(payment_id) if payment_id else None)
    inserted = insert_webhook_event(
        event_id=event_id,
        provider="dodo",
        event_type=event_type or None,
        status=status or "unknown",
        raw=payload,
        error=None,
    )
    if not inserted:
        return {"received": True}

    cleanup_metadata_tokens(_now_utc())

    token = _metadata_value(metadata, "token", "metadata_token")
    token_hash = hash_token(token) if token else None
    token_row = get_metadata_token(token_hash) if token_hash else None
    if token:
        try:
            verify_token(token, _auth_secret())
        except ValueError:
            token_row = None

    if not token_row:
        token_hash = None

    user_id = _metadata_value(metadata, "user_id", "userId", "metadata_user_id")
    tier = _metadata_value(metadata, "tier", "metadata_tier")
    if token_row:
        if token_row.get("used_at") or (
            token_row.get("expires_at") and token_row["expires_at"] < _now_utc()
        ):
            token_row = None
        else:
            user_id = token_row["user_id"]
            tier = token_row["tier"]

    user = get_user_by_id(user_id) if user_id else None
    if not user:
        email = _extract_email(data, metadata)
        if email:
            user = get_user_by_email(email)
    if not user:
        insert_unmatched_payment(
            entry_id=str(uuid.uuid4()),
            provider="dodo",
            payment_id=str(payment_id) if payment_id else None,
            email=_extract_email(data, metadata),
            status=status,
            raw=payload,
            reason="Webhook missing user metadata.",
        )
        logger.warning("Webhook payment could not be matched to a user.")
        mark_webhook_event_processed(event_id, _now_utc())
        return {"received": True}

    processed = False
    try:
        if _dodo_refund(status, event_type) or _dodo_success(status, event_type):
            paid_at = _parse_timestamp(
                data.get("paid_at") or data.get("created_at") or payload.get("created_at")
            ) or _now_utc()
            tier_value = _normalize_tier(tier) if tier else None
            if not tier_value:
                tier_value = _tier_from_product_id(_extract_product_id(data)) or TIER_FREE
            expiry = paid_at + timedelta(days=_tier_expiry_days()) if tier_value != TIER_FREE else None
            if _dodo_refund(status, event_type):
                tier_value = TIER_FREE
                expiry = None
            apply_payment_update(
                user_id=user["id"],
                tier=tier_value,
                status=status,
                payment_id=f"dodo:{payment_id or uuid.uuid4()}",
                raw=payload,
                payment_date=paid_at,
                expires_at=expiry,
                email=_extract_email(data, metadata),
                event_id=event_id,
                metadata_token_hash=token_hash,
            )
        processed = True
    except Exception as exc:  # pragma: no cover
        _alert_admin("Dodo webhook processing failed", str(exc))
        raise HTTPException(status_code=500, detail="Webhook processing failed.") from exc
    if processed:
        mark_webhook_event_processed(event_id, _now_utc())

    return {"received": True}


@app.get("/api/usage", response_model=UsageResponse)
def get_usage(raw_request: Request) -> UsageResponse:
    user = _require_user(raw_request)
    tier = _effective_tier(raw_request, user)
    user_id = user["id"]
    _rate_limit(raw_request, tier, user_id)
    touch_session(user_id)
    period_start = get_day_start() if tier == TIER_FREE else get_week_start()
    count = get_usage_count(user_id, period_start)
    meta = _build_usage_meta(tier, count, period_start)
    return UsageResponse(**meta.model_dump())


@app.post("/api/messagecraft", response_model=MessageCraftResponse)
def messagecraft(request: MessageCraftRequest, raw_request: Request) -> MessageCraftResponse:
    user = _require_user(raw_request)
    tier = _effective_tier(raw_request, user)
    user_id = user["id"]
    _rate_limit(raw_request, tier, user_id)
    touch_session(user_id)

    period_start = get_day_start() if tier == TIER_FREE else get_week_start()
    limit = TIER_LIMITS[tier]
    count = get_usage_count(user_id, period_start)
    if limit is not None and count >= limit:
        reset_at = (
            get_next_day_start(period_start)
            if tier == TIER_FREE
            else get_next_week_start(period_start)
        ).isoformat()
        raise HTTPException(
            status_code=429,
            detail={"message": "Translation limit reached.", "reset_at": reset_at},
        )

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing input text.")

    prompt = _build_messagecraft_prompt(request)
    messages = [SystemMessage(content=prompt), HumanMessage(content=text)]

    try:
        response = _get_llm().invoke(messages)
        raw = getattr(response, "content", None) or str(response)
        data = _extract_json(raw)
        parsed = MessageCraftResponse.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail="Invalid response format from model.") from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    new_count = increment_usage(user_id, period_start, 1)
    parsed.meta = _build_usage_meta(tier, new_count, period_start)
    return _apply_tier_limits(parsed, tier)


@app.post("/api/conversations")
def create_conversation(entry: ConversationEntryRequest, raw_request: Request) -> dict[str, str]:
    user = _require_user(raw_request)
    tier = _effective_tier(raw_request, user)
    user_id = user["id"]
    _rate_limit(raw_request, tier, user_id)
    if tier == TIER_FREE:
        raise HTTPException(status_code=403, detail="Conversation memory is not available on Free.")

    contact = entry.contact.strip()
    if not contact:
        raise HTTPException(status_code=400, detail="Contact is required.")

    touch_session(user_id)

    limit = CONTACT_LIMITS[tier]
    if limit is not None and not contact_exists(user_id, contact):
        current_contacts = count_contacts(user_id)
        if current_contacts >= limit:
            raise HTTPException(
                status_code=403,
                detail=f"Contact limit reached for {tier}.",
            )

    insert_conversation_entry(
        entry_id=str(uuid.uuid4()),
        session_id=user_id,
        contact=contact,
        input_text=entry.input_text,
        output_text=entry.output_text,
        tone_key=entry.tone_key,
        tone_scores=entry.tone_scores.model_dump(),
        impact_prediction=entry.impact_prediction,
        clarity_before=entry.clarity_before,
        clarity_after=entry.clarity_after,
    )

    return {"status": "saved"}


@app.get("/api/conversations")
def list_conversations(raw_request: Request, contact: str | None = None) -> dict[str, Any]:
    user = _require_user(raw_request)
    tier = _effective_tier(raw_request, user)
    user_id = user["id"]
    _rate_limit(raw_request, tier, user_id)
    if tier == TIER_FREE:
        return {"entries": []}

    touch_session(user_id)
    entries = fetch_conversation_entries(session_id=user_id, contact=contact)
    return {"entries": entries}


@app.post("/api/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest, raw_request: Request) -> TranslateResponse:
    user = _require_user(raw_request)
    tier = _effective_tier(raw_request, user)
    _rate_limit(raw_request, tier, user["id"])
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing input text.")

    mode = request.mode.strip().lower()
    prompt = _build_prompt(mode)
    messages = [SystemMessage(content=prompt), HumanMessage(content=text)]

    try:
        response = _get_llm().invoke(messages)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    content = getattr(response, "content", None) or str(response)
    return TranslateResponse(output=content)
