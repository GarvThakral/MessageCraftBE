from __future__ import annotations

import inspect
import json
import os
import re
import uuid
from datetime import date
from functools import lru_cache
from typing import Any, TYPE_CHECKING

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from db import (
    contact_exists,
    count_contacts,
    fetch_conversation_entries,
    get_next_week_start,
    get_usage_count,
    get_week_start,
    increment_usage,
    insert_conversation_entry,
    touch_session,
)

load_dotenv()

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

app = FastAPI(title="MessageCraft Pro Backend")

TIER_LIMITS = {
    "FREE": 3,
    "STARTER": 25,
    "PRO": None,
}

TIER_TONE_LIMITS = {
    "FREE": 3,
    "STARTER": 5,
    "PRO": 5,
}

CONTACT_LIMITS = {
    "FREE": 0,
    "STARTER": 5,
    "PRO": None,
}


def _parse_origins(value: str | None) -> list[str]:
    if not value or value.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in value.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_tier(value: str | None) -> str:
    if not value:
        return "FREE"
    normalized = value.strip().upper()
    if normalized in TIER_LIMITS:
        return normalized
    return "FREE"


def _get_session_id(request: Request) -> str:
    session_id = request.headers.get("x-session-id") or request.headers.get("X-Session-Id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session id.")
    return session_id.strip()


def _get_tier(request: Request) -> str:
    return _normalize_tier(request.headers.get("x-user-tier") or request.headers.get("X-User-Tier"))


def _build_usage_meta(tier: str, count: int, week_start: date) -> UsageMeta:
    limit = TIER_LIMITS[tier]
    reset_at = get_next_week_start(week_start).isoformat()
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
        "Preserve intent. Do not add new facts, decisions, or requests. "
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
            "Keep it concise, respectful, and specific."
        )
    if mode == "emotion":
        return (
            "Convert overly logical or dismissive text into warm, emotionally "
            "validating language. Keep it concise, genuine, and non-blaming."
        )
    raise HTTPException(status_code=400, detail="Unsupported mode.")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


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
    if tier == "FREE":
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

    if tier == "STARTER":
        response.one_click_scenarios = _empty_scenarios()

    return response


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/usage", response_model=UsageResponse)
def get_usage(raw_request: Request) -> UsageResponse:
    session_id = _get_session_id(raw_request)
    tier = _get_tier(raw_request)
    touch_session(session_id)
    week_start = get_week_start()
    count = get_usage_count(session_id, week_start)
    meta = _build_usage_meta(tier, count, week_start)
    return UsageResponse(**meta.model_dump())


@app.post("/api/messagecraft", response_model=MessageCraftResponse)
def messagecraft(request: MessageCraftRequest, raw_request: Request) -> MessageCraftResponse:
    session_id = _get_session_id(raw_request)
    tier = _get_tier(raw_request)
    touch_session(session_id)

    week_start = get_week_start()
    limit = TIER_LIMITS[tier]
    count = get_usage_count(session_id, week_start)
    if limit is not None and count >= limit:
        reset_at = get_next_week_start(week_start).isoformat()
        raise HTTPException(
            status_code=429,
            detail={"message": "Weekly translation limit reached.", "reset_at": reset_at},
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

    new_count = increment_usage(session_id, week_start, 1)
    parsed.meta = _build_usage_meta(tier, new_count, week_start)
    return _apply_tier_limits(parsed, tier)


@app.post("/api/conversations")
def create_conversation(entry: ConversationEntryRequest, raw_request: Request) -> dict[str, str]:
    session_id = _get_session_id(raw_request)
    tier = _get_tier(raw_request)
    if tier == "FREE":
        raise HTTPException(status_code=403, detail="Conversation memory is not available on Free.")

    contact = entry.contact.strip()
    if not contact:
        raise HTTPException(status_code=400, detail="Contact is required.")

    touch_session(session_id)

    limit = CONTACT_LIMITS[tier]
    if limit is not None and not contact_exists(session_id, contact):
        current_contacts = count_contacts(session_id)
        if current_contacts >= limit:
            raise HTTPException(
                status_code=403,
                detail=f"Contact limit reached for {tier}.",
            )

    insert_conversation_entry(
        entry_id=str(uuid.uuid4()),
        session_id=session_id,
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
    session_id = _get_session_id(raw_request)
    tier = _get_tier(raw_request)
    if tier == "FREE":
        return {"entries": []}

    touch_session(session_id)
    entries = fetch_conversation_entries(session_id=session_id, contact=contact)
    return {"entries": entries}


@app.post("/api/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest) -> TranslateResponse:
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
