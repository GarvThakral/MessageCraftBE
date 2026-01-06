from __future__ import annotations

import inspect
import json
import os
import re
from functools import lru_cache
from typing import Any

import stripe
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()


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


class TranslateRequest(BaseModel):
    text: str
    mode: str


class TranslateResponse(BaseModel):
    output: str


class CheckoutSessionRequest(BaseModel):
    plan: str
    billing_cycle: str
    success_url: str | None = None
    cancel_url: str | None = None


class CheckoutSessionResponse(BaseModel):
    url: str


class PortalSessionRequest(BaseModel):
    customer_id: str
    return_url: str | None = None


class PortalSessionResponse(BaseModel):
    url: str


app = FastAPI(title="MessageCraft Pro Backend")


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
    audience = request.audience_style or "neutral"
    tone_balance = "balanced" if request.tone_balance is None else str(request.tone_balance)
    goal = request.user_goal or "general clarity and respect"

    return (
        "You are MessageCraft Pro, an expert communication strategist. "
        "Analyze the input message and return ONE JSON object that matches the schema. "
        "Do not include markdown or extra text.\n\n"
        f"Audience style target: {audience}. "
        f"Tone balance target (0-100, low=emotional high=logical): {tone_balance}. "
        f"Primary user goal: {goal}.\n\n"
        "Rules:"
        "\n- Provide 5 complete rewrites in tone_versions."
        "\n- Scores are integers 0-100. after_clarity_score must be >= before_clarity_score."
        "\n- relationship_impact_prediction is 0-100 percent chance of positive response."
        "\n- Keep warnings short, actionable, and non-judgmental."
        "\n- If no red flags exist, return empty arrays."
        "\n- Use plain text, avoid emojis unless the original text includes them."
        "\n\n"
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
def _get_llm() -> ChatOpenAI:
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


def _price_id(plan: str, billing_cycle: str) -> str:
    plan_key = plan.upper()
    cycle_key = billing_cycle.lower()

    if plan_key == "STARTER" and cycle_key == "weekly":
        return _env("STRIPE_STARTER_WEEKLY_PRICE_ID", "") or ""
    if plan_key == "STARTER" and cycle_key == "monthly":
        return _env("STRIPE_STARTER_MONTHLY_PRICE_ID", "") or ""
    if plan_key == "PRO" and cycle_key == "weekly":
        return _env("STRIPE_PRO_WEEKLY_PRICE_ID", "") or ""
    if plan_key == "PRO" and cycle_key == "monthly":
        return _env("STRIPE_PRO_MONTHLY_PRICE_ID", "") or ""

    raise HTTPException(status_code=400, detail="Unsupported plan or billing cycle.")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/messagecraft", response_model=MessageCraftResponse)
def messagecraft(request: MessageCraftRequest) -> MessageCraftResponse:
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

    return parsed


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


@app.post("/api/stripe/checkout-session", response_model=CheckoutSessionResponse)
def create_checkout_session(payload: CheckoutSessionRequest) -> CheckoutSessionResponse:
    stripe.api_key = _env("STRIPE_SECRET_KEY")
    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Stripe is not configured.")

    price_id = _price_id(payload.plan, payload.billing_cycle)
    if not price_id:
        raise HTTPException(status_code=500, detail="Stripe price ID is missing.")

    success_url = payload.success_url or _env("STRIPE_SUCCESS_URL", "http://localhost:5173")
    cancel_url = payload.cancel_url or _env("STRIPE_CANCEL_URL", "http://localhost:5173/pricing")

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=f"{success_url}?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=cancel_url,
        allow_promotion_codes=True,
    )

    return CheckoutSessionResponse(url=session.url)


@app.get("/api/stripe/session")
def get_checkout_session(session_id: str) -> dict[str, str]:
    stripe.api_key = _env("STRIPE_SECRET_KEY")
    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Stripe is not configured.")

    session = stripe.checkout.Session.retrieve(session_id)
    price_id = ""
    customer_id = ""
    if session and session.get("subscription"):
        subscription = stripe.Subscription.retrieve(session["subscription"])
        if subscription and subscription.get("items"):
            price_id = subscription["items"]["data"][0]["price"]["id"]
    if session and session.get("customer"):
        customer_id = session["customer"]

    tier = "FREE"
    if price_id in (
        _env("STRIPE_STARTER_WEEKLY_PRICE_ID"),
        _env("STRIPE_STARTER_MONTHLY_PRICE_ID"),
    ):
        tier = "STARTER"
    if price_id in (
        _env("STRIPE_PRO_WEEKLY_PRICE_ID"),
        _env("STRIPE_PRO_MONTHLY_PRICE_ID"),
    ):
        tier = "PRO"

    return {"tier": tier, "customer_id": customer_id}


@app.post("/api/stripe/portal-session", response_model=PortalSessionResponse)
def create_portal_session(payload: PortalSessionRequest) -> PortalSessionResponse:
    stripe.api_key = _env("STRIPE_SECRET_KEY")
    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Stripe is not configured.")

    if not payload.customer_id:
        raise HTTPException(status_code=400, detail="Customer ID is required.")

    return_url = payload.return_url or _env("STRIPE_PORTAL_RETURN_URL", "http://localhost:5173")

    session = stripe.billing_portal.Session.create(
        customer=payload.customer_id,
        return_url=return_url,
    )

    return PortalSessionResponse(url=session.url)


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request) -> dict[str, bool]:
    stripe.api_key = _env("STRIPE_SECRET_KEY")
    webhook_secret = _env("STRIPE_WEBHOOK_SECRET")
    if not stripe.api_key or not webhook_secret:
        raise HTTPException(status_code=500, detail="Stripe webhook not configured.")

    payload = await request.body()
    signature = request.headers.get("stripe-signature")

    try:
        stripe.Webhook.construct_event(payload, signature, webhook_secret)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"received": True}
