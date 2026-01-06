# MessageCraft Pro Backend

## Setup
1. Create a virtual environment.
2. Install dependencies.
3. Copy `.env.example` to `.env` and set your OpenRouter + Stripe values.

## Run
```bash
uvicorn main:app --reload --port 8000
```

## Endpoints
- `POST /api/messagecraft` returns the full MessageCraft analysis bundle.
- `POST /api/translate` returns a simple logic/emotion rewrite.
- `POST /api/stripe/checkout-session` creates a Stripe checkout session.
- `GET /api/stripe/session` verifies a session and returns the tier.
- `POST /api/stripe/portal-session` creates a Stripe customer portal session.
