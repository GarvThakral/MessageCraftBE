# MessageCraft Pro Backend

## Setup
1. Create a virtual environment.
2. Install dependencies.
3. Copy `.env.example` to `.env` and set your OpenRouter + DATABASE_URL + AUTH_SECRET + DODO_* values.

## Run
```bash
uvicorn main:app --reload --port 8000
```

## Endpoints
- `POST /api/auth/signup` creates a user and returns a token.
- `POST /api/auth/login` returns a token for existing users.
- `GET /api/auth/me` returns the current user.
- `POST /api/messagecraft` returns the full MessageCraft analysis bundle.
- `GET /api/usage` returns the weekly usage count for the current session + tier.
- `POST /api/conversations` stores conversation history for the current session.
- `GET /api/conversations` returns recent conversation entries.
- `POST /api/translate` returns a simple logic/emotion rewrite.
- `POST /api/account/tier` updates the user tier (manual overrides for now).
- `POST /api/payments/dodo/checkout-link` returns a hosted Dodo checkout URL.
- `POST /api/payments/dodo/webhook` processes Dodo webhook events.
