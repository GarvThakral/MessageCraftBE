# MessageCraft Pro Backend

## Setup
1. Create a virtual environment.
2. Install dependencies.
3. Copy `.env.example` to `.env` and set your OpenRouter + DATABASE_URL values.

## Run
```bash
uvicorn main:app --reload --port 8000
```

## Endpoints
- `POST /api/messagecraft` returns the full MessageCraft analysis bundle.
- `GET /api/usage` returns the weekly usage count for the current session + tier.
- `POST /api/conversations` stores conversation history for the current session.
- `GET /api/conversations` returns recent conversation entries.
- `POST /api/translate` returns a simple logic/emotion rewrite.
