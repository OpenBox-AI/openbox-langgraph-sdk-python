# LangGraph Test Agent (OpenBox)

A minimal LangGraph agent for validating OpenBox governance using `openbox-langgraph-sdk`.

## Setup

1. Copy env file:

```bash
cp .env.example .env
```

2. Set:

- `OPENBOX_URL`
- `OPENBOX_API_KEY`
- `OPENAI_API_KEY`

3. Run:

```bash
uv run python agent.py
```

## Debugging

- Set `OPENBOX_DEBUG=1` to print all governance requests/responses and raw LangGraph events.
