# Variance Explanation Assistant (Track 4, Finance)

**User:** a finance analyst at month-end review, asked "Why did we miss the forecast?"
**JTBD:** When I see variance in numbers, I want to explain what caused it clearly and quickly, so I can defend my analysis.
**Key failure mode guarded against:** inventing causes the data does not support.

## How it works
Structured form -> validation -> **logic layer (before AI)** -> prompt -> LLM -> parse -> confidence/flag check -> SQLite -> report page -> admin dashboard.

| Stage | File |
|---|---|
| Input form | `templates/index.html` |
| Validation, variance %, materiality, ranking, price/volume split, rule-based classification | `logic.py` |
| Prompt build, LLM call (OpenAI or Anthropic), parsing, grounding checks, confidence blend | `llm.py` |
| Persistence and dashboard stats | `db.py` |
| Routes and pipeline | `app.py` |

### Guardrails against invented causes
- The model only sees computed facts, never raw free text it could embellish.
- `volume`/`price` are rejected unless units and prices were supplied; `timing` unless the note says so. Otherwise downgraded to `unclear`.
- Any number in the AI text that is not in the supplied figures is flagged and caps confidence.
- Confidence = 60% model self-rating + 40% evidence score from the logic layer, capped when top drivers explain under 60% of the variance.
- If the AI call fails, the report still saves with calculated figures and a visible notice.

## Run locally
```
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY or ANTHROPIC_API_KEY
export $(grep -v '^#' .env | xargs)
python app.py          # http://localhost:5000
python -m pytest tests # or run tests/test_app.py functions directly
```
Admin dashboard: `/admin` (password from `ADMIN_PASSWORD`).

## Deploy
Any Python host works (Render, Railway, Fly). Start command: `gunicorn app:app`. Set the env vars from `.env.example`. SQLite needs a persistent disk; otherwise point `DB_PATH` at a mounted volume.
