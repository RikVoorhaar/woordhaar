<p align="center">
  <img src="./banner.png" alt="Woordhaar" width="600" />
</p>

# Woordhaar

A cross-lingual dictionary that accepts a word in **Danish**, **Dutch**, or **English** and returns ranked translations into the other two languages—with definitions, examples, and cognate information. Dictionary lookups use local SQLite; translation candidates are generated and validated via a local LLM (Ollama).

## Stack

| Layer     | Technology              |
| --------- | ----------------------- |
| Frontend  | SvelteKit               |
| Backend   | FastAPI (async)         |
| Database  | SQLite (aiosqlite)      |
| Lemmatizer| simplemma               |
| LLM       | Ollama (local)          |

## What to Expect

- **Backend** — FastAPI app with `/api/translate`, `/api/lookup/{lang}/{word}`, and `/api/health`
- **Frontend** — Single-page UI with three input fields (DA/NL/EN), showing ranked translations per sense
- **Data** — Ingestion pipeline for kaikki.org, COR.SEM, and related sources into SQLite
- **Pipeline** — Lemmatize → dictionary lookups → LLM generation → LLM ranking → `TranslationResult`

See [`plan.md`](./plan.md) for the full implementation plan and architecture.

## Setup

### Step 1 — Data ingestion

1. Install deps and sync env: `uv sync`
2. Run ingestion (downloads ~3GB, populates `woordhaar.db`):
   ```bash
   uv run woordhaar-ingest
   ```
   Options: `--data-dir DIR`, `--db PATH`, `--skip-download` (use existing files), `--skip-corsem`

   Data sources: kaikki.org (Danish, Dutch, English), COR.SEM (Danish). Place `cor.sem.1.0.tsv` in `data/` if the COR.SEM download fails, or use `--skip-corsem`.

### Step 2 — Running the Application

**Backend (FastAPI):**

```bash
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`. OpenAPI docs at `http://127.0.0.1:8000/docs`.

**Frontend (SvelteKit):**

```bash
cd frontend
bun run dev
```

The frontend will be available at `http://localhost:5173` (default SvelteKit port).

**Environment Variables (Frontend):**

Optionally create `frontend/.env` to override backend URL:
```bash
PUBLIC_API_HOST=127.0.0.1
PUBLIC_API_PORT=8000
```

**Note:** Ensure Ollama is running locally (e.g. `ollama pull qwen3.5:35b-a3b`). The backend checks Ollama availability on startup.
