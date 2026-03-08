"""FastAPI application for Woordhaar translation API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import aiosqlite
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator

from backend.config import DB_PATH, OLLAMA_BASE_URL
from backend.models import ErrorResponse, TranslationResult
from backend.providers.base import DictionaryEntry
from backend.pipeline import TranslationPipeline

Lang = Literal["da", "nl", "en"]


class TranslateRequest(BaseModel):
    """Request body for /api/translate."""

    word: str
    language: Lang

    @model_validator(mode="after")
    def check_word_not_empty(self) -> "TranslateRequest":
        if not self.word or not self.word.strip():
            raise ValueError("word cannot be empty")
        return self

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify DB and Ollama; shutdown: cleanup."""
    db = _resolve_db_path()
    try:
        async with aiosqlite.connect(db) as conn:
            for table in ("da_entries", "nl_entries", "en_entries"):
                cursor = await conn.execute(f"SELECT count(*) FROM {table}")
                await cursor.fetchone()
    except Exception as e:
        raise RuntimeError(f"Database unavailable: {e}") from e

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
            app.state.ollama_ok = r.status_code == 200 and bool((r.json() or {}).get("models"))
    except Exception:
        app.state.ollama_ok = False

    app.state.pipeline = TranslationPipeline(db_path=db)
    app.state.db_path = db
    yield
    # shutdown: no persistent connections to close


app = FastAPI(
    title="Woordhaar API",
    description="Cross-lingual dictionary for Danish, Dutch, and English",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_db_path() -> Path:
    """Resolve DB path relative to project root (backend/..)."""
    p = Path(DB_PATH)
    if p.is_absolute():
        return p
    root = Path(__file__).resolve().parent.parent
    return root / p


async def _get_db_counts() -> dict[str, int]:
    """Return row counts for da_entries, nl_entries, en_entries."""
    db = app.state.db_path
    result: dict[str, int] = {}
    async with aiosqlite.connect(db) as conn:
        for lang, table in [("da", "da_entries"), ("nl", "nl_entries"), ("en", "en_entries")]:
            cursor = await conn.execute(f"SELECT count(*) FROM {table}")
            row = await cursor.fetchone()
            result[lang] = row[0] if row else 0
    return result


@app.post(
    "/api/translate",
    response_model=TranslationResult,
    responses={422: {"model": ErrorResponse, "description": "Validation error"}},
)
async def translate(req: TranslateRequest) -> TranslationResult:
    """Run the translation pipeline for a word in the given language."""
    pipeline: TranslationPipeline = app.state.pipeline
    return await pipeline.translate(req.word, req.language)


@app.get(
    "/api/lookup/{lang}/{word}",
    response_model=list[DictionaryEntry],
    responses={422: {"model": ErrorResponse, "description": "Invalid language"}},
)
async def lookup(lang: Lang, word: str) -> list[DictionaryEntry]:
    """Raw dictionary lookup for a word (debug/testing)."""
    pipeline: TranslationPipeline = app.state.pipeline
    return await pipeline.provider.lookup(word, lang)


@app.get("/api/health")
async def health() -> dict:
    """Health check: status, Ollama reachability, DB entry counts."""
    counts = await _get_db_counts()
    return {
        "status": "ok",
        "ollama": app.state.ollama_ok,
        "db_entries": counts,
    }
