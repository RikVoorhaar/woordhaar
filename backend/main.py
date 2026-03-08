"""FastAPI application for Woordhaar translation API."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import aiosqlite
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, model_validator

from backend.config import DB_PATH, LLM_MODEL, LOG_LEVEL, OLLAMA_BASE_URL
from backend.logging_config import setup_logging
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
    # Setup logging first
    setup_logging(log_level=LOG_LEVEL)
    logger.info("Starting Woordhaar API")

    db = _resolve_db_path()
    logger.info(f"Database path: {db}")
    
    try:
        async with aiosqlite.connect(db) as conn:
            for table in ("da_entries", "nl_entries", "en_entries"):
                cursor = await conn.execute(f"SELECT count(*) FROM {table}")
                row = await cursor.fetchone()
                count = row[0] if row else 0
                logger.info(f"Table {table}: {count} entries")
    except Exception as e:
        logger.error(f"Database unavailable: {e}", exc_info=True)
        raise RuntimeError(f"Database unavailable: {e}") from e

    logger.info(f"Checking Ollama availability at {OLLAMA_BASE_URL}")
    app.state.ollama_ok = False
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
            if r.status_code == 200:
                data = r.json() or {}
                models = data.get("models", [])
                if models:
                    model_names = [m.get("name") for m in models if m.get("name")]
                    app.state.ollama_ok = True
                    logger.info(f"Ollama is available with {len(model_names)} model(s)")
                    
                    # Verify configured model exists
                    if LLM_MODEL not in model_names:
                        logger.error(
                            f"Configured model '{LLM_MODEL}' not found in Ollama. "
                            f"Available models: {', '.join(model_names[:5])}"
                            + (f" (and {len(model_names) - 5} more)" if len(model_names) > 5 else "")
                        )
                    else:
                        logger.info(f"Configured model '{LLM_MODEL}' is available")
                else:
                    logger.warning("Ollama responded but no models found")
            else:
                logger.warning(f"Ollama responded with status {r.status_code}")
    except Exception as e:
        logger.warning(f"Ollama unavailable: {e}")

    app.state.pipeline = TranslationPipeline(db_path=db)
    app.state.db_path = db
    logger.info("Pipeline initialized, application ready")
    
    yield
    
    logger.info("Shutting down Woordhaar API")
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
    t0 = time.perf_counter()
    logger.info(f"Translation request: word='{req.word}', language='{req.language}'")
    
    try:
        pipeline: TranslationPipeline = app.state.pipeline
        result = await pipeline.translate(req.word, req.language)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            f"Translation completed: word='{req.word}', language='{req.language}', "
            f"senses={len(result.senses)}, time={elapsed_ms}ms"
        )
        return result
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            f"Translation failed: word='{req.word}', language='{req.language}', "
            f"time={elapsed_ms}ms",
            exc_info=True
        )
        raise


@app.get(
    "/api/lookup/{lang}/{word}",
    response_model=list[DictionaryEntry],
    responses={422: {"model": ErrorResponse, "description": "Invalid language"}},
)
async def lookup(lang: Lang, word: str) -> list[DictionaryEntry]:
    """Raw dictionary lookup for a word (debug/testing)."""
    logger.info(f"Dictionary lookup: word='{word}', language='{lang}'")
    try:
        pipeline: TranslationPipeline = app.state.pipeline
        entries = await pipeline.provider.lookup(word, lang)
        logger.info(f"Dictionary lookup completed: word='{word}', language='{lang}', entries={len(entries)}")
        return entries
    except Exception as e:
        logger.error(f"Dictionary lookup failed: word='{word}', language='{lang}'", exc_info=True)
        raise


@app.get("/api/health")
async def health() -> dict:
    """Health check: status, Ollama reachability, DB entry counts."""
    logger.debug("Health check requested")
    try:
        counts = await _get_db_counts()
        result = {
            "status": "ok",
            "ollama": app.state.ollama_ok,
            "db_entries": counts,
        }
        logger.debug(f"Health check: {result}")
        return result
    except Exception as e:
        logger.error("Health check failed", exc_info=True)
        raise
