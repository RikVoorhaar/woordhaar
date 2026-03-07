"""Tests for LLM service (Ollama)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.models import (
    LLMTranslationResult,
    RankedTranslation,
    TranslationCandidate,
    TranslationContext,
)
from backend.providers.base import DictionaryEntry
from backend.services import OllamaLLMService
from backend.services.base import LLMUnavailableError


pytestmark = pytest.mark.asyncio


async def _ollama_chat_reachable() -> tuple[bool, str | None]:
    """Check Ollama /api/chat with an available model. Returns (ok, model_name)."""
    try:
        async with httpx.AsyncClient() as client:
            tags = await client.get("http://localhost:11434/api/tags", timeout=2.0)
            if tags.status_code != 200:
                return False, None
            data = tags.json()
            models = data.get("models") or []
            if not models:
                return False, None
            model = models[0]["name"]
            r = await client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "."}],
                    "stream": False,
                },
                timeout=10.0,
            )
            if r.status_code == 404:
                return False, None  # endpoint missing
            return r.status_code == 200, model
    except Exception:
        return False, None


@pytest.fixture
async def ollama_service():
    """Service using first available Ollama model (or default if unreachable)."""
    ok, model = await _ollama_chat_reachable()
    return OllamaLLMService(model=model) if (ok and model) else OllamaLLMService()


@pytest.fixture
def translation_context():
    return TranslationContext(
        word="hund",
        pos="noun",
        definitions=["a domesticated carnivorous mammal", "canine"],
        etymology=None,
        known_candidates={"en": ["dog", "hound"], "nl": ["hond"]},
    )


async def test_generate_translations_returns_valid_result(
    ollama_service,
    translation_context,
):
    """generate_translations returns valid LLMTranslationResult with >=1 candidate per target lang."""
    if not await _ollama_chat_reachable():
        pytest.skip("Ollama /api/chat not available (run Ollama with a model, e.g. llama3.2)")
    result = await ollama_service.generate_translations(
        "hund", "da", ["en", "nl"], translation_context
    )
    assert isinstance(result, LLMTranslationResult)
    en_cands = [t for t in result.translations if t.language == "en"]
    nl_cands = [t for t in result.translations if t.language == "nl"]
    assert len(en_cands) >= 1
    assert len(nl_cands) >= 1
    for t in result.translations:
        assert t.word
        assert t.language in ("en", "nl")
        assert t.semantic_precision in ("exact", "near", "broader", "narrower")


async def test_output_parses_into_pydantic(ollama_service, translation_context):
    """LLM JSON parses into Pydantic without validation errors."""
    if not await _ollama_chat_reachable():
        pytest.skip("Ollama /api/chat not available (run Ollama with a model, e.g. llama3.2)")
    result = await ollama_service.generate_translations(
        "hund", "da", ["en", "nl"], translation_context
    )
    LLMTranslationResult.model_validate(result.model_dump())


async def test_response_time(ollama_service, translation_context):
    """Response time <10s on capable hardware."""
    if not await _ollama_chat_reachable():
        pytest.skip("Ollama /api/chat not available (run Ollama with a model, e.g. llama3.2)")
    t0 = time.perf_counter()
    await ollama_service.generate_translations(
        "hund", "da", ["en", "nl"], translation_context
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0


async def test_ollama_unreachable_raises_llm_unavailable():
    """When Ollama is unreachable, raises LLMUnavailableError."""
    service = OllamaLLMService(base_url="http://127.0.0.1:19999", timeout=1)
    ctx = TranslationContext(word="hund", definitions=["dog"])
    with pytest.raises(LLMUnavailableError):
        await service.generate_translations("hund", "da", ["en"], ctx)


async def test_filter_and_rank_removes_keep_false_sorted_by_rank():
    """filter_and_rank returns only keep=True, sorted by rank."""
    service = OllamaLLMService()
    source_entry = DictionaryEntry(
        word="hund", language="da", definitions=["a canine"]
    )
    candidates = [
        TranslationCandidate(word="dog", language="en", definition="canine"),
        TranslationCandidate(word="hound", language="en", definition="hunting dog"),
        TranslationCandidate(word="pooch", language="en", definition="dog"),
    ]
    target_entries = {
        "en": [
            DictionaryEntry(word="dog", language="en", definitions=["canine"]),
            DictionaryEntry(word="hound", language="en", definitions=["hunting dog"]),
            DictionaryEntry(word="pooch", language="en", definitions=["dog"]),
        ]
    }
    mock_response = {
        "candidates": [
            {"rank": 1, "keep": True, "is_cognate": True, "confidence": "high", "notes": None},
            {"rank": 2, "keep": False, "is_cognate": False, "confidence": "low", "notes": "less common"},
            {"rank": 3, "keep": True, "is_cognate": False, "confidence": "medium", "notes": None},
        ]
    }
    with patch.object(
        service, "_chat_with_retry", new_callable=AsyncMock, return_value=mock_response
    ):
        result = await service.filter_and_rank(
            candidates, source_entry, target_entries
        )
    assert len(result) == 2  # keep=True only
    assert result[0].word == "dog"
    assert result[0].confidence == "high"
    assert result[1].word == "pooch"
    assert result[1].confidence == "medium"


async def test_config_env_vars():
    """OllamaLLMService accepts model and temperature overrides (config-driven)."""
    service = OllamaLLMService(model="custom-model", temperature=0.5)
    assert service.model == "custom-model"
    assert service.temperature == 0.5


async def test_config_defaults():
    """Default config values are set."""
    from backend import config
    assert config.LLM_MODEL
    assert 0 <= config.LLM_TEMPERATURE <= 1
