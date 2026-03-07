"""Tests for TranslationPipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.models import TranslationResult
from backend.pipeline import TranslationPipeline
from backend.providers.base import DictionaryEntry
from backend.services.base import LLMUnavailableError


pytestmark = pytest.mark.asyncio

DB_PATH = Path(__file__).resolve().parent.parent.parent / "woordhaar.db"


async def _ollama_reachable() -> bool:
    """Check if Ollama /api/chat is available."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://localhost:11434/api/tags", timeout=2.0)
            return r.status_code == 200 and bool((r.json() or {}).get("models"))
    except Exception:
        return False


@pytest.fixture
def pipeline():
    """Pipeline with default provider (uses woordhaar.db)."""
    return TranslationPipeline(db_path=DB_PATH)


async def test_translate_hund_returns_dog(pipeline):
    """translate('hund', 'da') returns TranslationResult with 'dog' in EN translations."""
    result = await pipeline.translate("hund", "da")
    assert isinstance(result, TranslationResult)
    assert result.input_word == "hund"
    assert result.input_language == "da"
    assert result.lemmas
    assert result.senses
    en_trans = result.senses[0].translations.get("en", [])
    words = [t.word for t in en_trans]
    assert "dog" in words or any(
        w in str(words) for w in ["dog", "hound", "canine", "pooch"]
    ), f"Expected 'dog' or similar in EN translations, got: {words}"


async def test_translate_house_returns_hus_and_huis(pipeline):
    """translate('house', 'en') returns NL 'huis' and DA 'hus'."""
    if not await _ollama_reachable():
        pytest.skip("Ollama not available")
    result = await pipeline.translate("house", "en")
    assert isinstance(result, TranslationResult)
    da_words = [t.word for t in result.senses[0].translations.get("da", [])]
    nl_words = [t.word for t in result.senses[0].translations.get("nl", [])]
    assert "hus" in da_words, f"Expected 'hus' in DA translations, got: {da_words}"
    assert "huis" in nl_words, f"Expected 'huis' in NL translations, got: {nl_words}"


async def test_translate_fiets_returns_bicycle(pipeline):
    """translate('fiets', 'nl') returns EN 'bicycle' or 'bike'."""
    if not await _ollama_reachable():
        pytest.skip("Ollama not available")
    result = await pipeline.translate("fiets", "nl")
    assert isinstance(result, TranslationResult)
    en_words = [t.word for t in result.senses[0].translations.get("en", [])]
    assert any(
        w in en_words for w in ("bicycle", "bike", "cycle")
    ), f"Expected bicycle/bike in EN translations, got: {en_words}"


async def test_processing_time_under_5s(pipeline):
    """processing_time_ms < 5000 for dictionary words."""
    if not await _ollama_reachable():
        pytest.skip("Ollama not available")
    result = await pipeline.translate("hund", "da")
    assert result.processing_time_ms < 5000


async def test_pipeline_degrades_gracefully_when_llm_down(pipeline):
    """When LLM is unavailable, returns dictionary-only with confidence='medium'."""
    mock_llm = AsyncMock()
    mock_llm.generate_translations.side_effect = LLMUnavailableError("down")
    pipeline_llm_down = TranslationPipeline(
        db_path=DB_PATH,
        llm_service=mock_llm,
    )
    result = await pipeline_llm_down.translate("hund", "da")
    assert isinstance(result, TranslationResult)
    assert result.senses
    for sense in result.senses:
        for lang, trans in sense.translations.items():
            for t in trans:
                assert t.confidence == "medium"
                assert t.word


async def test_lemmatization_hunde_finds_hund(pipeline):
    """Lemmatization: translate('hunde', 'da') finds results (lemma 'hund')."""
    result = await pipeline.translate("hunde", "da")
    assert isinstance(result, TranslationResult)
    assert "hund" in result.lemmas
    assert result.senses, "Expected senses for lemmatized 'hunde' -> 'hund'"


async def test_translate_empty_word(pipeline):
    """translate('', 'da') returns empty result without error."""
    result = await pipeline.translate("", "da")
    assert result.input_word == ""
    assert result.lemmas == []
    assert result.senses == []
