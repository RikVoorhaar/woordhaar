"""Tests for DictionaryProvider and lemmatizer."""

from __future__ import annotations

import time

import pytest

from backend.providers import (
    SqliteDictionaryProvider,
    lemmatize,
)


pytestmark = pytest.mark.asyncio


async def test_lemmatize_hunde_da():
    result = await lemmatize("hunde", "da")
    assert "hunde" in result
    assert "hund" in result


async def test_lemmatize_huizen_nl():
    result = await lemmatize("huizen", "nl")
    assert "huizen" in result
    # simplemma may or may not reduce huizen->huis depending on version
    assert 1 <= len(result) <= 3


async def test_lemmatize_running_en():
    result = await lemmatize("running", "en")
    assert "running" in result
    assert "run" in result


async def test_provider_lookup_hund():
    provider = SqliteDictionaryProvider()
    entries = await provider.lookup("hund", "da")
    assert len(entries) >= 1
    assert entries[0].definitions
    assert entries[0].word == "hund"


async def test_provider_lookup_hond():
    provider = SqliteDictionaryProvider()
    entries = await provider.lookup("hond", "nl")
    assert len(entries) >= 1
    assert entries[0].definitions
    assert "hond" in entries[0].word or entries[0].word == "hond"


async def test_provider_lookup_nonexistent():
    provider = SqliteDictionaryProvider()
    entries = await provider.lookup("nonexistentword123", "en")
    assert entries == []


async def test_provider_has_word():
    provider = SqliteDictionaryProvider()
    assert await provider.has_word("dog", "en") is True


async def test_provider_lookup_translations():
    provider = SqliteDictionaryProvider()
    candidates = await provider.lookup_translations("hund", "da", "en")
    # Data may have "dog" or "pooch" (both valid translations of Danish "hund")
    assert len(candidates) >= 1
    assert any(t in ("dog", "pooch") for t in candidates)


async def test_provider_lookup_latency():
    provider = SqliteDictionaryProvider()
    t0 = time.perf_counter()
    await provider.lookup("hund", "da")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50
