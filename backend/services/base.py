"""LLM service abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.models import (
    LLMTranslationResult,
    RankedTranslation,
    TranslationCandidate,
    TranslationContext,
)
from backend.providers.base import DictionaryEntry


class LLMUnavailableError(Exception):
    """Raised when Ollama is unreachable or times out."""


class LLMService(ABC):
    """Abstract interface for LLM-backed translation and ranking."""

    @abstractmethod
    async def generate_translations(
        self,
        word: str,
        source_lang: str,
        target_langs: list[str],
        context: TranslationContext,
    ) -> LLMTranslationResult:
        """Generate translation candidates via LLM."""
        ...

    @abstractmethod
    async def filter_and_rank(
        self,
        candidates: list[TranslationCandidate],
        source_entry: DictionaryEntry,
        target_entries: dict[str, list[DictionaryEntry]],
    ) -> list[RankedTranslation]:
        """Filter and rank candidates via LLM."""
        ...
