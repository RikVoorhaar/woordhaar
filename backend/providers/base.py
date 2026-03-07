"""Dictionary provider interface and models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel


class DictionaryEntry(BaseModel):
    """A single dictionary entry."""

    word: str
    language: Literal["da", "nl", "en"]
    pos: str | None = None
    definitions: list[str]
    examples: list[str] = []
    synonyms: list[str] = []
    etymology: str | None = None
    raw_data: dict = {}


class DictionaryProvider(ABC):
    """Abstract interface for monolingual and bilingual lookups.
    All implementations must be async."""

    @abstractmethod
    async def lookup(self, word: str, lang: str) -> list[DictionaryEntry]:
        """Look up a word. Returns all matching entries."""
        ...

    @abstractmethod
    async def lookup_translations(
        self, word: str, source_lang: str, target_lang: str
    ) -> list[str]:
        """Return known translation candidates from bilingual data.
        May return empty list if not a bilingual source."""
        ...

    @abstractmethod
    async def has_word(self, word: str, lang: str) -> bool:
        """Fast existence check."""
        ...
