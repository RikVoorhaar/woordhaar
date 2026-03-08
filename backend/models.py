"""Pydantic models for LLM input/output and API contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class TranslationContext(BaseModel):
    """Input context for generate_translations."""

    word: str
    pos: str | None = None
    definitions: list[str] = []
    etymology: str | None = None
    known_candidates: dict[str, list[str]] = {}  # target_lang -> [word, ...]


class LLMTranslationCandidate(BaseModel):
    """Single candidate from LLM translation generation."""

    word: str
    language: str
    semantic_precision: Literal["exact", "near", "broader", "narrower"]


class LLMTranslationResult(BaseModel):
    """Output of generate_translations (LLM Call 1)."""

    translations: list[LLMTranslationCandidate] = []


class TranslationCandidate(BaseModel):
    """Input to filter_and_rank."""

    word: str
    language: str
    definition: str | None = None
    unverified: bool = False
    semantic_precision: Literal["exact", "near", "broader", "narrower"] | None = None


class RankedTranslation(BaseModel):
    """Output of filter_and_rank; part of API contract."""

    word: str
    language: str
    confidence: Literal["high", "medium", "low"]
    definition: str | None = None
    is_cognate: bool = False
    notes: str | None = None


class LLMRankingOutput(BaseModel):
    """Single candidate from LLM ranking response."""

    rank: int
    keep: bool
    is_cognate: bool
    confidence: Literal["high", "medium", "low"]
    notes: str | None = None


class LLMRankingResponse(BaseModel):
    """Full LLM ranking response (matches by index to input candidates)."""

    candidates: list[LLMRankingOutput] = []


class ErrorResponse(BaseModel):
    """API error response."""

    detail: str
    error_code: str


class TranslationSense(BaseModel):
    """One sense (definition) with its ranked translations per target language."""

    source_definition: str
    translations: dict[str, list[RankedTranslation]]  # lang -> ranked list


class TranslationResult(BaseModel):
    """API response for /api/translate (frontend contract)."""

    input_word: str
    input_language: str
    lemmas: list[str]
    senses: list[TranslationSense]
    etymology: str | None = None
    cognate_cluster: list[str] = []
    processing_time_ms: int

