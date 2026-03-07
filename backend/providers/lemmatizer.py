"""Lemmatizer wrapper using simplemma."""

from __future__ import annotations

import asyncio
from typing import Literal

import simplemma


async def lemmatize(
    word: str, lang: Literal["da", "nl", "en"]
) -> list[str]:
    """Return 1–3 lemma candidates: original + standard + greedy if different.

    Uses simplemma standard mode and greedy=True for compound decomposition.
    Returns deduplicated list (original, lemma_standard, lemma_greedy) filtered to unique values.
    """
    return await asyncio.to_thread(_lemmatize_sync, word, lang)


def _lemmatize_sync(word: str, lang: str) -> list[str]:
    """Synchronous lemmatization (called via to_thread)."""
    candidates: list[str] = [word]
    lemma_std = simplemma.lemmatize(word, lang=lang)
    if lemma_std and lemma_std != word:
        candidates.append(lemma_std)
    lemma_greedy = simplemma.lemmatize(word, lang=lang, greedy=True)
    if lemma_greedy and lemma_greedy not in candidates:
        candidates.append(lemma_greedy)
    if lemma_std == word and word != word.lower():
        lemma_lower = simplemma.lemmatize(word.lower(), lang=lang)
        if lemma_lower and lemma_lower not in candidates:
            candidates.append(lemma_lower)
    seen: set[str] = set()
    result: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            result.append(c)
    return result[:3]
