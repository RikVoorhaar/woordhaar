"""Prompt template builders for LLM calls."""

from __future__ import annotations

import json

from backend.models import TranslationCandidate, TranslationContext
from backend.providers.base import DictionaryEntry


TRANSLATION_SYSTEM = """You are a multilingual lexicographer. Given a source word and its definitions, produce translation candidates for the target languages.

Output strict JSON only, no markdown:
{
  "translations": [
    {"word": "<translation>", "language": "<lang>", "semantic_precision": "exact|near|broader|narrower"},
    ...
  ]
}

Rules:
- 3–5 candidates per target language
- Do not invent definitions; output only translation words
- semantic_precision: "exact" (best match), "near" (close), "broader" (more general), "narrower" (more specific)
"""

RANKING_SYSTEM = """You are a lexicographer evaluating translation candidates. Compare each candidate's definition to the source definition. Output strict JSON only, no markdown:

{
  "candidates": [
    {"rank": 1, "keep": true, "is_cognate": false, "confidence": "high|medium|low", "notes": null},
    ...
  ]
}

Rules:
- rank: 1 = best, ascending
- keep: false to reject (e.g. false friend, wrong sense)
- is_cognate: true if etymologically related
- confidence: high/medium/low based on semantic alignment
- notes: brief explanation for rejections; identify false friends explicitly
"""


def build_translation_prompt(
    context: TranslationContext,
    source_lang: str,
    target_langs: list[str],
) -> list[dict]:
    """Build messages for translation generation (LLM Call 1)."""
    definitions = context.definitions[:2]  # truncate to 2 senses
    payload = {
        "word": context.word,
        "pos": context.pos,
        "definitions": definitions,
        "etymology": context.etymology,
        "known_candidates": context.known_candidates,
    }
    user_content = f"Source language: {source_lang}. Target languages: {target_langs}.\n\nInput:\n{json.dumps(payload, ensure_ascii=False)}"
    return [
        {"role": "system", "content": TRANSLATION_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def build_ranking_prompt(
    candidates: list[TranslationCandidate],
    source_def: str,
    target_entries: dict[str, list[DictionaryEntry]],
) -> list[dict]:
    """Build messages for filter/rank (LLM Call 2)."""
    def _defn(c: TranslationCandidate) -> str:
        if c.definition:
            return c.definition
        for entries in target_entries.values():
            for e in entries:
                if e.word == c.word and e.definitions:
                    return e.definitions[0]
        return "[unverified - not in dictionary]"

    items = [
        {"word": c.word, "language": c.language, "definition": _defn(c)}
        for c in candidates[:15]
    ]
    payload = {
        "source_definition": source_def,
        "candidates": items,
    }
    user_content = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": RANKING_SYSTEM},
        {"role": "user", "content": user_content},
    ]
