"""Translation pipeline orchestrator."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Literal

from backend.models import (
    RankedTranslation,
    TranslationCandidate,
    TranslationContext,
    TranslationResult,
    TranslationSense,
)
from backend.providers.base import DictionaryEntry, DictionaryProvider
from backend.providers.lemmatizer import lemmatize
from backend.providers.sqlite_provider import SqliteDictionaryProvider
from backend.services.base import LLMService, LLMUnavailableError
from backend.services.ollama_service import OllamaLLMService

Lang = Literal["da", "nl", "en"]

TARGET_LANGS: dict[Lang, list[Lang]] = {
    "da": ["en", "nl"],
    "nl": ["da", "en"],
    "en": ["da", "nl"],
}


def _merge_source_entries(entries: list[DictionaryEntry]) -> list[DictionaryEntry]:
    """Merge and deduplicate source entries by (word, pos, first_def)."""
    seen: set[tuple[str, str | None, str]] = set()
    merged: list[DictionaryEntry] = []
    for e in entries:
        first_def = e.definitions[0] if e.definitions else ""
        key = (e.word, e.pos, first_def)
        if key not in seen:
            seen.add(key)
            merged.append(e)
    return merged


class TranslationPipeline:
    """Orchestrates lemmatizer → dictionary lookups → LLM calls → TranslationResult."""

    def __init__(
        self,
        provider: DictionaryProvider | None = None,
        llm_service: LLMService | None = None,
        db_path: str | Path = "woordhaar.db",
    ) -> None:
        self.provider = provider or SqliteDictionaryProvider(db_path)
        self.llm_service = llm_service or OllamaLLMService()

    async def translate(self, word: str, lang: Lang) -> TranslationResult:
        """Run the full pipeline: lemmatize → lookup → LLM generate → lookup → LLM rank."""
        t0 = time.perf_counter()

        if not word or not word.strip():
            return TranslationResult(
                input_word=word,
                input_language=lang,
                lemmas=[],
                senses=[],
                processing_time_ms=int((time.perf_counter() - t0) * 1000),
            )

        word = word.strip()
        target_langs = TARGET_LANGS.get(lang, ["en", "nl"])

        # 1. Lemmatize
        lemmas = await lemmatize(word, lang)

        # 2. Parallel: source dict + bilingual lookups for each lemma
        async def lookup_lemma(lm: str) -> tuple[list[DictionaryEntry], dict[str, list[str]]]:
            tasks = [self.provider.lookup(lm, lang)] + [
                self.provider.lookup_translations(lm, lang, tl) for tl in target_langs
            ]
            results = await asyncio.gather(*tasks)
            source_entries = results[0]
            known: dict[str, list[str]] = {}
            for i, tl in enumerate(target_langs):
                cands = results[1 + i] if 1 + i < len(results) else []
                if cands:
                    known[tl] = cands
            return source_entries, known

        lemma_results = await asyncio.gather(*[lookup_lemma(lm) for lm in lemmas])

        # Merge source entries and known candidates
        all_entries: list[DictionaryEntry] = []
        known_by_lang: dict[str, list[str]] = {"en": [], "nl": []}
        for entries, known in lemma_results:
            all_entries.extend(entries)
            for tl in target_langs:
                known_by_lang.setdefault(tl, [])
                for c in known.get(tl, []):
                    if c not in known_by_lang[tl]:
                        known_by_lang[tl].append(c)

        source_entries = _merge_source_entries(all_entries)

        if not source_entries:
            return TranslationResult(
                input_word=word,
                input_language=lang,
                lemmas=lemmas,
                senses=[],
                processing_time_ms=int((time.perf_counter() - t0) * 1000),
            )

        etymology = source_entries[0].etymology
        senses: list[TranslationSense] = []
        cognate_cluster: list[str] = []

        for entry in source_entries:
            for defn in entry.definitions[:3]:  # cap at 3 senses per entry
                ctx = TranslationContext(
                    word=entry.word,
                    pos=entry.pos,
                    definitions=[defn],
                    etymology=entry.etymology,
                    known_candidates={k: known_by_lang.get(k, []) for k in target_langs},
                )
                # LLM Call 1: generate translations
                try:
                    llm_result = await self.llm_service.generate_translations(
                        entry.word, lang, target_langs, ctx
                    )
                except LLMUnavailableError:
                    # Fallback: dictionary-only, no LLM ranking
                    trans_by_lang: dict[str, list[RankedTranslation]] = {}
                    for tl in target_langs:
                        cands = known_by_lang.get(tl, [])
                        trans_by_lang[tl] = [
                            RankedTranslation(
                                word=c, language=tl, confidence="medium",
                                definition=None, is_cognate=False, notes=None
                            )
                            for c in cands[:5]
                        ]
                    senses.append(
                        TranslationSense(
                            source_definition=defn,
                            translations=trans_by_lang,
                        )
                    )
                    continue

                # Collect candidates (LLM + known)
                cand_map: dict[tuple[str, str], TranslationCandidate] = {}
                for c in llm_result.translations:
                    key = (c.word, c.language)
                    if key not in cand_map:
                        cand_map[key] = TranslationCandidate(
                            word=c.word,
                            language=c.language,
                            definition=None,
                            unverified=True,
                            semantic_precision=c.semantic_precision,
                        )
                for tl in target_langs:
                    for c in known_by_lang.get(tl, []):
                        key = (c, tl)
                        if key not in cand_map:
                            cand_map[key] = TranslationCandidate(
                                word=c, language=tl,
                                definition=None, unverified=True,
                                semantic_precision=None,
                            )
                candidates = list(cand_map.values())

                # 5. Parallel: look up all candidates in target dicts
                async def lookup_candidate(c: TranslationCandidate) -> None:
                    entries = await self.provider.lookup(c.word, c.language)
                    defn = entries[0].definitions[0] if entries and entries[0].definitions else None
                    c.definition = defn
                    c.unverified = defn is None

                await asyncio.gather(*[lookup_candidate(c) for c in candidates])

                target_entries: dict[str, list[DictionaryEntry]] = {}
                for c in candidates:
                    target_entries.setdefault(c.language, [])
                    if not any(e.word == c.word for e in target_entries[c.language]):
                        target_entries[c.language].append(
                            DictionaryEntry(
                                word=c.word,
                                language=c.language,
                                definitions=[c.definition] if c.definition else [],
                            )
                        )

                # LLM Call 2: filter and rank
                try:
                    ranked = await self.llm_service.filter_and_rank(
                        candidates,
                        DictionaryEntry(
                            word=entry.word,
                            language=lang,
                            pos=entry.pos,
                            definitions=[defn],
                        ),
                        target_entries,
                    )
                except LLMUnavailableError:
                    ranked = [
                        RankedTranslation(
                            word=c.word, language=c.language, confidence="medium",
                            definition=c.definition, is_cognate=False, notes=None,
                        )
                        for c in candidates[:10]
                    ]

                trans_by_lang = {tl: [r for r in ranked if r.language == tl] for tl in target_langs}
                senses.append(
                    TranslationSense(
                        source_definition=defn,
                        translations=trans_by_lang,
                    )
                )
                for r in ranked:
                    if r.is_cognate:
                        cognate_cluster.append(f"{r.word} ({r.language.upper()})")

        cognate_cluster = list(dict.fromkeys(cognate_cluster))

        return TranslationResult(
            input_word=word,
            input_language=lang,
            lemmas=lemmas,
            senses=senses,
            etymology=etymology,
            cognate_cluster=cognate_cluster,
            processing_time_ms=int((time.perf_counter() - t0) * 1000),
        )
