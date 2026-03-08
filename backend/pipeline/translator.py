"""Translation pipeline orchestrator."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Literal

from loguru import logger

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
        log_ctx = logger.bind(word=word, lang=lang)

        if not word or not word.strip():
            log_ctx.warning("Empty word provided")
            return TranslationResult(
                input_word=word,
                input_language=lang,
                lemmas=[],
                senses=[],
                processing_time_ms=int((time.perf_counter() - t0) * 1000),
            )

        word = word.strip()
        target_langs = TARGET_LANGS.get(lang, ["en", "nl"])
        log_ctx.info(f"Starting translation pipeline, target languages: {target_langs}")

        # 1. Lemmatize
        lemmas = await lemmatize(word, lang)
        log_ctx.debug(f"Lemmatization: {lemmas}")

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
        
        # Log dictionary lookup results
        for tl in target_langs:
            count = len(known_by_lang.get(tl, []))
            log_ctx.debug(f"Dictionary translations {lang}→{tl}: {count} candidates")
        
        log_ctx.info(f"Source entries found: {len(source_entries)}")

        if not source_entries:
            log_ctx.warning("No source entries found in dictionary")
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
                llm_t0 = time.perf_counter()
                try:
                    llm_result = await self.llm_service.generate_translations(
                        entry.word, lang, target_langs, ctx
                    )
                    llm_time = int((time.perf_counter() - llm_t0) * 1000)
                    candidates_by_lang = {}
                    for c in llm_result.translations:
                        candidates_by_lang.setdefault(c.language, []).append(c.word)
                    for tl in target_langs:
                        count = len(candidates_by_lang.get(tl, []))
                        log_ctx.debug(f"LLM generated {count} candidates for {lang}→{tl} (took {llm_time}ms)")
                except LLMUnavailableError as e:
                    llm_time = int((time.perf_counter() - llm_t0) * 1000)
                    log_ctx.warning(
                        f"LLM unavailable for translation generation (took {llm_time}ms), "
                        f"falling back to dictionary-only mode",
                        exc_info=True
                    )
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
                        if not cands:
                            log_ctx.warning(f"No dictionary translations found for {lang}→{tl}")
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
                log_ctx.debug(f"Total candidates to lookup: {len(candidates)}")

                # 5. Parallel: look up all candidates in target dicts
                dict_lookup_t0 = time.perf_counter()
                async def lookup_candidate(c: TranslationCandidate) -> None:
                    entries = await self.provider.lookup(c.word, c.language)
                    defn = entries[0].definitions[0] if entries and entries[0].definitions else None
                    c.definition = defn
                    c.unverified = defn is None
                    if not defn:
                        log_ctx.debug(f"Candidate '{c.word}' ({c.language}) not found in dictionary")

                await asyncio.gather(*[lookup_candidate(c) for c in candidates])
                dict_lookup_time = int((time.perf_counter() - dict_lookup_t0) * 1000)
                verified_count = sum(1 for c in candidates if not c.unverified)
                log_ctx.debug(f"Dictionary lookups completed: {verified_count}/{len(candidates)} verified (took {dict_lookup_time}ms)")

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
                rank_t0 = time.perf_counter()
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
                    rank_time = int((time.perf_counter() - rank_t0) * 1000)
                    kept_count = len(ranked)
                    log_ctx.debug(f"LLM ranking: {kept_count}/{len(candidates)} candidates kept (took {rank_time}ms)")
                except LLMUnavailableError as e:
                    rank_time = int((time.perf_counter() - rank_t0) * 1000)
                    log_ctx.warning(
                        f"LLM unavailable for ranking (took {rank_time}ms), using all candidates",
                        exc_info=True
                    )
                    ranked = [
                        RankedTranslation(
                            word=c.word, language=c.language, confidence="medium",
                            definition=c.definition, is_cognate=False, notes=None,
                        )
                        for c in candidates[:10]
                    ]

                trans_by_lang = {tl: [r for r in ranked if r.language == tl] for tl in target_langs}
                
                # Log missing translations per target language
                for tl in target_langs:
                    count = len(trans_by_lang.get(tl, []))
                    if count == 0:
                        log_ctx.warning(f"No translations found for target language {tl} (sense: {defn[:50]}...)")
                    else:
                        log_ctx.debug(f"Final translations {lang}→{tl}: {count} candidates")
                
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
        total_time = int((time.perf_counter() - t0) * 1000)
        
        log_ctx.info(
            f"Pipeline completed: {len(senses)} senses, {len(cognate_cluster)} cognates, "
            f"total time: {total_time}ms"
        )

        return TranslationResult(
            input_word=word,
            input_language=lang,
            lemmas=lemmas,
            senses=senses,
            etymology=etymology,
            cognate_cluster=cognate_cluster,
            processing_time_ms=total_time,
        )
