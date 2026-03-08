"""SQLite-backed implementation of DictionaryProvider."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import aiosqlite
from loguru import logger

from .base import DictionaryEntry, DictionaryProvider

_TABLE_MAP = {"da": "da_entries", "nl": "nl_entries", "en": "en_entries"}


def _parse_list(s: str | None) -> list[str]:
    """Parse newline- or pipe-separated list from DB."""
    if not s or not s.strip():
        return []
    parts = re.split(r"[\n|]+", s)
    return [p.strip() for p in parts if p.strip()]


class SqliteDictionaryProvider(DictionaryProvider):
    """Dictionary provider backed by SQLite (woordhaar.db)."""

    def __init__(self, db_path: str | Path = "woordhaar.db") -> None:
        self.db_path = Path(db_path)

    async def lookup(self, word: str, lang: str) -> list[DictionaryEntry]:
        """Look up a word in the monolingual table."""
        log_ctx = logger.bind(word=word, lang=lang)
        table = _TABLE_MAP.get(lang)
        if not table:
            log_ctx.warning(f"Invalid language code: {lang}")
            return []
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(
                    f"""SELECT word, pos, definition, examples, etymology, synonyms, raw_json
                    FROM {table} WHERE word = ?""",
                    (word,),
                )
                rows = await cursor.fetchall()
            
            entries = [
                _row_to_entry(row, lang)
                for row in rows
            ]
            
            if not entries:
                log_ctx.debug(f"No dictionary entries found for '{word}' ({lang})")
            else:
                # Count entries with definitions
                entries_with_defs = sum(1 for e in entries if e.definitions)
                log_ctx.debug(
                    f"Dictionary lookup: {len(entries)} entries found, "
                    f"{entries_with_defs} with definitions"
                )
            
            return entries
        except Exception as e:
            log_ctx.error(f"Dictionary lookup failed: {e}", exc_info=True)
            raise

    async def lookup_translations(
        self, word: str, source_lang: str, target_lang: str
    ) -> list[str]:
        """Return translation candidates from the bilingual table."""
        log_ctx = logger.bind(
            word=word,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    """SELECT DISTINCT target_word FROM translations
                    WHERE word = ? AND source_lang = ? AND target_lang = ?""",
                    (word, source_lang, target_lang),
                )
                rows = await cursor.fetchall()
            
            translations = [r[0] for r in rows if r[0]]
            
            if not translations:
                log_ctx.debug(f"No bilingual translations found for {source_lang}→{target_lang}")
            else:
                log_ctx.debug(f"Bilingual lookup: {len(translations)} translations found")
            
            return translations
        except Exception as e:
            log_ctx.error(f"Bilingual lookup failed: {e}", exc_info=True)
            raise

    async def has_word(self, word: str, lang: str) -> bool:
        """Fast existence check."""
        table = _TABLE_MAP.get(lang)
        if not table:
            return False
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                f"SELECT 1 FROM {table} WHERE word = ? LIMIT 1",
                (word,),
            )
            row = await cursor.fetchone()
        return row is not None


def _row_to_entry(row: aiosqlite.Row, lang: str) -> DictionaryEntry:
    raw_json = row["raw_json"]
    raw_data = {}
    if raw_json:
        try:
            import json
            raw_data = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            logger.debug(f"Failed to parse raw_json for word '{row['word']}' ({lang})")
    
    definition_text = row["definition"]
    has_definition = bool(definition_text and definition_text.strip())
    
    return DictionaryEntry(
        word=row["word"],
        language=lang,
        pos=row["pos"] or None,
        definitions=[definition_text] if has_definition else [],
        examples=_parse_list(row["examples"]),
        synonyms=_parse_list(row["synonyms"]),
        etymology=row["etymology"] or None,
        raw_data=raw_data,
    )
