#!/usr/bin/env python3
"""
Data ingestion CLI for Woordhaar.
Downloads (or reads local) dictionary sources and populates woordhaar.db.

Usage:
  python ingest.py [--data-dir DIR] [--db PATH] [--skip-download]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
from pathlib import Path
from typing import Iterator

# Default URLs
KAIKKI_DA = "https://kaikki.org/dictionary/Danish/kaikki.org-dictionary-Danish.jsonl"
KAIKKI_NL = "https://kaikki.org/dictionary/downloads/nl/nl-extract.jsonl.gz"
KAIKKI_EN = "https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl"
COR_SEM_TSV = "https://ordregister.dk/files/cor.sem.1.0.tsv"


def create_schema(conn: sqlite3.Connection) -> None:
    """Create SQLite schema with tables and indexes."""
    conn.executescript("""
        DROP TABLE IF EXISTS translations;
        DROP TABLE IF EXISTS da_entries;
        DROP TABLE IF EXISTS nl_entries;
        DROP TABLE IF EXISTS en_entries;

        CREATE TABLE da_entries (
            word TEXT NOT NULL,
            pos TEXT,
            definition TEXT NOT NULL,
            examples TEXT,
            etymology TEXT,
            synonyms TEXT,
            source TEXT NOT NULL,
            raw_json TEXT
        );
        CREATE INDEX idx_da_word ON da_entries(word);
        CREATE INDEX idx_da_word_pos ON da_entries(word, pos);

        CREATE TABLE nl_entries (
            word TEXT NOT NULL,
            pos TEXT,
            definition TEXT NOT NULL,
            examples TEXT,
            etymology TEXT,
            synonyms TEXT,
            source TEXT NOT NULL,
            raw_json TEXT
        );
        CREATE INDEX idx_nl_word ON nl_entries(word);
        CREATE INDEX idx_nl_word_pos ON nl_entries(word, pos);

        CREATE TABLE en_entries (
            word TEXT NOT NULL,
            pos TEXT,
            definition TEXT NOT NULL,
            examples TEXT,
            etymology TEXT,
            synonyms TEXT,
            source TEXT NOT NULL,
            raw_json TEXT
        );
        CREATE INDEX idx_en_word ON en_entries(word);
        CREATE INDEX idx_en_word_pos ON en_entries(word, pos);

        CREATE TABLE translations (
            word TEXT NOT NULL,
            source_lang TEXT NOT NULL,
            target_word TEXT NOT NULL,
            target_lang TEXT NOT NULL,
            sense TEXT,
            source TEXT NOT NULL
        );
        CREATE INDEX idx_trans_source ON translations(word, source_lang);
        CREATE INDEX idx_trans_target ON translations(target_word, target_lang);
    """)


def _serialize_list(items: list) -> str:
    """Store list as newline-separated string."""
    if not items:
        return ""
    return "\n".join(str(x) for x in items)


def _iter_jsonl(path: Path, gzipped: bool = False) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file."""
    opener = gzip.open if gzipped else open
    mode = "rt" if gzipped else "r"
    with opener(path, mode, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _parse_kaikki_entry(
    entry: dict,
    lang: str,
    source: str,
) -> Iterator[tuple[str, str, str, str, str | None, str, str | None, dict]]:
    """Yield (word, pos, definition, examples, etymology, synonyms, raw_json) tuples."""
    word = entry.get("word", "").strip()
    if not word:
        return
    pos = entry.get("pos") or ""
    ety = entry.get("etymology_text") or entry.get("etymology_texts")
    if isinstance(ety, list):
        etymology = " | ".join(ety) if ety else None
    else:
        etymology = ety
    raw = json.dumps(entry, ensure_ascii=False)

    for sense in entry.get("senses", []):
        glosses = sense.get("glosses") or []
        if not glosses:
            continue
        definition = " | ".join(g for g in glosses if isinstance(g, str))
        if not definition:
            continue
        examples_raw = sense.get("examples", [])
        examples = []
        for ex in examples_raw:
            if isinstance(ex, dict) and "text" in ex:
                examples.append(ex["text"])
            elif isinstance(ex, str):
                examples.append(ex)
        synonyms_raw = sense.get("synonyms", []) or sense.get("antonyms", [])
        synonyms = []
        for s in synonyms_raw:
            if isinstance(s, dict) and "word" in s:
                synonyms.append(s["word"])
            elif isinstance(s, str):
                synonyms.append(s)
        yield (
            word,
            pos,
            definition,
            _serialize_list(examples),
            etymology,
            _serialize_list(synonyms),
            source,
            raw,
        )


def ingest_kaikki_da(conn: sqlite3.Connection, path: Path) -> int:
    """Ingest Danish entries from kaikki enwiktionary."""
    count = 0
    rows = []
    for entry in _iter_jsonl(path):
        for row in _parse_kaikki_entry(entry, "da", "kaikki"):
            rows.append(row)
            count += 1
            if len(rows) >= 5000:
                conn.executemany(
                    "INSERT INTO da_entries (word, pos, definition, examples, etymology, synonyms, source, raw_json) VALUES (?,?,?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
                rows = []
    if rows:
        conn.executemany(
            "INSERT INTO da_entries (word, pos, definition, examples, etymology, synonyms, source, raw_json) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    return count


def ingest_kaikki_nl(conn: sqlite3.Connection, path: Path) -> int:
    """Ingest Dutch entries from nlwiktionary (Dutch-language definitions)."""
    count = 0
    rows = []
    for entry in _iter_jsonl(path, gzipped=path.suffix == ".gz"):
        if entry.get("lang_code") != "nl":
            continue
        for row in _parse_kaikki_entry(entry, "nl", "kaikki_nl"):
            rows.append(row)
            count += 1
            if len(rows) >= 5000:
                conn.executemany(
                    "INSERT INTO nl_entries (word, pos, definition, examples, etymology, synonyms, source, raw_json) VALUES (?,?,?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
                rows = []
    if rows:
        conn.executemany(
            "INSERT INTO nl_entries (word, pos, definition, examples, etymology, synonyms, source, raw_json) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    return count


def ingest_kaikki_en(conn: sqlite3.Connection, path: Path) -> int:
    """Ingest English entries from kaikki enwiktionary."""
    count = 0
    rows = []
    for entry in _iter_jsonl(path):
        if entry.get("lang_code") != "en":
            continue
        for row in _parse_kaikki_entry(entry, "en", "kaikki"):
            rows.append(row)
            count += 1
            if len(rows) >= 5000:
                conn.executemany(
                    "INSERT INTO en_entries (word, pos, definition, examples, etymology, synonyms, source, raw_json) VALUES (?,?,?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
                rows = []
    if rows:
        conn.executemany(
            "INSERT INTO en_entries (word, pos, definition, examples, etymology, synonyms, source, raw_json) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    return count


def ingest_translations_from_en(
    conn: sqlite3.Connection,
    path: Path,
    target_langs: tuple[str, ...] = ("da", "nl"),
) -> int:
    """Extract translation pairs from English dictionary (en↔da, en↔nl)."""
    count = 0
    rows = []
    for entry in _iter_jsonl(path):
        word = entry.get("word", "").strip()
        if not word or entry.get("lang_code") != "en":
            continue
        for sense in entry.get("senses", []):
            sense_desc = sense.get("sense") or " | ".join(sense.get("glosses", []) or [])[:200]
            for t in sense.get("translations", []):
                lc = (t.get("lang_code") or t.get("code") or "").lower()
                if lc not in target_langs:
                    continue
                tw = (t.get("word") or "").strip()
                if not tw:
                    continue
                # Forward: en -> da/nl
                rows.append((word, "en", tw, lc, sense_desc, "kaikki"))
                # Reverse: da/nl -> en
                rows.append((tw, lc, word, "en", sense_desc, "kaikki"))
                count += 2
                if len(rows) >= 10000:
                    conn.executemany(
                        "INSERT INTO translations (word, source_lang, target_word, target_lang, sense, source) VALUES (?,?,?,?,?,?)",
                        rows,
                    )
                    conn.commit()
                    rows = []
    if rows:
        conn.executemany(
            "INSERT INTO translations (word, source_lang, target_word, target_lang, sense, source) VALUES (?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    return count


def ingest_corsem(conn: sqlite3.Connection, path: Path) -> int:
    """Ingest COR.SEM Danish senses.
    Uses overbegreb-tekst (col 11) as definition, DDO-opslagsord (col 6) as word.
    See https://ordregister.dk/files/COR.SEM_1.0_specifikation.html
    """
    count = 0
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue
            word = (parts[5] or "").strip()  # DDO-opslagsord
            if not word:
                continue
            pos = (parts[6] or "").strip()   # DDO-ordklasse
            overbegreb = (parts[10] or "").strip()  # overbegreb-tekst (hypernym)
            relaterede = (parts[12] or "").strip() if len(parts) > 12 else ""
            synonym = (parts[13] or "").strip() if len(parts) > 13 else ""
            definition = overbegreb or relaterede or synonym or "."
            synonyms = " | ".join(filter(None, [synonym.replace("|", " "), relaterede.replace("|", " ")]))
            rows.append((word, pos, definition, "", "", synonyms, "corsem", ""))
            count += 1
            if len(rows) >= 5000:
                conn.executemany(
                    "INSERT INTO da_entries (word, pos, definition, examples, etymology, synonyms, source, raw_json) VALUES (?,?,?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
                rows = []
    if rows:
        conn.executemany(
            "INSERT INTO da_entries (word, pos, definition, examples, etymology, synonyms, source, raw_json) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    return count


def download_file(url: str, dest: Path) -> bool:
    """Download URL to dest. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest} ...")
    try:
        import httpx
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=65536):
                    f.write(chunk)
    except ImportError:
        from urllib.request import urlopen
        with urlopen(url, timeout=300) as r:
            with open(dest, "wb") as f:
                f.write(r.read())
    print(f"  Done ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest dictionary data into woordhaar.db")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory for source files")
    parser.add_argument("--db", type=Path, default=Path("woordhaar.db"), help="Output SQLite DB path")
    parser.add_argument("--skip-download", action="store_true", help="Use existing files in data-dir only")
    parser.add_argument("--skip-corsem", action="store_true", help="Skip COR.SEM (requires separate download)")
    args = parser.parse_args()
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download if needed
    if not args.skip_download:
        download_file(KAIKKI_DA, data_dir / "kaikki-dictionary-Danish.jsonl")
        download_file(KAIKKI_NL, data_dir / "nl-extract.jsonl.gz")
        download_file(KAIKKI_EN, data_dir / "kaikki-dictionary-English.jsonl")
        if not args.skip_corsem:
            download_file(COR_SEM_TSV, data_dir / "cor.sem.1.0.tsv")

    conn = sqlite3.connect(args.db)
    create_schema(conn)

    da_path = data_dir / "kaikki-dictionary-Danish.jsonl"
    nl_path = data_dir / "nl-extract.jsonl.gz"
    en_path = data_dir / "kaikki-dictionary-English.jsonl"
    corsem_path = data_dir / "cor.sem.1.0.tsv"

    total = 0
    if da_path.exists():
        n = ingest_kaikki_da(conn, da_path)
        print(f"Ingested {n} Danish entries from kaikki")
        total += n
    else:
        print(f"Missing {da_path} (run without --skip-download)")

    if corsem_path.exists() and not args.skip_corsem:
        n = ingest_corsem(conn, corsem_path)
        print(f"Ingested {n} Danish entries from COR.SEM")
        total += n

    if nl_path.exists():
        n = ingest_kaikki_nl(conn, nl_path)
        print(f"Ingested {n} Dutch entries from nlwiktionary")
        total += n
    else:
        print(f"Missing {nl_path} (run without --skip-download)")

    if en_path.exists():
        n = ingest_kaikki_en(conn, en_path)
        print(f"Ingested {n} English entries from kaikki")
        total += n
        n2 = ingest_translations_from_en(conn, en_path)
        print(f"Extracted {n2} translation pairs from English dictionary")
    else:
        print(f"Missing {en_path} (run without --skip-download)")

    conn.close()
    print(f"\nTotal entries ingested: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
