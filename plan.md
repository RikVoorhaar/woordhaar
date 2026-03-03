# Woordhaar MVP — Implementation Plan

A cross-lingual dictionary tool that accepts a word in Danish, Dutch, or English, and returns ranked translations into the other two languages with definitions, examples, and cognate information. All dictionary lookups are backed by local SQLite; translation candidates are generated and validated via a local LLM served by Ollama.

## Architecture Overview

```
┌────────────────────────────────┐
│  SvelteKit Frontend            │
│  3 input fields (DA / NL / EN) │
│  Displays TranslationResult    │
└──────────┬─────────────────────┘
           │ HTTP/JSON (async)
           ▼
┌────────────────────────────────┐
│  FastAPI Backend               │
│  - /api/translate              │
│  - /api/lookup/{lang}/{word}   │
│  - /api/health                 │
│  Pydantic models throughout    │
└──┬─────────┬──────────┬───────┘
   │         │          │
   ▼         ▼          ▼
SQLite    simplemma    Ollama
(dict DB)  (lemma)   (LLM API)
```

## Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Backend framework | FastAPI (async) | Native async, Pydantic integration, OpenAPI docs for free |
| Database | SQLite (via aiosqlite) | Single-file, zero-config, fast reads for dictionary data |
| Validation | Pydantic v2 | Used for request/response models and internal data classes |
| Lemmatizer | **simplemma** | Pure Python, no deps, supports DA (0.92), NL (0.92), EN (0.94) on UD treebanks; >250k words/sec[1] |
| LLM inference | Ollama (local) | REST API at `localhost:11434`, model-agnostic, easy swapping |
| Frontend | SvelteKit | Lightweight, reactive, TypeScript-native |
| Data sources | kaikki.org (EN, NL, bilingual), COR.SEM/EXT + da.wiktionary (DA) | Best available monolingual coverage per language[2][3][1] |

### LLM Model Recommendations

The LLM must handle multilingual text (DA/NL/EN), produce structured JSON, and run on a single RTX 3090 (24 GB VRAM). Three strong candidates:

| Model | Params | RTX 3090 Speed | Strengths | Weaknesses |
|-------|--------|----------------|-----------|------------|
| **Qwen3.5-35B-A3B** (MoE) | 35B (3B active) | ~77–90 t/s Q8[4] | Fastest; excellent multilingual; good structured output | Newer, less battle-tested |
| **Mistral Small 3.1 24B** | 24B (dense) | ~30–40 t/s Q4[5][6] | Best translation quality in European langs[7]; Apache 2.0; solid JSON mode | Slower than MoE; Danish not explicitly listed but works well |
| **Gemma 3 12B** | 12B (dense) | ~50–70 t/s | 140+ language pretraining[8]; good structured JSON output[9]; smallest footprint | Weaker on nuanced translation compared to larger models |

**Recommendation:** Start with **Qwen3.5-35B-A3B** for speed during development; benchmark against **Mistral Small 3.1** for translation quality before shipping. The Ollama abstraction makes swapping trivial (`ollama pull qwen3.5:35b-a3b` vs `ollama pull mistral-small3.1`).

## Core Interfaces

### Dictionary Provider Interface

```python
from abc import ABC, abstractmethod

class DictionaryEntry(BaseModel):
    word: str
    language: Literal["da", "nl", "en"]
    pos: str | None = None
    definitions: list[str]
    examples: list[str] = []
    synonyms: list[str] = []
    etymology: str | None = None
    raw_data: dict = {}  # preserve full source record for future use

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
```

Concrete implementations: `SqliteDictionaryProvider` (wraps aiosqlite), later optionally `DanNetProvider`, `OrdnetScraperProvider`.

### LLM Service Interface

```python
class LLMService(ABC):
    @abstractmethod
    async def generate_translations(
        self, word: str, source_lang: str, target_langs: list[str],
        context: TranslationContext
    ) -> LLMTranslationResult:
        ...

    @abstractmethod
    async def filter_and_rank(
        self, candidates: list[TranslationCandidate],
        source_entry: DictionaryEntry,
        target_entries: dict[str, list[DictionaryEntry]]
    ) -> list[RankedTranslation]:
        ...
```

Concrete: `OllamaLLMService` using `httpx.AsyncClient` against `localhost:11434/api/generate`.

### API Response Model (Frontend Contract)

```python
class TranslationSense(BaseModel):
    source_definition: str
    translations: dict[str, list[RankedTranslation]]  # lang -> ranked list

class RankedTranslation(BaseModel):
    word: str
    language: str
    confidence: Literal["high", "medium", "low"]
    definition: str | None = None  # from target monolingual dict
    is_cognate: bool = False
    notes: str | None = None  # e.g. "false friend", "formal register"

class TranslationResult(BaseModel):
    input_word: str
    input_language: str
    lemmas: list[str]
    senses: list[TranslationSense]
    etymology: str | None = None
    cognate_cluster: list[str] = []  # e.g. ["hus (DA)", "huis (NL)", "house (EN)"]
    processing_time_ms: int
```

This is the single contract between backend and frontend. The frontend renders `TranslationResult` as-is.

### API Endpoints

| Method | Path | Description | Request | Response |
|--------|------|-------------|---------|----------|
| POST | `/api/translate` | Main translation pipeline | `{"word": str, "language": "da"\|"nl"\|"en"}` | `TranslationResult` |
| GET | `/api/lookup/{lang}/{word}` | Raw dictionary lookup (debug/testing) | — | `list[DictionaryEntry]` |
| GET | `/api/health` | Healthcheck (DB + Ollama reachable) | — | `{"status": "ok", "ollama": bool, "db_entries": dict}` |

All endpoints are async. `/api/translate` is the only one the frontend calls in normal use.

## Prompt Design Criteria

Full prompt text is **not** included in this document. Instead, prompts must satisfy these criteria:

### Translation Generation Prompt (LLM Call 1)

- **Input format:** Structured JSON containing the source word, POS, all available definitions (in source language), etymology text, and any known translation candidates from bilingual dictionaries.
- **Output format:** JSON conforming to a strict schema (enforced via Ollama's `format` parameter or prompt-level JSON schema). Must include `translations` array with `word`, `language`, `semantic_precision` ("exact", "near", "broader", "narrower") per candidate.
- **Instruction tone:** Concise, directive. The model is a "multilingual lexicographer."
- **Constraint:** Must generate 3–5 candidates per target language per sense. Must not invent definitions — only translation words.
- **Context budget:** Keep total prompt under 800 tokens to ensure fast inference. Source definitions should be truncated to first 2 senses if longer.

### Filtering/Ranking Prompt (LLM Call 2)

- **Input format:** Source definition (in source language) + each candidate word paired with its monolingual definition from the target dictionary. Candidates not found in the dictionary are flagged as `unverified`.
- **Output format:** JSON array of candidates with `rank`, `keep` (bool), `is_cognate` (bool), `confidence`, `notes`.
- **Instruction tone:** Evaluative. The model compares definitions and judges semantic alignment.
- **Constraint:** Must explain rejections briefly (1 sentence in `notes`). Must identify false friends explicitly.
- **Context budget:** Under 1,200 tokens total. If many candidates, batch into groups of 5.

### General Prompt Principles

- All prompts use system + user message separation (Ollama chat API).
- System message defines the role and output schema once; user message provides the per-word data.
- No few-shot examples in MVP (add later if quality is insufficient).
- Temperature = 0.1 for translation generation, 0.0 for ranking.
- Always request `"format": "json"` in the Ollama API call.

## Implementation Steps

***

### Step 1 — Data Ingestion Pipeline

**Goal:** Populate a SQLite database with monolingual and bilingual dictionary data from all sources.

**Deliverables:**
- `ingest.py` — CLI script that downloads (or reads local) data files and populates `woordhaar.db`
- SQLite schema with tables: `da_entries`, `nl_entries`, `en_entries`, `translations` (bilingual pairs)
- Each `*_entries` table has columns: `word TEXT, pos TEXT, definition TEXT, examples TEXT, etymology TEXT, synonyms TEXT, source TEXT, raw_json TEXT`
- `translations` table: `word TEXT, source_lang TEXT, target_word TEXT, target_lang TEXT, sense TEXT, source TEXT`
- Indexes on `(word)` and `(word, pos)` for all tables

**Data sources ingested:**

| Source | Target Table | Records (approx) |
|--------|-------------|-------------------|
| kaikki.org enwiktionary — Danish entries | `da_entries` | ~23k |
| COR.SEM + COR.SEM.EXT | `da_entries` (source=`corsem`) | ~34k senses |
| kaikki.org nlwiktionary — Dutch entries | `nl_entries` | ~200k+ (filtered to `lang_code=nl`)[2] |
| kaikki.org enwiktionary — English entries | `en_entries` | ~500k (filtered to `lang_code=en`) |
| kaikki.org enwiktionary — translation pairs | `translations` | ~1M+ pairs (DA↔EN, NL↔EN extracted from translation tables)[10] |

**Test criteria:**
- [ ] `woordhaar.db` exists and is <8 GB
- [ ] `SELECT count(*) FROM da_entries` ≥ 20,000
- [ ] `SELECT count(*) FROM nl_entries` ≥ 100,000
- [ ] `SELECT count(*) FROM en_entries` ≥ 200,000
- [ ] `SELECT count(*) FROM translations` ≥ 500,000
- [ ] `SELECT definition FROM da_entries WHERE word='hund' LIMIT 1` returns a non-empty Danish or English definition
- [ ] `SELECT definition FROM nl_entries WHERE word='hond' LIMIT 1` returns a Dutch-language definition
- [ ] `SELECT definition FROM en_entries WHERE word='dog' LIMIT 1` returns an English definition
- [ ] Ingestion completes in <30 minutes on a modern machine

***

### Step 2 — Dictionary Provider + Lemmatizer

**Goal:** Implement the `DictionaryProvider` interface backed by SQLite, and integrate simplemma for lemmatization.

**Deliverables:**
- `providers/sqlite_provider.py` — implements `DictionaryProvider` with aiosqlite
- `providers/lemmatizer.py` — wraps simplemma, exposes `async def lemmatize(word: str, lang: str) -> list[str]` that returns 1–3 lemma candidates (original form + simplemma output + greedy variant if different)
- `providers/__init__.py` — re-exports both

**simplemma integration details:**
- Use `simplemma.lemmatize(word, lang=code)` for standard mode and `greedy=True` for compound decomposition[1]
- If simplemma returns the same form, also try lowercase
- Return deduplicated list: `[original, lemma_standard, lemma_greedy]` (filtered to unique values)

**Test criteria:**
- [ ] `lemmatize("hunde", "da")` returns `["hunde", "hund"]`
- [ ] `lemmatize("huizen", "nl")` returns `["huizen", "huis"]`
- [ ] `lemmatize("running", "en")` returns `["running", "run"]`
- [ ] `provider.lookup("hund", "da")` returns ≥1 `DictionaryEntry` with non-empty `definitions`
- [ ] `provider.lookup("hond", "nl")` returns ≥1 entry with a Dutch-language definition
- [ ] `provider.lookup("nonexistentword123", "en")` returns `[]`
- [ ] `provider.has_word("dog", "en")` returns `True`
- [ ] `provider.lookup_translations("hund", "da", "en")` returns a list containing `"dog"`
- [ ] All methods are async and complete in <50ms for single-word lookups

***

### Step 3 — LLM Service (Ollama)

**Goal:** Implement the `LLMService` interface using Ollama's HTTP API, with structured JSON output.

**Deliverables:**
- `services/ollama_service.py` — implements `LLMService` via `httpx.AsyncClient`
- `services/prompts.py` — prompt template builder functions (not raw strings; parameterized functions that construct the messages list)
- `config.py` — model name, Ollama base URL, temperature, timeout settings (all overridable via env vars, prefixed `WOORDHAAR_`)

**Implementation details:**
- Use Ollama's `/api/chat` endpoint with `"format": "json"` and `"stream": false`
- Timeout per call: 15 seconds (generous for RTX 3090 speeds)
- Retry once on timeout; return partial results (dictionary-only) if LLM is unavailable
- Prompt builder functions take typed dataclasses as input and return `list[dict]` (messages)

**Test criteria:**
- [ ] `ollama_service.generate_translations("hund", "da", ["en", "nl"], context)` returns valid `LLMTranslationResult` with ≥1 candidate per target language
- [ ] Output JSON parses into the Pydantic model without validation errors
- [ ] Response time <10s on RTX 3090 with Qwen3.5-35B-A3B or equivalent
- [ ] When Ollama is unreachable, raises `LLMUnavailableError` (not an unhandled exception)
- [ ] `filter_and_rank()` returns candidates sorted by rank, with `keep=False` candidates removed
- [ ] Temperature and model name are configurable via `WOORDHAAR_LLM_MODEL` and `WOORDHAAR_LLM_TEMPERATURE` env vars

***

### Step 4 — Translation Pipeline (Orchestrator)

**Goal:** Wire together lemmatizer → dictionary lookups → LLM calls → validation → ranking into a single async pipeline that powers `/api/translate`.

**Deliverables:**
- `pipeline/translator.py` — `TranslationPipeline` class with method `async def translate(word: str, lang: str) -> TranslationResult`
- Orchestrates the full flow:
  1. Lemmatize input → get 1–3 lemma candidates
  2. **In parallel (asyncio.gather):** look up each lemma in source monolingual dict + bilingual translations table
  3. Merge and deduplicate source entries
  4. **LLM Call 1:** generate additional translation candidates (pass source definitions + bilingual candidates as context)
  5. **In parallel:** look up all EN/NL candidates in their respective monolingual dictionaries
  6. **LLM Call 2:** filter, rank, identify cognates (pass source def + target defs)
  7. Assemble `TranslationResult`

**Parallelism strategy:**
- Steps 2a (source dict) and 2b (bilingual) run concurrently
- Step 5 (target dict lookups for all candidates) runs concurrently per candidate
- Steps 4 and 6 (LLM calls) are sequential (they depend on prior results)
- Total wall-clock target: <5 seconds for common words on RTX 3090

**Test criteria:**
- [ ] `translate("hund", "da")` returns a `TranslationResult` where `senses[0].translations["en"]` contains an entry with `word="dog"`
- [ ] `translate("house", "en")` returns NL translations containing `"huis"` and DA translations containing `"hus"`
- [ ] `translate("fiets", "nl")` returns EN translations containing `"bicycle"` or `"bike"`
- [ ] `processing_time_ms` is <5000 for words present in the dictionary
- [ ] Pipeline degrades gracefully if LLM is down: returns dictionary-only results with empty `confidence` fields
- [ ] Lemmatization handles inflected forms: `translate("hunde", "da")` finds results for "hund"

***

### Step 5 — FastAPI Application

**Goal:** Expose the translation pipeline as a REST API with proper error handling, CORS, and OpenAPI documentation.

**Deliverables:**
- `main.py` — FastAPI app with three endpoints (`/api/translate`, `/api/lookup/{lang}/{word}`, `/api/health`)
- CORS middleware configured for `localhost:5173` (SvelteKit dev server)
- Startup event: open SQLite connection pool, verify Ollama reachability
- Shutdown event: close connections
- Error responses use Pydantic models: `ErrorResponse(detail: str, error_code: str)`

**Test criteria:**
- [ ] `pytest` test suite with ≥10 tests covering all endpoints
- [ ] `POST /api/translate {"word": "hund", "language": "da"}` returns 200 with valid `TranslationResult` JSON
- [ ] `POST /api/translate {"word": "", "language": "da"}` returns 422 (validation error)
- [ ] `POST /api/translate {"word": "hund", "language": "xx"}` returns 422 (invalid language)
- [ ] `GET /api/lookup/da/hund` returns list of `DictionaryEntry` objects
- [ ] `GET /api/health` returns `{"status": "ok", "ollama": true, "db_entries": {"da": N, "nl": N, "en": N}}`
- [ ] OpenAPI docs accessible at `/docs` and correctly describe all models
- [ ] All endpoints respond within 10s under normal conditions
- [ ] CORS headers present for `localhost:5173`

***

### Step 6 — SvelteKit Frontend

**Goal:** A minimal single-page UI that allows entering a word in any of three languages and displays the `TranslationResult`.

**Deliverables:**
- SvelteKit project with a single route (`/`)
- Three text input fields labeled "Dansk", "Nederlands", "English" — pressing Enter in any field triggers a `POST /api/translate` with the appropriate language code
- Loading spinner while waiting for response
- Result display:
  - Word + lemma(s) + etymology (if available)
  - Per-sense block showing source definition and a table of translations per target language (word, confidence badge, definition snippet, cognate marker)
  - Cognate cluster displayed as a highlighted row if present
- Responsive layout (works on mobile)
- No authentication, no routing, no state management library — keep it as simple as possible

**Test criteria:**
- [ ] Typing "hund" in the "Dansk" field and pressing Enter shows English and Dutch translations
- [ ] Typing "house" in the "English" field and pressing Enter shows Danish and Dutch translations
- [ ] Loading state is visible while the API call is in flight
- [ ] An empty input shows a validation hint, not an API error
- [ ] The page loads in <1s and works in Chrome/Firefox
- [ ] API errors display a user-friendly message (e.g., "Translation service temporarily unavailable")

***

### Step 7 — Integration Testing + Polish

**Goal:** End-to-end validation of the full stack, performance tuning, and basic UX polish.

**Deliverables:**
- `tests/test_e2e.py` — end-to-end tests that boot the FastAPI server and make real HTTP calls (using `httpx.AsyncClient` with `ASGITransport`)
- A test word list of 50 common words (mix of DA/NL/EN, including inflected forms, compound words, and known false friends) with expected translation pairs
- Performance benchmark script that measures p50/p95 latency over the test word list
- Any UX fixes identified during testing (e.g., timeout handling, empty-result messaging)

**Test criteria:**
- [ ] ≥80% of test words return at least one correct translation per target language (manually verified against the 50-word list)
- [ ] p50 latency <3s, p95 <6s on RTX 3090 with chosen model
- [ ] No unhandled exceptions in server logs during the full test suite
- [ ] False friends in the test list (e.g., DA "gift" ≠ EN "gift") are flagged in `notes`
- [ ] Compound words (e.g., NL "fietsenstalling") are lemmatized and produce results
- [ ] The system handles unknown words gracefully (returns empty senses, not a 500 error)

## File Structure

```
woordhaar/
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── config.py                # Settings (env-var overridable)
│   ├── models.py                # All Pydantic models
│   ├── pipeline/
│   │   └── translator.py        # TranslationPipeline orchestrator
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py              # ABC: DictionaryProvider
│   │   ├── sqlite_provider.py   # SQLite implementation
│   │   └── lemmatizer.py        # simplemma wrapper
│   ├── services/
│   │   ├── __init__.py
│   │   ├── base.py              # ABC: LLMService
│   │   ├── ollama_service.py    # Ollama implementation
│   │   └── prompts.py           # Prompt builder functions
│   ├── ingest.py                # Data ingestion CLI
│   └── tests/
│       ├── test_provider.py
│       ├── test_llm.py
│       ├── test_pipeline.py
│       ├── test_api.py
│       └── test_e2e.py
├── frontend/
│   ├── src/
│   │   └── routes/
│   │       └── +page.svelte     # Single-page UI
│   ├── package.json
│   └── svelte.config.js
├── data/                        # Downloaded source files (gitignored)
├── woordhaar.db                 # Generated SQLite DB (gitignored)
└── README.md
```

## Dependency Versions

```
# backend/requirements.txt
fastapi>=0.115
uvicorn[standard]>=0.34
aiosqlite>=0.21
httpx>=0.28
pydantic>=2.10
simplemma>=1.1
```

## Step Dependency Graph

```
Step 1 (Data Ingestion)
   │
   ▼
Step 2 (Dict Provider + Lemmatizer)
   │
   ├──────────────────┐
   ▼                  ▼
Step 3 (LLM Service)  (independent)
   │                  │
   ▼                  │
Step 4 (Pipeline) ◄───┘
   │
   ▼
Step 5 (FastAPI)
   │
   ├──────────────────┐
   ▼                  ▼
Step 6 (Frontend)   Step 7 (E2E Tests)
```

Steps 2 and 3 can be developed in parallel since they have no mutual dependency. Step 6 (frontend) and Step 7 (E2E tests) can also proceed in parallel once Step 5 is complete.