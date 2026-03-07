# Agent Instructions

Instructions for AI agents working on this codebase.

## Python and uv

This project uses **uv** for dependency management and running Python. Do not use pip, poetry, or other package managers.

- **Install dependencies**: `uv sync`
- **Run Python scripts**: `uv run <script-or-module>` — never use `python -m` or bare `python script.py`
- **Add a dependency**: Add to `pyproject.toml` under `[project] dependencies`, then run `uv sync`

Examples:
```bash
uv run woordhaar-ingest
uv run pytest
uv run python -c "print('hello')"   # interactive one-liners: uv run python -c "..."
```

## Project layout

- `backend/` — Python package (FastAPI, ingestion, providers)
- `frontend/` — SvelteKit (planned)
- `data/` — Downloaded dictionary sources (gitignored)
- `plan.md` — Implementation plan and architecture
