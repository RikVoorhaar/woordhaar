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

## Implementation discipline

When you add or change code (scripts, APIs, etc.), run it to confirm it works before asking the user for feedback — unless the operation would be destructive. If it fails, fix it and re-run until it succeeds. Do not report "done" until you have verified it yourself.

After any substantial change, run the unit tests to confirm they still pass. If they fail, and the cause is probably because of code you just changed, then fix it. If the failures is completely unrelated, then report the issue to the user.

Do not autonomosly decide to skip or work around an issue, such as a failing test, or an error in the code. Hiding issues is not a good strategy. If you are not sure, ask the user for instructions.

## Project layout

- `backend/` — Python package (FastAPI, ingestion, providers)
- `frontend/` — SvelteKit (planned)
- `data/` — Downloaded dictionary sources (gitignored)
- `plan.md` — Implementation plan and architecture
