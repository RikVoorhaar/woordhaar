"""Application configuration (env-var overridable)."""

from __future__ import annotations

import os


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


DB_PATH = _env("WOORDHAAR_DB_PATH", "woordhaar.db")
OLLAMA_BASE_URL = _env("WOORDHAAR_OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = _env("WOORDHAAR_LLM_MODEL", "qwen3.5:35b-a3b")
LLM_TEMPERATURE = _env_float("WOORDHAAR_LLM_TEMPERATURE", 0.1)
LLM_TEMP_RANKING = _env_float("WOORDHAAR_LLM_TEMP_RANKING", 0.0)
LLM_TIMEOUT = _env_int("WOORDHAAR_LLM_TIMEOUT", 15)
LOG_LEVEL = _env("WOORDHAAR_LOG_LEVEL", "INFO")
