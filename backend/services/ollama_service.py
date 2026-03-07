"""Ollama-backed LLM service implementation."""

from __future__ import annotations

import json

import httpx

from backend import config
from backend.models import (
    LLMRankingOutput,
    LLMRankingResponse,
    LLMTranslationResult,
    RankedTranslation,
    TranslationCandidate,
    TranslationContext,
)
from backend.providers.base import DictionaryEntry
from backend.services.base import LLMService, LLMUnavailableError
from backend.services.prompts import build_ranking_prompt, build_translation_prompt


class OllamaLLMService(LLMService):
    """LLM service using Ollama's /api/chat endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        temp_ranking: float | None = None,
    ) -> None:
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.LLM_MODEL
        self.timeout = timeout if timeout is not None else float(config.LLM_TIMEOUT)
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.temp_ranking = temp_ranking if temp_ranking is not None else config.LLM_TEMP_RANKING

    async def _chat(
        self,
        client: httpx.AsyncClient,
        messages: list[dict],
        temperature: float,
    ) -> dict:
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }
        resp = await client.post(url, json=body, timeout=self.timeout)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMUnavailableError(f"Ollama HTTP error: {e}") from e
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        if not content or not content.strip():
            raise LLMUnavailableError("Empty response from Ollama")
        return json.loads(content)

    async def _chat_with_retry(
        self,
        messages: list[dict],
        temperature: float,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            try:
                return await self._chat(client, messages, temperature)
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                try:
                    return await self._chat(client, messages, temperature)
                except (httpx.TimeoutException, httpx.ConnectError):
                    raise LLMUnavailableError("Ollama unreachable or timeout") from e
            except LLMUnavailableError:
                raise
            except json.JSONDecodeError as e:
                raise LLMUnavailableError("Invalid JSON from Ollama") from e

    async def generate_translations(
        self,
        word: str,
        source_lang: str,
        target_langs: list[str],
        context: TranslationContext,
    ) -> LLMTranslationResult:
        messages = build_translation_prompt(context, source_lang, target_langs)
        data = await self._chat_with_retry(messages, self.temperature)
        return LLMTranslationResult.model_validate(data)

    async def filter_and_rank(
        self,
        candidates: list[TranslationCandidate],
        source_entry: DictionaryEntry,
        target_entries: dict[str, list[DictionaryEntry]],
    ) -> list[RankedTranslation]:
        if not candidates:
            return []
        source_def = source_entry.definitions[0] if source_entry.definitions else ""
        messages = build_ranking_prompt(candidates, source_def, target_entries)
        data = await self._chat_with_retry(messages, self.temp_ranking)
        parsed = LLMRankingResponse.model_validate(data)
        ranked: list[tuple[int, LLMRankingOutput, TranslationCandidate]] = []
        for i, out in enumerate(parsed.candidates):
            if i >= len(candidates) or not out.keep:
                continue
            ranked.append((out.rank, out, candidates[i]))
        ranked.sort(key=lambda x: x[0])
        return [
            RankedTranslation(
                word=cand.word,
                language=cand.language,
                confidence=out.confidence,
                definition=cand.definition,
                is_cognate=out.is_cognate,
                notes=out.notes,
            )
            for _, out, cand in ranked
        ]
