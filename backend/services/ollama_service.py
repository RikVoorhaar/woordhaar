"""Ollama-backed LLM service implementation."""

from __future__ import annotations

import json
import time

import httpx
from loguru import logger

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
        call_type: str = "chat",
    ) -> dict:
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }
        
        # Estimate prompt size (rough token estimate: ~4 chars per token)
        prompt_text = json.dumps(messages, ensure_ascii=False)
        estimated_tokens = len(prompt_text) // 4
        
        log_ctx = logger.bind(model=self.model, call_type=call_type)
        log_ctx.debug(f"LLM request: {call_type}, ~{estimated_tokens} tokens, temperature={temperature}")
        
        t0 = time.perf_counter()
        try:
            resp = await client.post(url, json=body, timeout=self.timeout)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                log_ctx.error(f"LLM HTTP error: {e.response.status_code} (took {elapsed_ms}ms)")
                raise LLMUnavailableError(f"Ollama HTTP error: {e}") from e
            
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            response_size = len(content)
            
            if not content or not content.strip():
                log_ctx.warning(f"Empty response from Ollama (took {elapsed_ms}ms)")
                raise LLMUnavailableError("Empty response from Ollama")
            
            log_ctx.debug(f"LLM response: {response_size} chars (took {elapsed_ms}ms)")
            
            try:
                parsed = json.loads(content)
                return parsed
            except json.JSONDecodeError as e:
                log_ctx.error(
                    f"JSON parsing failed (took {elapsed_ms}ms). "
                    f"Response preview: {content[:200]}..."
                )
                raise LLMUnavailableError(f"Invalid JSON from Ollama: {e}") from e
                
        except httpx.TimeoutException:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            log_ctx.warning(f"LLM timeout after {elapsed_ms}ms (timeout={self.timeout}s)")
            raise
        except httpx.ConnectError as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            log_ctx.error(f"LLM connection error (took {elapsed_ms}ms): {e}")
            raise

    async def _chat_with_retry(
        self,
        messages: list[dict],
        temperature: float,
        call_type: str = "chat",
    ) -> dict:
        async with httpx.AsyncClient() as client:
            try:
                return await self._chat(client, messages, temperature, call_type=call_type)
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                logger.warning(f"LLM call failed, retrying once: {e}")
                try:
                    return await self._chat(client, messages, temperature, call_type=call_type)
                except (httpx.TimeoutException, httpx.ConnectError) as retry_e:
                    logger.error(f"LLM retry also failed: {retry_e}")
                    raise LLMUnavailableError("Ollama unreachable or timeout") from retry_e
            except LLMUnavailableError:
                raise
            except Exception as e:
                logger.error(f"Unexpected error in LLM call: {e}", exc_info=True)
                raise LLMUnavailableError(f"Unexpected error: {e}") from e

    async def generate_translations(
        self,
        word: str,
        source_lang: str,
        target_langs: list[str],
        context: TranslationContext,
    ) -> LLMTranslationResult:
        log_ctx = logger.bind(word=word, source_lang=source_lang, target_langs=target_langs)
        log_ctx.info(f"Generating translations: {source_lang}→{target_langs}")
        
        messages = build_translation_prompt(context, source_lang, target_langs)
        try:
            data = await self._chat_with_retry(messages, self.temperature, call_type="generate_translations")
            result = LLMTranslationResult.model_validate(data)
            log_ctx.info(f"Translation generation successful: {len(result.translations)} candidates")
            return result
        except LLMUnavailableError as e:
            log_ctx.error(f"Translation generation failed: {e}", exc_info=True)
            raise

    async def filter_and_rank(
        self,
        candidates: list[TranslationCandidate],
        source_entry: DictionaryEntry,
        target_entries: dict[str, list[DictionaryEntry]],
    ) -> list[RankedTranslation]:
        if not candidates:
            logger.debug("No candidates to rank")
            return []
        
        log_ctx = logger.bind(
            candidates_count=len(candidates),
            source_word=source_entry.word,
        )
        log_ctx.info(f"Filtering and ranking {len(candidates)} candidates")
        
        source_def = source_entry.definitions[0] if source_entry.definitions else ""
        messages = build_ranking_prompt(candidates, source_def, target_entries)
        
        try:
            data = await self._chat_with_retry(messages, self.temp_ranking, call_type="filter_and_rank")
            parsed = LLMRankingResponse.model_validate(data)
            
            ranked: list[tuple[int, LLMRankingOutput, TranslationCandidate]] = []
            kept_count = 0
            for i, out in enumerate(parsed.candidates):
                if i >= len(candidates):
                    log_ctx.warning(f"LLM returned more candidates than input: {len(parsed.candidates)} vs {len(candidates)}")
                    break
                if out.keep:
                    ranked.append((out.rank, out, candidates[i]))
                    kept_count += 1
            
            ranked.sort(key=lambda x: x[0])
            log_ctx.info(f"Ranking completed: {kept_count}/{len(candidates)} candidates kept")
            
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
        except LLMUnavailableError as e:
            log_ctx.error(f"Ranking failed: {e}", exc_info=True)
            raise
