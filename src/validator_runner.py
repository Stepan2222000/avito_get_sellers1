"""Асинхронный запуск валидации каталога через LLM-провайдер."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

from google import genai
from google.genai import types
from openai import AsyncOpenAI

import config
from src.log import log_event
from src.validator import build_prompt, parse_model_response


_logger = logging.getLogger(__name__)
if not _logger.handlers:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
else:
    _logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))


BANNED_KEYWORDS = (
    "авторазбор",
    "авто разбор",
    "разборка",
    "разбор",
    "разборы",
    "контракт",
    "контрактн",
    "б/у",
    "б \u002f у",  # встречается экранирование
    "бу",
    "б.у",
    "шрот",
    "восстановл",
)


class ValidationError(RuntimeError):
    """Ошибка валидации каталога."""


@dataclass(slots=True)
class ValidationOutcome:
    """Итог валидации каталога."""

    article: str
    total: int
    skipped_prefilter: int
    model_invalid: int
    model_missing: int
    model_used: bool
    valid_listings: list[dict[str, Any]]
    invalid_sellers: set[str]


def extract_article_from_url(url: str) -> str:
    """Вытащить артикул из параметра `q` ссылки каталога."""

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    value = query.get("q", [""])[0]
    return value or ""


def _contains_banned(text: str | None) -> bool:
    if not text:
        return False
    normalized = text.casefold()
    return any(keyword in normalized for keyword in BANNED_KEYWORDS)


def _seller_key(listing: Mapping[str, Any]) -> str | None:
    seller_id = listing.get("seller_id")
    if isinstance(seller_id, str) and seller_id:
        return f"id::{seller_id}"
    seller_name = listing.get("seller_name")
    if isinstance(seller_name, str) and seller_name:
        return f"name::{seller_name.strip()}"
    item_id = listing.get("item_id")
    if isinstance(item_id, str) and item_id:
        return f"item::{item_id}"
    return None


def _prefilter_listings(
    listings: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], set[str]]:
    candidates: list[tuple[str | None, dict[str, Any]]] = []
    invalid_sellers: set[str] = set()
    for listing in listings:
        key = _seller_key(listing)
        title = listing.get("title")
        snippet = listing.get("snippet_text")
        seller_name = listing.get("seller_name")
        if _contains_banned(title) or _contains_banned(snippet) or _contains_banned(seller_name):
            if key:
                invalid_sellers.add(key)
            continue
        candidates.append((key, dict(listing)))

    allowed = [payload for key, payload in candidates if not key or key not in invalid_sellers]
    return allowed, invalid_sellers


async def validate_catalog(
    *,
    article: str,
    listings: Sequence[Mapping[str, Any]],
    source_url: str,
) -> ValidationOutcome:
    """Отфильтровать карточки каталога и вернуть только валидные."""

    total = len(listings)
    prefiltered, invalid_sellers = _prefilter_listings(listings)
    skipped = total - len(prefiltered)

    provider_name = config.PROVIDER_NAME
    if provider_name == "openai":
        api_key = config.OPENAI_API_KEY
        model_name = config.OPENAI_MODEL
        requires_model = True
        provider_label = "openai"
    elif provider_name == "deepinfra":
        api_key = config.DEEPINFRA_API_KEY
        model_name = config.DEEPINFRA_MODEL
        requires_model = True
        provider_label = "deepinfra"
    else:
        api_key = config.GENAI_API_KEY
        model_name = config.GENAI_MODEL
        requires_model = False
        provider_label = "gemini"

    model_configured = bool(api_key) and (bool(model_name) if requires_model else True)

    if not prefiltered:
        log_event(
            "validation_skipped_all",
            article=article,
            skipped_prefilter=skipped,
            url=source_url,
        )
        return ValidationOutcome(
            article=article,
            total=total,
            skipped_prefilter=skipped,
            model_invalid=0,
            model_missing=0,
            model_used=model_configured,
            valid_listings=[],
            invalid_sellers=set(invalid_sellers),
        )

    if not api_key:
        log_event(
            "validation_disabled",
            article=article,
            reason="missing_api_key",
            kept=len(prefiltered),
            url=source_url,
            provider=provider_label,
        )
        allowed = [item for item in prefiltered if _seller_key(item) not in invalid_sellers]
        return ValidationOutcome(
            article=article,
            total=total,
            skipped_prefilter=skipped,
            model_invalid=0,
            model_missing=0,
            model_used=False,
            valid_listings=allowed,
            invalid_sellers=set(invalid_sellers),
        )

    if requires_model and not model_name:
        log_event(
            "validation_disabled",
            article=article,
            reason="missing_model",
            kept=len(prefiltered),
            url=source_url,
            provider=provider_label,
        )
        allowed = [item for item in prefiltered if _seller_key(item) not in invalid_sellers]
        return ValidationOutcome(
            article=article,
            total=total,
            skipped_prefilter=skipped,
            model_invalid=0,
            model_missing=0,
            model_used=False,
            valid_listings=allowed,
            invalid_sellers=set(invalid_sellers),
        )

    prompt = build_prompt(article, prefiltered)
    log_event(
        "validation_request",
        article=article,
        listings=len(prefiltered),
        url=source_url,
    )

    try:
        decisions = await _request_decisions(prompt)
    except ValidationError as exc:
        log_event(
            "validation_failed",
            article=article,
            reason=str(exc),
            url=source_url,
        )
        return ValidationOutcome(
            article=article,
            total=total,
            skipped_prefilter=skipped,
            model_invalid=len(prefiltered),
            model_missing=0,
            model_used=True,
            valid_listings=[],
            invalid_sellers=set(invalid_sellers),
        )

    valid_refs = {ref for ref, is_valid in decisions.items() if is_valid}
    decided_refs = set(decisions.keys())

    model_invalid = 0
    model_missing = 0
    for item in prefiltered:
        reference = item.get("item_id")
        key = _seller_key(item)
        if reference not in decided_refs:
            model_missing += 1
            if key:
                invalid_sellers.add(key)
            continue
        if reference not in valid_refs:
            model_invalid += 1
            if key:
                invalid_sellers.add(key)

    valid_listings = [
        item for item in prefiltered if item.get("item_id") in valid_refs and _seller_key(item) not in invalid_sellers
    ]

    log_event(
        "validation_done",
        article=article,
        valid=len(valid_listings),
        invalid=model_invalid,
        undecided=model_missing,
        url=source_url,
    )

    return ValidationOutcome(
        article=article,
        total=total,
        skipped_prefilter=skipped,
        model_invalid=model_invalid,
        model_missing=model_missing,
        model_used=True,
        valid_listings=valid_listings,
        invalid_sellers=set(invalid_sellers),
    )


async def _request_decisions(prompt: str) -> dict[str, bool]:
    """Отправить подсказку в выбранный LLM с учётом повторов."""

    if config.USE_OPENAI:
        return await _request_decisions_openai(prompt)
    if config.USE_DEEPINFRA:
        return await _request_decisions_deepinfra(prompt)
    return await _request_decisions_gemini(prompt)


async def _request_decisions_gemini(prompt: str) -> dict[str, bool]:
    """Запрос решений у Google Gemini с повторами."""

    for attempt in range(1, config.GENAI_MAX_RETRIES + 1):
        try:
            log_event(
                "validation_api_attempt",
                attempt=attempt,
                model=config.GENAI_MODEL,
                provider="gemini",
            )
            async with genai.Client(api_key=config.GENAI_API_KEY).aio as aclient:
                response_config = types.GenerateContentConfig(
                    responseMimeType="application/json",
                    automaticFunctionCalling=types.AutomaticFunctionCallingConfig(disable=True),
                )
                response = await asyncio.wait_for(
                    aclient.models.generate_content(
                        model=config.GENAI_MODEL,
                        contents=prompt,
                        config=response_config,
                    ),
                    timeout=config.GENAI_REQUEST_TIMEOUT_SEC,
                )
            raw_text = _extract_genai_response_text(response)
            decisions = parse_model_response(raw_text)
            if decisions:
                log_event(
                    "validation_api_success",
                    attempt=attempt,
                    decisions=len(decisions),
                    response_chars=len(raw_text),
                    provider="gemini",
                )
                return decisions
            log_event(
                "validation_api_empty",
                attempt=attempt,
                response_chars=len(raw_text),
                provider="gemini",
            )
        except asyncio.TimeoutError:
            log_event(
                "validation_api_retry",
                attempt=attempt,
                reason="timeout",
                provider="gemini",
            )
        except Exception as exc:  # noqa: BLE001
            log_event(
                "validation_api_retry",
                attempt=attempt,
                reason=exc.__class__.__name__,
                error_message=str(exc),
                provider="gemini",
            )

        await asyncio.sleep(config.GENAI_RETRY_DELAY_SEC * attempt)

    log_event(
        "validation_api_failed",
        attempts=config.GENAI_MAX_RETRIES,
        provider="gemini",
    )
    raise ValidationError("genai_exhausted")


async def _request_decisions_openai(prompt: str) -> dict[str, bool]:
    """Запрос решений у OpenAI Responses API с повторами."""

    if not config.OPENAI_API_KEY:
        raise ValidationError("openai_missing_api_key")
    if not config.OPENAI_MODEL:
        raise ValidationError("openai_missing_model")

    async with AsyncOpenAI(api_key=config.OPENAI_API_KEY) as client:
        for attempt in range(1, config.OPENAI_MAX_RETRIES + 1):
            try:
                log_event(
                    "validation_api_attempt",
                    attempt=attempt,
                    model=config.OPENAI_MODEL,
                    provider="openai",
                )
                response = await asyncio.wait_for(
                    client.responses.create(
                        model=config.OPENAI_MODEL,
                        input=prompt,
                    ),
                    timeout=config.OPENAI_REQUEST_TIMEOUT_SEC,
                )
                raw_text = _extract_openai_response_text(response)
                decisions = parse_model_response(raw_text)
                if decisions:
                    log_event(
                        "validation_api_success",
                        attempt=attempt,
                        decisions=len(decisions),
                        response_chars=len(raw_text),
                        provider="openai",
                    )
                    return decisions
                log_event(
                    "validation_api_empty",
                    attempt=attempt,
                    response_chars=len(raw_text),
                    provider="openai",
                )
            except asyncio.TimeoutError:
                log_event(
                    "validation_api_retry",
                    attempt=attempt,
                    reason="timeout",
                    provider="openai",
                )
            except Exception as exc:  # noqa: BLE001
                log_event(
                    "validation_api_retry",
                    attempt=attempt,
                    reason=exc.__class__.__name__,
                    error_message=str(exc),
                    provider="openai",
                )

            await asyncio.sleep(config.OPENAI_RETRY_DELAY_SEC * attempt)

    log_event(
        "validation_api_failed",
        attempts=config.OPENAI_MAX_RETRIES,
        provider="openai",
    )
    raise ValidationError("genai_exhausted")


async def _request_decisions_deepinfra(prompt: str) -> dict[str, bool]:
    """Запрос решений у DeepInfra Chat Completions API с повторами."""

    if not config.DEEPINFRA_API_KEY:
        raise ValidationError("deepinfra_missing_api_key")
    if not config.DEEPINFRA_MODEL:
        raise ValidationError("deepinfra_missing_model")

    async with AsyncOpenAI(api_key=config.DEEPINFRA_API_KEY, base_url=config.DEEPINFRA_BASE_URL) as client:
        for attempt in range(1, config.DEEPINFRA_MAX_RETRIES + 1):
            try:
                log_event(
                    "validation_api_attempt",
                    attempt=attempt,
                    model=config.DEEPINFRA_MODEL,
                    provider="deepinfra",
                )
                request_kwargs: dict[str, Any] = {
                    "model": config.DEEPINFRA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                }
                if config.DEEPINFRA_REASONING_ENABLED:
                    request_kwargs["extra_body"] = {
                        "reasoning": {
                            "enabled": True,
                            "effort": config.DEEPINFRA_REASONING_EFFORT,
                        }
                    }
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        **request_kwargs,
                    ),
                    timeout=config.DEEPINFRA_REQUEST_TIMEOUT_SEC,
                )
                raw_text = _extract_chat_completion_text(response)
                decisions = parse_model_response(raw_text)
                if decisions:
                    log_event(
                        "validation_api_success",
                        attempt=attempt,
                        decisions=len(decisions),
                        response_chars=len(raw_text),
                        provider="deepinfra",
                    )
                    return decisions
                log_event(
                    "validation_api_empty",
                    attempt=attempt,
                    response_chars=len(raw_text),
                    provider="deepinfra",
                )
            except asyncio.TimeoutError:
                log_event(
                    "validation_api_retry",
                    attempt=attempt,
                    reason="timeout",
                    provider="deepinfra",
                )
            except Exception as exc:  # noqa: BLE001
                log_event(
                    "validation_api_retry",
                    attempt=attempt,
                    reason=exc.__class__.__name__,
                    error_message=str(exc),
                    provider="deepinfra",
                )

            await asyncio.sleep(config.DEEPINFRA_RETRY_DELAY_SEC * attempt)

    log_event(
        "validation_api_failed",
        attempts=config.DEEPINFRA_MAX_RETRIES,
        provider="deepinfra",
    )
    raise ValidationError("deepinfra_exhausted")


def _extract_genai_response_text(response: Any) -> str:
    """Получить текст из ответа Google GenAI SDK."""

    if hasattr(response, "text") and response.text:
        return str(response.text)

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None)
        if not parts:
            continue
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                return str(text)
    return ""


def _extract_openai_response_text(response: Any) -> str:
    """Получить текст из ответа OpenAI SDK."""

    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    if hasattr(response, "model_dump"):
        data = response.model_dump()
    else:
        data = response

    output = data.get("output") if isinstance(data, dict) else getattr(data, "output", None)
    if not output:
        return ""

    for block in output:
        content = block.get("content") if isinstance(block, dict) else getattr(block, "content", None)
        if not content:
            continue
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                return str(text)

    return ""


def _extract_chat_completion_text(response: Any) -> str:
    """Получить текст из ответа Chat Completions."""

    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    for choice in choices:
        message = getattr(choice, "message", None)
        if not message:
            message = choice.get("message") if isinstance(choice, Mapping) else None
        if not message:
            continue
        content = getattr(message, "content", None)
        if not content and isinstance(message, Mapping):
            content = message.get("content")
        if content:
            return str(content)
    return ""
