"""Вспомогательные функции для валидации каталогов через Google GenAI."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

__all__ = [
    "sanitize_field",
    "format_listing_block",
    "build_prompt",
    "parse_model_response",
    "normalize_flag",
]


def sanitize_field(value: Any) -> str:
    """Нормализовать произвольное значение для использования в подсказке."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return " ".join(str(value).split())


def format_listing_block(index: int, listing: Mapping[str, Any]) -> str:
    """Представить карточку каталога компактным текстовым блоком."""

    parts = [
        f"Listing {index}",
        f"item_reference: {sanitize_field(listing.get('item_id'))}",
        f"title: {sanitize_field(listing.get('title'))}",
        f"price: {sanitize_field(listing.get('price'))}",
        f"snippet: {sanitize_field(listing.get('snippet_text'))}",
        f"location_city: {sanitize_field(listing.get('location_city'))}",
        f"location_area: {sanitize_field(listing.get('location_area'))}",
        f"location_extra: {sanitize_field(listing.get('location_extra'))}",
        f"seller_name: {sanitize_field(listing.get('seller_name'))}",
        f"seller_rating: {sanitize_field(listing.get('seller_rating'))}",
        f"seller_reviews: {sanitize_field(listing.get('seller_reviews'))}",
        f"promoted: {sanitize_field(listing.get('promoted'))}",
        f"published_ago: {sanitize_field(listing.get('published_ago'))}",
    ]
    return "\n".join(parts)


def build_prompt(article: str, listings: Sequence[Mapping[str, Any]]) -> str:
    """Построить инструкцию для модели с учётом артикула и карточек."""

    text_parts: list[str] = []

    text_parts.append(
        "Ты — эксперт по подбору оригинальных новых автомобильных запчастей. "
        "Работай только на русском языке. Мы рассматриваем объявления каталога Авито, "
        "каждое объявление описывает запчасть. Для каждой карточки нужно решить, "
        "валидна ли она."
    )
    text_parts.append(
        "Запрос клиента — найти детали строго по артикулу. Артикул, с которым мы работаем: "
        f"{article}. Сравнивай название, описание и атрибуты. Если речь идёт не об этой "
        "детали, объявление обязательно отклоняй."
    )
    text_parts.append(
        "Проверяй признаки оригинальности и новизны. Положительные сигналы: слова 'оригинал', "
        "'OEM', 'новый', 'в упаковке'. Отрицательные сигналы: 'аналог', 'копия', 'реплика', "
        "'восстановленный', 'контрактный', 'разбор', 'б/у'. Любой признак б/у или авторазбора "
        "делает объявление невалидным."
    )
    text_parts.append(
        "Если цена отсутствует, объявление невалидно. Если цена сильно ниже обычной, "
        "полагайся на текст: допускай только в случае явного подтверждения, что это новая "
        "оригинальная деталь именно по нашему артикулу."
    )
    text_parts.append(
        "Ответ возвращай строго в JSON-формате: {\"listings\":[{\"item_reference\":\"...\",\"is_valid\":true}]}. "
        "Никаких пояснений, только перечисление объявлений из списка. Если карточка должна быть "
        "отклонена, ставь is_valid=false."
    )

    for index, listing in enumerate(listings, start=1):
        text_parts.append("")
        text_parts.append(format_listing_block(index, listing))

    return "\n".join(text_parts)


def parse_model_response(raw_text: str) -> dict[str, bool]:
    """Разобрать ответ модели в отображение item_reference -> is_valid."""

    text = (raw_text or "").strip()
    if not text:
        return {}

    data: Any
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return {}
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}

    if isinstance(data, dict):
        items = data.get("listings")
    elif isinstance(data, list):
        items = data
    else:
        return {}

    decisions: dict[str, bool] = {}
    if not isinstance(items, list):
        return decisions

    for entry in items:
        if not isinstance(entry, Mapping):
            continue
        reference = entry.get("item_reference")
        if not isinstance(reference, str) or not reference:
            continue
        flag = normalize_flag(entry.get("is_valid"))
        if flag is None:
            continue
        decisions[reference] = flag

    return decisions


def normalize_flag(value: Any) -> bool | None:
    """Преобразовать значение модели в булеву форму."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1", "valid", "ok"}:
            return True
        if lowered in {"false", "no", "n", "0", "invalid"}:
            return False
    return None
