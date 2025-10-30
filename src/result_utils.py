"""Вспомогательные утилиты для сериализации результатов каталога."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from avito_library.parsers.catalog_parser import CatalogParseStatus

SELLER_FIELDS = ("seller_url", "seller_reviews", "seller_id", "seller_name")


def _extract_seller(listing: Any) -> Optional[Dict[str, Any]]:
    """Достать информацию о продавце из структуры каталога."""

    def _get(attr: str) -> Any:
        if isinstance(listing, Mapping):
            return listing.get(attr)
        return getattr(listing, attr, None)

    seller_url = _get("seller_url")
    seller_reviews = _get("seller_reviews")
    seller_id = _get("seller_id")
    seller_name = _get("seller_name")

    if not seller_url and not seller_id:
        return None
    return {
        "seller_url": seller_url,
        "seller_reviews": seller_reviews,
        "seller_id": seller_id,
        "seller_name": seller_name,
    }


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def collect_sellers(listings: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Собрать уникальных продавцов по максимальному числу отзывов."""

    result: Dict[str, dict[str, Any]] = {}
    for listing in listings:
        seller = _extract_seller(listing)
        if not seller:
            continue
        key = seller.get("seller_id") or seller.get("seller_url") or seller.get("seller_name")
        if not key:
            continue
        reviews = _to_int(seller.get("seller_reviews"))
        current = result.get(key)
        if current is None or reviews > _to_int(current.get("seller_reviews")):
            seller_copy = dict(seller)
            seller_copy["seller_reviews"] = reviews
            result[key] = seller_copy
    return list(result.values())


def meta_to_dict(meta: Any) -> Dict[str, Any]:
    """Сконвертировать объект метаданных в сериализуемый словарь."""
    status = getattr(meta, "status", None)
    status_value = status.value if isinstance(status, CatalogParseStatus) else status
    return {
        "status": status_value,
        "pages_processed": getattr(meta, "processed_pages", None),
        "cards_processed": getattr(meta, "processed_cards", None),
        "details": getattr(meta, "details", None),
        "last_state": getattr(meta, "last_state", None),
        "last_url": getattr(meta, "last_url", None),
    }


async def ensure_parent_dir(path: Path) -> None:
    """Создать родительскую директорию для файла, если её ещё нет."""
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)


__all__ = ["collect_sellers", "meta_to_dict", "SELLER_FIELDS", "ensure_parent_dir"]
