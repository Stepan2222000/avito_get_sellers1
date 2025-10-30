"""Реестр продавцов, с которыми уже работала LLM."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Set, Tuple


def _extract_seller_id(payload: Mapping[str, object]) -> Optional[str]:
    value = payload.get("seller_id")
    if value is None:
        return None
    if isinstance(value, str):
        clean = value.strip()
        return clean or None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


class SellerRegistry:
    """Поддерживает списки просмотренных и валидных продавцов."""

    def __init__(
        self,
        *,
        seen_path: Path,
        valid_path: Path,
        seen_ids: Set[str],
        valid_ids: Set[str],
        min_reviews_for_valid_export: int,
    ) -> None:
        self._seen_path = seen_path
        self._valid_path = valid_path
        self._seen_ids = seen_ids
        self._valid_ids = valid_ids
        self._seen_lock = asyncio.Lock()
        self._valid_lock = asyncio.Lock()
        self._min_reviews = max(0, min_reviews_for_valid_export)

    @classmethod
    async def create(
        cls,
        *,
        seen_path: Path,
        valid_path: Path,
        min_reviews_for_valid_export: int,
    ) -> "SellerRegistry":
        seen_ids = await asyncio.to_thread(cls._load_ids, seen_path)
        valid_ids = await asyncio.to_thread(cls._load_ids, valid_path)
        return cls(
            seen_path=seen_path,
            valid_path=valid_path,
            seen_ids=seen_ids,
            valid_ids=valid_ids,
            min_reviews_for_valid_export=min_reviews_for_valid_export,
        )

    @staticmethod
    def _load_ids(path: Path) -> Set[str]:
        try:
            content = path.read_text("utf-8")
        except FileNotFoundError:
            return set()
        return {line.strip() for line in content.splitlines() if line.strip()}

    async def filter_new_listings(
        self, listings: Sequence[Mapping[str, object]]
    ) -> Tuple[list[Mapping[str, object]], Set[str]]:
        """Вернуть только те карточки, чьи продавцы ещё не рассматривались."""
        filtered: list[Mapping[str, object]] = []
        skipped: Set[str] = set()
        async with self._seen_lock:
            for listing in listings:
                seller_id = _extract_seller_id(listing)
                if seller_id is None:
                    filtered.append(listing)
                    continue
                if seller_id in self._seen_ids:
                    skipped.add(seller_id)
                    continue
                filtered.append(listing)
        return filtered, skipped

    async def mark_seen(self, seller_ids: Iterable[str]) -> None:
        new_ids: list[str] = []
        async with self._seen_lock:
            for seller_id in seller_ids:
                if not seller_id:
                    continue
                if seller_id in self._seen_ids:
                    continue
                self._seen_ids.add(seller_id)
                new_ids.append(seller_id)
        if new_ids:
            await self._append_lines(self._seen_path, new_ids)

    async def append_valid_sellers(self, sellers: Iterable[Mapping[str, object]]) -> None:
        new_urls: list[str] = []
        async with self._valid_lock:
            for seller in sellers:
                seller_id = _extract_seller_id(seller)
                if not seller_id:
                    continue
                if seller_id in self._valid_ids:
                    continue
                reviews = _extract_reviews(seller.get("seller_reviews"))
                if reviews < self._min_reviews:
                    continue
                self._valid_ids.add(seller_id)
                new_urls.append(f"https://www.avito.ru/brand/{seller_id}")
        if new_urls:
            await self._append_lines(self._valid_path, new_urls)

    @staticmethod
    async def _append_lines(path: Path, lines: Sequence[str]) -> None:
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        text = "".join(f"{line}\n" for line in lines)

        def _write() -> None:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(text)

        await asyncio.to_thread(_write)


def _extract_reviews(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return 0


__all__ = ["SellerRegistry", "_extract_seller_id"]
