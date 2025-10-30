"""Обёртка над очередью задач из `avito_library` с проектным логированием."""
from __future__ import annotations

import hashlib
from typing import Any, Hashable, Iterable, Mapping, Optional, Tuple, cast

from avito_library.reuse_utils import task_queue as base_queue

from .log import log_event

base_queue.log_event = log_event

TaskState = base_queue.TaskState

__all__ = ["CatalogTask", "CatalogTaskQueue", "TaskState", "build_task_items"]


class CatalogTask(base_queue.ProcessingTask):
    """Совместимая с прежним API задача очереди."""

    __slots__ = ()

    @property
    def item_id(self) -> str:
        """Возвращает идентификатор задачи, совместимый с историческим API."""
        payload = self.payload
        if isinstance(payload, Mapping) and "item_id" in payload:
            return cast(str, payload["item_id"])
        return str(self.task_key)

    @property
    def url(self) -> Optional[str]:
        """Удобное обращение к URL, даже если payload — простая строка."""
        payload = self.payload
        if isinstance(payload, Mapping):
            url = payload.get("url")
            return cast(Optional[str], url) if url is not None else None
        if isinstance(payload, str):
            return payload
        return None

    @property
    def wait_until_resumed(self) -> bool:
        """Флаг ожидает ли задача возобновления очереди после паузы."""
        payload = self.payload
        if isinstance(payload, Mapping):
            return bool(payload.get("wait_until_resumed", False))
        return False

    def set_wait_until_resumed(self, value: bool) -> None:
        """Позволяет воркеру отметить, что задача ждёт разблокировки очереди."""
        payload = self.payload
        if isinstance(payload, dict):
            payload["wait_until_resumed"] = value


class CatalogTaskQueue(base_queue.TaskQueue):
    """Очередь задач с детерминированными ключами и проектным логированием."""

    def __init__(self, *, max_attempts: int) -> None:
        # Базовый класс уже умеет управлять блокировками и повторными попытками.
        super().__init__(max_attempts=max_attempts)

    @staticmethod
    def make_item_id(url: str) -> str:
        """Построить детерминированный идентификатор задачи по URL."""
        return hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest()

    @staticmethod
    def _normalize_payload(task_key: Hashable, payload: Any) -> dict[str, Any]:
        if isinstance(payload, Mapping):
            normalized = dict(payload)
        else:
            # Если передана просто строка, делаем минимальный payload.
            normalized = {"url": str(payload)}
        normalized.setdefault("item_id", str(task_key))
        normalized.setdefault("wait_until_resumed", False)
        return normalized

    async def put_many(self, items: Iterable[Tuple[Hashable, Any]]) -> int:
        """Добавить сразу набор задач, исключив дубликаты по ключу."""
        inserted = 0
        async with self._lock:
            for task_key, payload in items:
                if task_key in self._tasks:
                    continue
                task_payload = self._normalize_payload(task_key, payload)
                task = CatalogTask(task_key=task_key, payload=task_payload)
                self._tasks[task_key] = task
                self._pending_order.append(task_key)
                inserted += 1
        if inserted:
            log_event("queue_enqueued", inserted=inserted)
        return inserted

    async def get(self) -> Optional[CatalogTask]:
        task = await super().get()
        if task is None:
            return None
        return cast(CatalogTask, task)

    async def enqueue_urls(self, urls: Iterable[str]) -> int:
        """Поставить в очередь набор ссылок каталога."""
        payloads = []
        for url in urls:
            clean = url.strip()
            if not clean:
                continue
            item_id = self.make_item_id(clean)
            # Формируем payload с обязательными полями.
            payloads.append((item_id, {"url": clean, "item_id": item_id}))
        return await self.put_many(payloads)


def build_task_items(urls: Iterable[str]) -> list[tuple[str, dict[str, Any]]]:
    """Подготовить пары `(task_key, payload)` для загрузки в очередь."""
    items: list[tuple[str, dict[str, Any]]] = []
    for url in urls:
        clean = url.strip()
        if not clean:
            continue
        item_id = CatalogTaskQueue.make_item_id(clean)
        # wait_until_resumed держим явным, чтобы воркер мог переключать флаг.
        items.append((item_id, {"url": clean, "item_id": item_id, "wait_until_resumed": False}))
    return items
