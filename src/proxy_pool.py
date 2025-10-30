"""Проектная обёртка над ``ProxyPool`` из ``avito_library``."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from avito_library.reuse_utils import proxy_pool as base_proxy_pool

from .log import log_event

base_proxy_pool.log_event = log_event

ProxyEndpoint = base_proxy_pool.ProxyEndpoint

__all__ = ["ProxyPool", "ProxyEndpoint"]


def _get_setting(settings: Mapping[str, Any] | Any, name: str) -> Path:
    """Получить путь из словаря настроек или объекта-конфига."""
    if isinstance(settings, Mapping):
        value = settings[name]
    else:
        value = getattr(settings, name)
    return Path(value)


class ProxyPool(base_proxy_pool.ProxyPool):
    """Расширенный пул прокси с дополнительными методами управления."""

    @classmethod
    async def create(cls, settings: Mapping[str, Any] | Any) -> "ProxyPool":
        """Инициализировать пул по настройкам проекта."""
        proxies_file = _get_setting(settings, "PROXIES_FILE")
        blocked_file = _get_setting(settings, "BLOCKED_PROXIES_FILE")
        pool = cls(proxies_file=proxies_file, blocked_file=blocked_file)
        total = await pool.reload()
        # Сигнализируем запуск пулу: сколько адресов и сколько уже в блоке.
        log_event("proxy_pool_ready", proxies=total, blocked=len(pool._blocked))
        return pool

    async def refresh_blocked(self) -> int:
        """Перечитать файл блокировок и обновить состояние."""
        blocked = await self._read_blocked()
        async with self._lock:
            # Переносим новые записи blocklist в память.
            self._blocked = blocked
            for proxy in self._proxies:
                proxy.is_blocked = proxy.address in blocked
            has_available = self._has_unblocked_locked()
        self._set_availability_event(has_available)
        log_event("proxy_blocklist_refreshed", blocked=len(blocked))
        return len(blocked)

    async def mark_available(self, address: str, *, reason: str = "manual_unblock") -> bool:
        """Удалить прокси из blacklist и открыть для повторного использования."""
        removed = False
        async with self._lock:
            if address in self._blocked:
                self._blocked.remove(address)
                removed = True
            proxy = self._proxy_map.get(address)
            if proxy:
                proxy.is_blocked = False
                proxy.failures = 0
            self._in_use.discard(address)
            has_available = self._has_unblocked_locked()
        if removed:
            await asyncio.to_thread(self._remove_blocked_record, address)
            log_event("proxy_available", proxy=address, reason=reason)
        self._set_availability_event(has_available)
        return removed

    async def list_available(self) -> Sequence[ProxyEndpoint]:
        """Вернуть список всех доступных сейчас прокси."""
        proxies = await self.all_proxies()
        return tuple(proxy for proxy in proxies if not proxy.is_blocked)

    # ------------------------------------------------------------------ #
    # Вспомогательные методы                                            #
    # ------------------------------------------------------------------ #

    def _remove_blocked_record(self, address: str) -> None:
        try:
            lines = self._blocked_file.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return
        filtered = [line for line in lines if f"\t{address}\t" not in line]
        if filtered:
            content = "\n".join(filtered) + "\n"
            self._blocked_file.parent.mkdir(parents=True, exist_ok=True)
            self._blocked_file.write_text(content, encoding="utf-8")
        else:
            # Если адрес последний в списке — удаляем файл blacklist целиком.
            self._blocked_file.unlink(missing_ok=True)
