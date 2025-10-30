"""Проектный логгер с единым форматом `event=<name> key=value`."""
from __future__ import annotations

from typing import Any, Mapping

import json
import sys

__all__ = ["log_event"]


def _stringify(value: Any) -> str:
    """Преобразовать значение к тексту без потери структуры."""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def log_event(event: str, *payloads: Mapping[str, Any] | None, **context: Any) -> None:
    """Вывести строку лога в проектном формате.

    Пример:
        log_event("task_start", item_id="abc", attempt=1)
        # event=task_start item_id=abc attempt=1
    """

    data: dict[str, Any] = {}
    for payload in payloads:
        if payload:
            # Объединяем именованные параметры из переданных маппингов.
            data.update(payload)
    if context:
        # Позиционные аргументы могут дополняться kwargs.
        data.update(context)

    parts = [f"event={event}"]
    for key in sorted(data):
        # Сортировка ключей облегчает чтение и сравнение логов.
        parts.append(f"{key}={_stringify(data[key])}")
    sys.stdout.write(" ".join(parts) + "\n")
    sys.stdout.flush()
