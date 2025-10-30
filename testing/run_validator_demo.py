"""Тестовый запуск валидации каталога.

Скрипт генерирует минимальный набор объявлений и вызывает
`validator_runner.validate_catalog`, чтобы проверить интеграцию с
текущим LLM-провайдером.

Запуск: `python testing/run_validator_demo.py`
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from pprint import pprint
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validator_runner import ValidationError, validate_catalog


SAMPLE_LISTINGS = [
    {
        "item_id": "demo-1",
        "seller_id": "seller-1",
        "seller_name": "ООО Деталь",
        "title": "Контрактный двигатель BMW",  # будет отфильтровано как запрещённое
        "snippet_text": "Контрактный двигатель, доставка по РФ",
    },
    {
        "item_id": "demo-2",
        "seller_id": "seller-2",
        "seller_name": "АвтоМастер",
        "title": "Новый датчик ABS для BMW",
        "snippet_text": "Оригинал, гарантия 12 месяцев",
    },
]


async def main() -> None:
    outcome = await validate_catalog(
        article="demo-article",
        listings=SAMPLE_LISTINGS,
        source_url="https://example.com/catalog?q=demo-article",
    )
    pprint(outcome)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ValidationError as exc:
        print(f"Validation failed with error: {exc}")
