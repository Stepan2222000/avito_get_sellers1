"""Проверка интеграции DeepInfra через AsyncOpenAI Chat Completions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from pprint import pprint
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.validator_runner import ValidationError, validate_catalog


SAMPLE_LISTINGS = [
    {
        "item_id": "deepinfra-1",
        "seller_id": "seller-a",
        "seller_name": "АвтоМастер",
        "title": "Новый датчик ABS для BMW",
        "snippet_text": "Оригинал, гарантия 12 месяцев",
    },
    {
        "item_id": "deepinfra-2",
        "seller_id": "seller-b",
        "seller_name": "ООО Деталь",
        "title": "Контрактный двигатель BMW",
        "snippet_text": "Контрактный двигатель, доставка по РФ",
    },
]


def _switch_to_deepinfra() -> None:
    config.PROVIDER.name = "deepinfra"
    config.PROVIDER_NAME = "deepinfra"  # type: ignore[attr-defined]
    config.USE_OPENAI = False  # type: ignore[attr-defined]
    config.USE_DEEPINFRA = True  # type: ignore[attr-defined]


async def main() -> None:
    _switch_to_deepinfra()
    outcome = await validate_catalog(
        article="deepinfra-demo",
        listings=SAMPLE_LISTINGS,
        source_url="https://example.com/catalog?q=deepinfra-demo",
    )
    pprint(outcome)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ValidationError as exc:
        print(f"Validation failed with error: {exc}")
