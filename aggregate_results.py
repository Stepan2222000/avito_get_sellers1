"""Агрегатор продавцов из результатов парсинга."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import config


class SummaryBuilder:
    def __init__(self) -> None:
        self._users: dict[str, dict] = {}
        self._stats = defaultdict(int)

    def add_record(self, record: dict) -> None:
        key = record.get("seller_id") or record.get("seller_url")
        if not key:
            self._stats["skipped_without_key"] += 1
            return
        reviews = _as_int(record.get("seller_reviews"))
        existing = self._users.get(key)
        if existing is None or reviews > _as_int(existing.get("seller_reviews")):
            record = dict(record)
            record["seller_reviews"] = reviews
            self._users[key] = record
            self._stats["upserted"] += 1
        else:
            self._stats["duplicate"] += 1

    def build_summary(self) -> dict:
        threshold = getattr(config, 'MIN_REVIEWS_FOR_SUMMARY', 0)
        sellers = [user for user in self._users.values() if user["seller_reviews"] >= threshold]
        sellers.sort(key=lambda item: item["seller_reviews"], reverse=True)
        return {
            "seller_count": len(sellers),
            "total_unique": len(self._users),
            "threshold": threshold,
            "stats": dict(self._stats),
            "sellers": sellers,
        }


def _as_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_results(results_dir: Path) -> Iterable[dict]:
    for path in results_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        sellers = data.get("sellers")
        if not sellers:
            continue
        yield from sellers


def main() -> None:
    results_dir = config.RESULTS_DIR
    if not results_dir.exists():
        print(f"results dir {results_dir} missing")
        return

    builder = SummaryBuilder()
    for seller in load_results(results_dir):
        builder.add_record(seller)

    summary = builder.build_summary()
    output_path = results_dir / "summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(f"summary written to {output_path}")


if __name__ == "__main__":
    main()
