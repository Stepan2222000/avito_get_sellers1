"""Генератор ссылок на профили продавцов из summary.json."""
from __future__ import annotations

import json
from pathlib import Path

import config


def generate_seller_urls(summary_path: Path, output_path: Path) -> None:
    """Генерирует ссылки на профили продавцов из summary.json.
    
    Args:
        summary_path: Путь к файлу summary.json
        output_path: Путь для сохранения списка URL'ов
    """
    # Читаем summary.json
    try:
        data = json.loads(summary_path.read_text("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
        print(f"Ошибка при чтении {summary_path}: {e}")
        return
    
    sellers = data.get("sellers", [])
    if not sellers:
        print("Продавцы не найдены в summary.json")
        return
    
    # Генерируем URL'ы
    urls = []
    for seller in sellers:
        seller_id = seller.get("seller_id")
        if seller_id:
            url = f"https://www.avito.ru/brands/{seller_id}"
            urls.append(url)
    
    # Сохраняем в файл
    output_path.write_text("\n".join(urls) + "\n", "utf-8")
    
    print(f"✅ Создано {len(urls)} ссылок на профили продавцов")
    print(f"📁 Сохранено в: {output_path}")
    print(f"\nПримеры первых 5 URL'ов:")
    for url in urls[:5]:
        print(f"  - {url}")


def main() -> None:
    """Точка входа."""
    summary_path = config.RESULTS_DIR / "summary.json"
    output_path = config.RESULTS_DIR / "seller_urls.txt"
    
    if not summary_path.exists():
        print(f"❌ Файл {summary_path} не найден!")
        print("Сначала запустите aggregate_results.py")
        return
    
    generate_seller_urls(summary_path, output_path)


if __name__ == "__main__":
    main()


