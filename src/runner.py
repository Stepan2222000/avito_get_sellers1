"""Оркестратор асинхронной обработки каталогов Avito."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import config
from src.log import log_event
from src.proxy_pool import ProxyPool
from src.queue import CatalogTaskQueue
from src.worker import CatalogWorker
from src.validation_executor import ValidationExecutor

WORKER_RESTART_DELAY = 1.0


async def _read_lines(path: Path) -> list[str]:
    """Асинхронно прочитать файл, где каждая строка — потенциальный URL."""
    try:
        text = await asyncio.to_thread(path.read_text, "utf-8")
    except FileNotFoundError:
        return []
    return text.splitlines()


def _deduplicate_urls(lines: Iterable[str]) -> list[str]:
    """Убрать комментарии и дубликаты, сохранив порядок ссылок."""
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in lines:
        url = raw.strip()
        if not url or url.startswith("#"):
            continue
        if url in seen:
            continue
        seen.add(url)
        ordered.append(url)
    return ordered


def _result_path(item_id: str) -> Path:
    """Определить путь до JSON-файла по идентификатору задачи."""
    return config.RESULTS_DIR / f"{item_id}.json"


def _existing_success(path: Path) -> bool:
    """Проверить, существует ли уже успешный результат для каталога."""
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return bool(data.get("is_complete"))


async def _prepare_tasks(urls: Iterable[str]) -> list[Tuple[str, dict[str, str]]]:
    """Собрать payload для очереди, пропуская каталоги с готовым результатом."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tasks: list[Tuple[str, dict[str, str]]] = []
    for url in urls:
        item_id = CatalogTaskQueue.make_item_id(url)
        result_path = _result_path(item_id)
        if _existing_success(result_path):
            log_event("task_skipped_existing", url=url, file=str(result_path))
            continue
        payload = {
            "url": url,
            "item_id": item_id,
            "result_path": str(result_path),
        }
        tasks.append((item_id, payload))
    return tasks


async def bootstrap_runner() -> tuple[CatalogTaskQueue, ProxyPool, ValidationExecutor]:
    """Прочитать входные данные, инициализировать очередь и пул прокси."""
    links = await _read_lines(config.LINKS_FILE)
    urls = _deduplicate_urls(links)
    task_items = await _prepare_tasks(urls)

    queue = CatalogTaskQueue(max_attempts=config.MAX_PROXY_ATTEMPTS)
    inserted = await queue.put_many(task_items)

    proxy_pool = await ProxyPool.create(config)
    validation_executor = ValidationExecutor(
        queue=queue,
        proxy_pool=proxy_pool,
        concurrency=config.VALIDATION_CONCURRENCY,
        request_timeout=config.VALIDATION_REQUEST_TIMEOUT_SEC,
        max_retries=config.VALIDATION_MAX_RETRIES,
        retry_delay=config.VALIDATION_RETRY_DELAY_SEC,
    )
    validation_executor.start()

    log_event(
        "runner_bootstrap",
        total_urls=len(urls),
        queued=inserted,
        worker_count=config.WORKER_COUNT,
        validation_concurrency=config.VALIDATION_CONCURRENCY,
    )
    return queue, proxy_pool, validation_executor


async def runner_loop(queue: CatalogTaskQueue, proxy_pool: ProxyPool, validation_executor: ValidationExecutor) -> None:
    """Создать воркеры и подождать их завершения."""
    tasks = [
        asyncio.create_task(
            _run_worker_with_restart(
                worker_id=i,
                queue=queue,
                proxy_pool=proxy_pool,
                validation_executor=validation_executor,
            )
        )
        for i in range(config.WORKER_COUNT)
    ]

    # Гарантируем, что поддерживаем постоянное число воркеров,
    # перезапуская их при аварийном завершении.
    try:
        await asyncio.gather(*tasks)
        await validation_executor.drain()
    finally:
        await validation_executor.shutdown()


async def main() -> None:
    """Точка входа для запуска через `python -m src.runner`."""
    queue, proxy_pool, validation_executor = await bootstrap_runner()
    await runner_loop(queue, proxy_pool, validation_executor)


async def _run_worker_with_restart(
    *,
    worker_id: int,
    queue: CatalogTaskQueue,
    proxy_pool: ProxyPool,
    validation_executor: ValidationExecutor,
) -> None:
    """Обёртка, которая перезапускает воркер при аварийном завершении."""
    restart_count = 0
    while True:
        worker = CatalogWorker(
            worker_id=worker_id,
            queue=queue,
            proxy_pool=proxy_pool,
            validation_executor=validation_executor,
        )
        try:
            await worker.run()
            log_event("worker_stopped", worker_id=worker_id, reason="completed")
            break
        except asyncio.CancelledError:
            log_event("worker_stopped", worker_id=worker_id, reason="cancelled")
            raise
        except Exception as exc:  # pylint: disable=broad-except
            restart_count += 1
            log_event(
                "worker_crashed",
                worker_id=worker_id,
                error=exc.__class__.__name__,
                restart_count=restart_count,
            )
            await asyncio.sleep(WORKER_RESTART_DELAY)
            log_event("worker_restart", worker_id=worker_id, restart_count=restart_count)
            continue


if __name__ == "__main__":
    asyncio.run(main())
