"""Воркеры обработки каталогов Avito с поддержкой восстановления страниц."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
import contextvars
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from avito_library import detect_page_state, resolve_captcha_flow
from avito_library.parsers.catalog_parser import (
    CatalogParseStatus,
    parse_catalog_until_complete,
    set_page_exchange,
    supply_page,
    wait_for_page_request,
)
from avito_library.reuse_utils.task_queue import ProcessingTask
from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

import config
from src.log import log_event
from src.proxy_pool import ProxyEndpoint, ProxyPool
from src.queue import CatalogTask, CatalogTaskQueue
from src.validator_runner import extract_article_from_url, validate_catalog

CATALOG_FIELDS = {
    "item_id",
    "title",
    "price",
    "snippet_text",
    "location_city",
    "location_area",
    "location_extra",
    "seller_name",
    "seller_id",
    "seller_rating",
    "seller_reviews",
    "promoted",
    "published_ago",
}

SELLER_FIELDS = ("seller_url", "seller_reviews", "seller_id", "seller_name")

LISTING_SERIALIZED_FIELDS = tuple(sorted(CATALOG_FIELDS | set(SELLER_FIELDS)))

SUCCESS_STATUSES = {
    CatalogParseStatus.SUCCESS,
    CatalogParseStatus.EMPTY,
}

RETRYABLE_STATUSES = {
    CatalogParseStatus.CAPTCHA_UNSOLVED,
    CatalogParseStatus.RATE_LIMIT,
    CatalogParseStatus.PROXY_BLOCKED,
    CatalogParseStatus.PROXY_AUTH_REQUIRED,
    CatalogParseStatus.LOAD_FAILED,
    CatalogParseStatus.INVALID_STATE,
}

BLOCKING_PROXY_STATUSES = {
    CatalogParseStatus.PROXY_BLOCKED,
    CatalogParseStatus.PROXY_AUTH_REQUIRED,
}

DELAY_STATUSES = {
    CatalogParseStatus.RATE_LIMIT,
}

BLOCK_REASON = {
    CatalogParseStatus.PROXY_BLOCKED: "http_403",
    CatalogParseStatus.PROXY_AUTH_REQUIRED: "http_407",
}


def _task_payload(task: ProcessingTask) -> Mapping[str, Any]:
    """Нормализовать payload задачи независимо от исходного типа."""
    payload = task.payload
    if isinstance(payload, Mapping):
        return payload
    return {"url": str(payload), "item_id": str(task.task_key)}


def _result_path(payload: Mapping[str, Any]) -> Path:
    """Подобрать путь для сохранения результатов каталога."""
    value = payload.get("result_path")
    if value:
        return Path(str(value))
    item_id = str(payload.get("item_id", ""))
    if item_id:
        return config.RESULTS_DIR / f"{item_id}.json"
    return config.RESULTS_DIR / "result.json"


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


def _meta_to_dict(meta: Any) -> Dict[str, Any]:
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


async def _ensure_dir(path: Path) -> None:
    """Создать директорию под файл результата, если её ещё нет."""
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)


def _listing_to_dict(listing: Any) -> Dict[str, Any]:
    """Собрать сериализуемое представление карточки каталога."""

    data: Dict[str, Any] = {}
    if isinstance(listing, Mapping):
        for field in LISTING_SERIALIZED_FIELDS:
            data[field] = listing.get(field)
    else:
        for field in LISTING_SERIALIZED_FIELDS:
            data[field] = getattr(listing, field, None)
    # Поле seller_url может отсутствовать в модели каталога.
    data.setdefault("seller_url", None)
    return data


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _collect_sellers(listings: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Собрать уникальных продавцов из валидных карточек."""

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


_CURRENT_WORKER: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "avito_worker_id", default=None
)


class _WorkerPageExchange:
    """Мультиплексор `PageRequest` по воркерам с использованием contextvars."""

    def __init__(self) -> None:
        self._request_queues: dict[int, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._response_queues: dict[int, asyncio.Queue] = defaultdict(asyncio.Queue)

    def _require_worker(self) -> int:
        worker_id = _CURRENT_WORKER.get()
        if worker_id is None:
            raise RuntimeError("worker context not initialized for page exchange")
        return worker_id

    async def request_page(self, payload: Any) -> Page:
        worker_id = self._require_worker()
        await self._request_queues[worker_id].put(payload)
        return await self._response_queues[worker_id].get()

    async def next_request(self) -> Any:
        worker_id = self._require_worker()
        return await self._request_queues[worker_id].get()

    def supply_page(self, page: Page) -> None:
        worker_id = self._require_worker()
        self._response_queues[worker_id].put_nowait(page)


set_page_exchange(_WorkerPageExchange())


class CatalogWorker:
    """Асинхронный воркер, обрабатывающий задачи очереди."""

    def __init__(self, *, worker_id: int, queue: CatalogTaskQueue, proxy_pool: ProxyPool) -> None:
        self.worker_id = worker_id
        self.queue = queue
        self.proxy_pool = proxy_pool

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._proxy: Optional[ProxyEndpoint] = None
        self._current_task: Optional[CatalogTask] = None
        self._supplier_task: Optional[asyncio.Task] = None

    async def run(self) -> None:
        log_event("worker_start", worker_id=self.worker_id)
        async with async_playwright() as playwright:
            self._playwright = playwright
            try:
                await self._run_loop()
            finally:
                await self._cleanup()

    async def _run_loop(self) -> None:
        while True:
            task = await self.queue.get()
            if task is None:
                pending = await self.queue.pending_count()
                if pending == 0:
                    log_event("worker_finished", worker_id=self.worker_id)
                    break
                await asyncio.sleep(0.5)
                continue
            self._current_task = task
            try:
                await self._process_task(task)
            finally:
                self._current_task = None

    async def _process_task(self, task: CatalogTask) -> None:
        payload = _task_payload(task)
        url = str(payload.get("url", ""))
        if not url:
            log_event("task_failed", worker_id=self.worker_id, item_id=task.task_key, reason="missing_url", attempt=task.attempt)
            await self.queue.mark_done(task.task_key)
            return

        log_event("task_start", worker_id=self.worker_id, item_id=task.task_key, attempt=task.attempt)

        try:
            await self._ensure_session(reason="bootstrap")
            state = await self._prepare_catalog_page(url)
            if state != "catalog_page_detector":
                status_mapping = {
                    "captcha_geetest_detector": CatalogParseStatus.CAPTCHA_UNSOLVED,
                    "proxy_block_429_detector": CatalogParseStatus.RATE_LIMIT,
                    "continue_button_detector": CatalogParseStatus.CAPTCHA_UNSOLVED,
                    "proxy_block_403_detector": CatalogParseStatus.PROXY_BLOCKED,
                    "proxy_auth_407_detector": CatalogParseStatus.PROXY_AUTH_REQUIRED,
                }
                mapped_status = status_mapping.get(state or "")
                await self._retry_task(
                    task,
                    status=mapped_status,
                    reason=state or "unexpected_state",
                )
                return

            token = _CURRENT_WORKER.set(self.worker_id)
            self._supplier_task = asyncio.create_task(self._page_supplier(url))
            try:
                listings, meta = await parse_catalog_until_complete(
                    self._page,  # type: ignore[arg-type]
                    url,
                    fields=CATALOG_FIELDS,
                    max_pages=config.CATALOG_MAX_PAGES,
                    start_page=1,
                )
            finally:
                if self._supplier_task:
                    self._supplier_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._supplier_task
                    self._supplier_task = None
                _CURRENT_WORKER.reset(token)

            article = extract_article_from_url(url)
            listing_payloads = [_listing_to_dict(listing) for listing in listings]

            validation = await validate_catalog(
                article=article,
                listings=listing_payloads,
                source_url=url,
            )
            valid_listings = validation.valid_listings
            sellers = _collect_sellers(valid_listings)

            log_event(
                "validation_summary",
                worker_id=self.worker_id,
                item_id=task.task_key,
                article=article or None,
                listings_total=validation.total,
                listings_valid=len(valid_listings),
                sellers=len(sellers),
                skipped_prefilter=validation.skipped_prefilter,
                invalid_sellers=len(validation.invalid_sellers),
                model_invalid=validation.model_invalid,
                model_missing=validation.model_missing,
                model_used=validation.model_used,
            )

            result_path = _result_path(payload)
            if sellers:
                await _ensure_dir(result_path)
                data = {
                    "source_url": url,
                    "article": article,
                    "is_complete": meta.status is CatalogParseStatus.SUCCESS if meta else False,
                    "valid_listing_count": len(valid_listings),
                    "sellers": sellers,
                    "meta": _meta_to_dict(meta),
                }
                await asyncio.to_thread(result_path.write_text, json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

            status = meta.status if meta else None
            status_value = status.value if isinstance(status, CatalogParseStatus) else status

            if status in SUCCESS_STATUSES:
                event = "task_success_empty" if status is CatalogParseStatus.EMPTY or not sellers else "task_success"
                log_event(
                    event,
                    worker_id=self.worker_id,
                    item_id=task.task_key,
                    sellers=len(sellers),
                    valid_listings=len(valid_listings),
                    proxy=self._proxy.address if self._proxy else None,
                    attempt=task.attempt,
                )
                await self.queue.mark_done(task.task_key)
                return

            retryable = status in RETRYABLE_STATUSES or status is None
            if not retryable:
                log_event(
                    "task_failed",
                    worker_id=self.worker_id,
                    item_id=task.task_key,
                    reason=status_value or "unhandled_status",
                    proxy=self._proxy.address if self._proxy else None,
                    attempt=task.attempt,
                )
                await self.queue.mark_done(task.task_key)
                return

            if sellers:
                log_event(
                    "task_partial",
                    worker_id=self.worker_id,
                    item_id=task.task_key,
                    sellers=len(sellers),
                    valid_listings=len(valid_listings),
                    proxy=self._proxy.address if self._proxy else None,
                    status=status_value,
                    attempt=task.attempt,
                )
            else:
                log_event(
                    "task_failed",
                    worker_id=self.worker_id,
                    item_id=task.task_key,
                    reason=status_value or "unknown_status",
                    valid_listings=len(valid_listings),
                    proxy=self._proxy.address if self._proxy else None,
                    attempt=task.attempt,
                )
            await self._retry_task(task, status=status, reason=status_value or "unknown_status")
        except Exception as exc:  # pylint: disable=broad-except
            log_event(
                "task_failed",
                worker_id=self.worker_id,
                item_id=task.task_key,
                reason=exc.__class__.__name__,
                attempt=task.attempt,
            )
            await self._retry_task(task, status=None, reason=exc.__class__.__name__)
        finally:
            await self._reset_session()

    async def _ensure_session(self, *, fresh_page: bool = False, reason: Optional[str] = None) -> None:
        if not self._playwright:
            raise RuntimeError("Playwright environment is not initialized")
        if self._browser is None or self._context is None or self._proxy is None:
            proxy = await self._acquire_proxy()
            args = proxy.as_playwright_arguments()
            self._browser = await self._playwright.chromium.launch(
                headless=config.PLAYWRIGHT_HEADLESS,
                proxy=args,
            )
            self._context = await self._browser.new_context()
            self._proxy = proxy
            fresh_page = True
        if fresh_page or self._page is None:
            if self._page:
                await self._page.close()
            self._page = await self._context.new_page()
            log_event(
                "worker_page_ready",
                worker_id=self.worker_id,
                proxy=self._proxy.address if self._proxy else None,
                reason=reason or ("refresh" if fresh_page else "reuse"),
            )

    async def _prepare_catalog_page(self, url: str) -> Optional[str]:
        await self._page.goto(url, wait_until="domcontentloaded")  # type: ignore[union-attr]

        # await asyncio.sleep(100)
      
        state = await detect_page_state(self._page)  # type: ignore[arg-type]
        while state in {
            "captcha_geetest_detector",
            "proxy_block_429_detector",
            "continue_button_detector",
        }:
            await resolve_captcha_flow(self._page)  # type: ignore[arg-type]
            state = await detect_page_state(self._page)  # type: ignore[arg-type]
            if state == "catalog_page_detector":
                log_event(
                    "captcha_resolved",
                    worker_id=self.worker_id,
                    item_id=self._current_task.task_key if self._current_task else None,
                    attempt=self._current_task.attempt if self._current_task else None,
                )
            else:
                log_event(
                    "captcha_failed",
                    worker_id=self.worker_id,
                    item_id=self._current_task.task_key if self._current_task else None,
                    state=state,
                    attempt=self._current_task.attempt if self._current_task else None,
                )
                return state
        return state

    async def _reset_session(self) -> None:
        await self._close_resources()
        if self._proxy:
            await self.proxy_pool.release(self._proxy.address)
        self._page = None
        self._context = None
        self._browser = None
        self._proxy = None

    async def _cleanup(self) -> None:
        await self._reset_session()

    async def _close_resources(self) -> None:
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()

    async def _retry_task(
        self,
        task: CatalogTask,
        *,
        status: Optional[CatalogParseStatus],
        reason: str,
    ) -> None:
        proxy_addr = self._proxy.address if self._proxy else None
        if status in BLOCKING_PROXY_STATUSES and proxy_addr:
            await self.proxy_pool.mark_blocked(proxy_addr, reason=BLOCK_REASON.get(status, reason))
        await self._close_resources()
        if status in DELAY_STATUSES and config.RETRY_DELAY_SEC > 0:
            await asyncio.sleep(config.RETRY_DELAY_SEC)
        requeued = await self.queue.retry(task.task_key, last_proxy=proxy_addr)
        self._set_waiting(False)
        log_event(
            "task_retry",
            worker_id=self.worker_id,
            item_id=task.task_key,
            reason=reason,
            status=status.value if isinstance(status, CatalogParseStatus) else status,
            requeued=requeued,
            proxy=proxy_addr,
            attempt=task.attempt,
        )
        if not requeued:
            log_event(
                "task_failed",
                worker_id=self.worker_id,
                item_id=task.task_key,
                reason="attempt_limit",
                proxy=proxy_addr,
                attempt=task.attempt,
            )

    async def _acquire_proxy(self) -> ProxyEndpoint:
        while True:
            proxy = await self.proxy_pool.acquire()
            if proxy:
                self._set_waiting(False)
                return proxy
            self._set_waiting(True)
            paused = await self.queue.pause(reason=f"no_proxy_available worker_id={self.worker_id}")
            await self.proxy_pool.wait_for_unblocked()
            if paused:
                await self.queue.resume(reason=f"proxy_available worker_id={self.worker_id}")
            self._set_waiting(False)

    def _set_waiting(self, value: bool) -> None:
        if isinstance(self._current_task, CatalogTask):
            self._current_task.set_wait_until_resumed(value)

    async def _page_supplier(self, url: str) -> None:
        while True:
            try:
                request = await wait_for_page_request()
            except asyncio.CancelledError:
                break
            status = getattr(request, "status", None)
            status_label = status.value if isinstance(status, CatalogParseStatus) else status
            await self._ensure_session(fresh_page=True, reason=f"retry:{status_label}")
            state = await self._prepare_catalog_page(url)
            if state != "catalog_page_detector":
                log_event(
                    "page_supply_state_mismatch",
                    worker_id=self.worker_id,
                    item_id=self._current_task.task_key if self._current_task else None,
                    state=state,
                    status=status_label,
                )
            supply_page(self._page)  # type: ignore[arg-type]
