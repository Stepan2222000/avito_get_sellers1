"""Фоновый исполнитель запросов к LLM-валидации."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from avito_library.parsers.catalog_parser import CatalogParseStatus

import config
from src.log import log_event
from src.proxy_pool import ProxyPool
from src.queue import CatalogTaskQueue
from src.result_utils import collect_sellers, ensure_parent_dir, meta_to_dict
from src.validator_runner import ValidationOutcome, validate_catalog


@dataclass(slots=True)
class ValidationJob:
    task_key: str
    attempt: int
    url: str
    article: str
    listings: list[dict[str, Any]]
    meta: Any
    status: Optional[CatalogParseStatus]
    status_label: Optional[str]
    result_path: Path
    proxy_address: Optional[str]
    worker_id: int
    retryable: bool
    success_status: bool
    delay_on_retry: bool
    blocking_status: Optional[CatalogParseStatus]
    blocking_reason: Optional[str]


class ValidationExecutor:
    """Очередь, ограничивающая конкурентные запросы к LLM."""

    _STOP = object()

    def __init__(
        self,
        *,
        queue: CatalogTaskQueue,
        proxy_pool: ProxyPool,
        concurrency: int,
        request_timeout: float,
        max_retries: int,
        retry_delay: float,
    ) -> None:
        self._queue = queue
        self._proxy_pool = proxy_pool
        self._request_timeout = request_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._job_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._started = False
        self._closed = False
        self._concurrency = max(1, concurrency)
        self._active_jobs = 0
        self._active_lock = asyncio.Lock()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for index in range(self._concurrency):
            task = asyncio.create_task(self._worker_loop(index), name=f"validation-executor-{index}")
            self._workers.append(task)

    async def submit(self, job: ValidationJob) -> None:
        if self._closed:
            raise RuntimeError("validation executor is closed")
        await self._job_queue.put(job)
        log_event(
            "validation_job_queued",
            item_id=job.task_key,
            worker_id=job.worker_id,
            listings=len(job.listings),
        )

    async def drain(self) -> None:
        await self._job_queue.join()

    async def has_pending_jobs(self) -> bool:
        async with self._active_lock:
            return self._active_jobs > 0 or not self._job_queue.empty()

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._job_queue.join()
        for _ in self._workers:
            await self._job_queue.put(self._STOP)
        await asyncio.gather(*self._workers, return_exceptions=True)

    async def _worker_loop(self, index: int) -> None:
        while True:
            job = await self._job_queue.get()
            if job is self._STOP:
                self._job_queue.task_done()
                break
            async with self._active_lock:
                self._active_jobs += 1
            try:
                await self._process_job(job)
            finally:
                async with self._active_lock:
                    self._active_jobs -= 1
                self._job_queue.task_done()

    async def _process_job(self, job: ValidationJob) -> None:
        outcome = await self._attempt_validation(job)
        if outcome is None:
            await self._handle_validation_failure(job)
            return
        await self._handle_validation_success(job, outcome)

    async def _attempt_validation(self, job: ValidationJob) -> Optional[ValidationOutcome]:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                log_event(
                    "validation_job_start",
                    item_id=job.task_key,
                    worker_id=job.worker_id,
                    attempt=job.attempt,
                    validation_attempt=attempt,
                )
                return await asyncio.wait_for(
                    validate_catalog(
                        article=job.article,
                        listings=job.listings,
                        source_url=job.url,
                    ),
                    timeout=self._request_timeout,
                )
            except asyncio.TimeoutError as exc:
                last_exc = exc
                log_event(
                    "validation_job_retry",
                    item_id=job.task_key,
                    worker_id=job.worker_id,
                    attempt=job.attempt,
                    validation_attempt=attempt,
                    reason="timeout",
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log_event(
                    "validation_job_retry",
                    item_id=job.task_key,
                    worker_id=job.worker_id,
                    attempt=job.attempt,
                    validation_attempt=attempt,
                    reason=exc.__class__.__name__,
                )
            if attempt < self._max_retries and self._retry_delay > 0:
                await asyncio.sleep(self._retry_delay)
        if last_exc:
            log_event(
                "validation_job_failed",
                item_id=job.task_key,
                worker_id=job.worker_id,
                attempt=job.attempt,
                reason=last_exc.__class__.__name__,
                message=str(last_exc),
            )
        else:
            log_event(
                "validation_job_failed",
                item_id=job.task_key,
                worker_id=job.worker_id,
                attempt=job.attempt,
                reason="unknown",
            )
        return None

    async def _handle_validation_success(self, job: ValidationJob, outcome: ValidationOutcome) -> None:
        valid_listings = outcome.valid_listings
        sellers = collect_sellers(valid_listings)

        log_event(
            "validation_summary",
            worker_id=job.worker_id,
            item_id=job.task_key,
            article=job.article or None,
            listings_total=outcome.total,
            listings_valid=len(valid_listings),
            sellers=len(sellers),
            skipped_prefilter=outcome.skipped_prefilter,
            invalid_sellers=len(outcome.invalid_sellers),
            model_invalid=outcome.model_invalid,
            model_missing=outcome.model_missing,
            model_used=outcome.model_used,
        )

        if sellers:
            await ensure_parent_dir(job.result_path)
            data = {
                "source_url": job.url,
                "article": job.article,
                "is_complete": bool(
                    getattr(job.meta, "status", None) is CatalogParseStatus.SUCCESS
                ),
                "valid_listing_count": len(valid_listings),
                "sellers": sellers,
                "meta": meta_to_dict(job.meta),
            }
            await asyncio.to_thread(
                job.result_path.write_text,
                json.dumps(data, ensure_ascii=False, indent=2),
                "utf-8",
            )

        status_value = job.status.value if isinstance(job.status, CatalogParseStatus) else job.status_label

        if job.success_status:
            event = "task_success_empty" if not sellers or job.status is CatalogParseStatus.EMPTY else "task_success"
            log_event(
                event,
                worker_id=job.worker_id,
                item_id=job.task_key,
                sellers=len(sellers),
                valid_listings=len(valid_listings),
                proxy=job.proxy_address,
                attempt=job.attempt,
            )
            await self._queue.mark_done(job.task_key)
            return

        if not job.retryable:
            log_event(
                "task_failed",
                worker_id=job.worker_id,
                item_id=job.task_key,
                reason=status_value or "unhandled_status",
                proxy=job.proxy_address,
                attempt=job.attempt,
            )
            await self._queue.mark_done(job.task_key)
            return

        if sellers:
            log_event(
                "task_partial",
                worker_id=job.worker_id,
                item_id=job.task_key,
                sellers=len(sellers),
                valid_listings=len(valid_listings),
                proxy=job.proxy_address,
                status=status_value,
                attempt=job.attempt,
            )
        else:
            log_event(
                "task_failed",
                worker_id=job.worker_id,
                item_id=job.task_key,
                reason=status_value or "unknown_status",
                valid_listings=len(valid_listings),
                proxy=job.proxy_address,
                attempt=job.attempt,
            )

        await self._retry_job(job, status=job.status, reason=status_value or "unknown_status")

    async def _handle_validation_failure(self, job: ValidationJob) -> None:
        await self._retry_job(job, status=None, reason="validation_failed")

    async def _retry_job(
        self,
        job: ValidationJob,
        *,
        status: Optional[CatalogParseStatus],
        reason: str,
    ) -> None:
        proxy_addr = job.proxy_address
        if job.blocking_status and proxy_addr:
            await self._proxy_pool.mark_blocked(proxy_addr, reason=job.blocking_reason or reason)
        if job.delay_on_retry and config.RETRY_DELAY_SEC > 0:
            await asyncio.sleep(config.RETRY_DELAY_SEC)
        requeued = await self._queue.retry(job.task_key, last_proxy=proxy_addr)
        log_event(
            "task_retry",
            worker_id=job.worker_id,
            item_id=job.task_key,
            reason=reason,
            status=status.value if isinstance(status, CatalogParseStatus) else status,
            requeued=requeued,
            proxy=proxy_addr,
            attempt=job.attempt,
        )
        if not requeued:
            log_event(
                "task_failed",
                worker_id=job.worker_id,
                item_id=job.task_key,
                reason="attempt_limit",
                proxy=proxy_addr,
                attempt=job.attempt,
            )


__all__ = ["ValidationExecutor", "ValidationJob"]
