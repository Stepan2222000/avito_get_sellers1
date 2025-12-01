from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from typing import Literal
except ImportError:  # pragma: no cover
    Literal = str  # type: ignore[misc]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@dataclass(slots=True)
class PathSettings:
    base_dir: Path = BASE_DIR
    urls_file: Path = field(default_factory=lambda: DATA_DIR / "urls.txt")
    proxies_file: Path = field(default_factory=lambda: DATA_DIR / "proxies.txt")
    seen_sellers_file: Path = field(default_factory=lambda: DATA_DIR / "seen_sellers.txt")
    valid_sellers_file: Path = field(default_factory=lambda: DATA_DIR / "valid_sellers.txt")
    blocked_proxies_file: Path = field(default_factory=lambda: BASE_DIR / "results" / "blocked_proxies.txt")
    results_dir: Path = field(default_factory=lambda: BASE_DIR / "results")
    details_dir: Path = field(default_factory=lambda: BASE_DIR / "results" / "details")
    jsonl_file: Path = field(default_factory=lambda: BASE_DIR / "results" / "seller_validation_results.jsonl")
    validated_urls_file: Path = field(default_factory=lambda: BASE_DIR / "results" / "validated_seller_urls.txt")


@dataclass(slots=True)
class QueueSettings:
    max_attempts: int = 4
    idle_sleep_seconds: float = 1.0
    retry_base_delay_seconds: float = 15.0
    retry_multiplier: float = 1.7
    retry_cap_seconds: float = 180.0


@dataclass(slots=True)
class PlaywrightSettings:
    headless: bool = False
    navigation_timeout_ms: int = 35000
    request_timeout_ms: int = 30000
    captcha_retry_limit: int = 3
    goto_wait_until: str = "domcontentloaded"


@dataclass(slots=True)
class GeminiSettings:
    api_key: str = ""
    model: str = "gemini-2.5-flash"
    concurrency: int = 10
    temperature: float = 0.1
    max_output_tokens: Optional[int] = None
    request_timeout: float = 90.0
    max_retries: int = 3
    retry_delay_seconds: float = 3.0


@dataclass(slots=True)
class OpenAISettings:
    api_key: Optional[str] = None
    model: Optional[str] = None
    request_timeout: float = 90.0
    max_retries: int = 3
    retry_delay_seconds: float = 3.0


@dataclass(slots=True)
class DeepInfraSettings:
    api_key: Optional[str] = None
    base_url: str = "https://api.deepinfra.com/v1/openai"
    model: str = "deepseek-ai/DeepSeek-V3.1-Terminus"
    reasoning_enabled: bool = True
    reasoning_effort: str = "high"
    request_timeout: float = 290.0
    max_retries: int = 3
    retry_delay_seconds: float = 3.0


@dataclass(slots=True)
class ProviderSettings:
    name: Literal["gemini", "openai", "deepinfra"] = "deepinfra"


@dataclass(slots=True)
class ValidationSettings:
    worker_count: int = 15
    source_name: str = "urls_file"
    max_items_per_seller: int = 100
    max_pages_per_seller: int = 5
    min_price_filter: Optional[int] = None


@dataclass(slots=True)
class ValidationExecutorSettings:
    concurrency: int = 30
    request_timeout: float = 500.0
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    min_reviews_for_valid_export: int = 40


@dataclass(slots=True)
class LoggingSettings:
    level: str = "INFO"
    json_logs: bool = False


@dataclass(slots=True)
class Config:
    paths: PathSettings = field(default_factory=PathSettings)
    queue: QueueSettings = field(default_factory=QueueSettings)
    playwright: PlaywrightSettings = field(default_factory=PlaywrightSettings)
    gemini: GeminiSettings = field(default_factory=GeminiSettings)
    openai: OpenAISettings = field(default_factory=OpenAISettings)
    deepinfra: DeepInfraSettings = field(default_factory=DeepInfraSettings)
    provider: ProviderSettings = field(default_factory=ProviderSettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    validation_executor: ValidationExecutorSettings = field(default_factory=ValidationExecutorSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


CONFIG = Config()

PATHS = CONFIG.paths
QUEUE = CONFIG.queue
PLAYWRIGHT = CONFIG.playwright
GEMINI = CONFIG.gemini
OPENAI = CONFIG.openai
PROVIDER = CONFIG.provider
VALIDATION = CONFIG.validation
VALIDATION_EXECUTOR = CONFIG.validation_executor
LOGGING = CONFIG.logging

RESULTS_DIR = PATHS.results_dir
LINKS_FILE = PATHS.urls_file
PROXIES_FILE = PATHS.proxies_file
BLOCKED_PROXIES_FILE = PATHS.blocked_proxies_file
SEEN_SELLERS_FILE = PATHS.seen_sellers_file
VALID_SELLERS_FILE = PATHS.valid_sellers_file

MAX_PROXY_ATTEMPTS = QUEUE.max_attempts
RETRY_DELAY_SEC = QUEUE.retry_base_delay_seconds

WORKER_COUNT = VALIDATION.worker_count
CATALOG_MAX_PAGES = VALIDATION.max_pages_per_seller

PLAYWRIGHT_HEADLESS = PLAYWRIGHT.headless

GENAI_API_KEY = GEMINI.api_key
GENAI_MODEL = GEMINI.model
GENAI_CONCURRENCY = GEMINI.concurrency
GENAI_REQUEST_TIMEOUT_SEC = GEMINI.request_timeout
GENAI_MAX_RETRIES = GEMINI.max_retries
GENAI_RETRY_DELAY_SEC = GEMINI.retry_delay_seconds

OPENAI_API_KEY = OPENAI.api_key
OPENAI_MODEL = OPENAI.model
OPENAI_REQUEST_TIMEOUT_SEC = OPENAI.request_timeout
OPENAI_MAX_RETRIES = OPENAI.max_retries
OPENAI_RETRY_DELAY_SEC = OPENAI.retry_delay_seconds

DEEPINFRA = CONFIG.deepinfra

DEEPINFRA_API_KEY = DEEPINFRA.api_key
DEEPINFRA_BASE_URL = DEEPINFRA.base_url
DEEPINFRA_MODEL = DEEPINFRA.model
DEEPINFRA_REASONING_ENABLED = DEEPINFRA.reasoning_enabled
DEEPINFRA_REASONING_EFFORT = DEEPINFRA.reasoning_effort
DEEPINFRA_REQUEST_TIMEOUT_SEC = DEEPINFRA.request_timeout
DEEPINFRA_MAX_RETRIES = DEEPINFRA.max_retries
DEEPINFRA_RETRY_DELAY_SEC = DEEPINFRA.retry_delay_seconds

PROVIDER_NAME = PROVIDER.name
USE_OPENAI = PROVIDER_NAME == "openai"
USE_DEEPINFRA = PROVIDER_NAME == "deepinfra"

LOG_LEVEL = LOGGING.level

VALIDATION_CONCURRENCY = VALIDATION_EXECUTOR.concurrency
VALIDATION_REQUEST_TIMEOUT_SEC = VALIDATION_EXECUTOR.request_timeout
VALIDATION_MAX_RETRIES = VALIDATION_EXECUTOR.max_retries
VALIDATION_RETRY_DELAY_SEC = VALIDATION_EXECUTOR.retry_delay_seconds
VALID_SELLER_REVIEWS_THRESHOLD = VALIDATION_EXECUTOR.min_reviews_for_valid_export

__all__ = [
    "CONFIG",
    "Config",
    "PathSettings",
    "QueueSettings",
    "PlaywrightSettings",
    "GeminiSettings",
    "ValidationSettings",
    "ValidationExecutorSettings",
    "LoggingSettings",
    "DATA_DIR",
    "RESULTS_DIR",
    "LINKS_FILE",
    "PROXIES_FILE",
    "BLOCKED_PROXIES_FILE",
    "SEEN_SELLERS_FILE",
    "VALID_SELLERS_FILE",
    "MAX_PROXY_ATTEMPTS",
    "RETRY_DELAY_SEC",
    "WORKER_COUNT",
    "CATALOG_MAX_PAGES",
    "PLAYWRIGHT_HEADLESS",
    "GENAI_API_KEY",
    "GENAI_MODEL",
    "GENAI_CONCURRENCY",
    "GENAI_REQUEST_TIMEOUT_SEC",
    "GENAI_MAX_RETRIES",
    "GENAI_RETRY_DELAY_SEC",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_REQUEST_TIMEOUT_SEC",
    "OPENAI_MAX_RETRIES",
    "OPENAI_RETRY_DELAY_SEC",
    "DEEPINFRA",
    "DEEPINFRA_API_KEY",
    "DEEPINFRA_BASE_URL",
    "DEEPINFRA_MODEL",
    "DEEPINFRA_REASONING_ENABLED",
    "DEEPINFRA_REASONING_EFFORT",
    "DEEPINFRA_REQUEST_TIMEOUT_SEC",
    "DEEPINFRA_MAX_RETRIES",
    "DEEPINFRA_RETRY_DELAY_SEC",
    "PROVIDER_NAME",
    "USE_OPENAI",
    "USE_DEEPINFRA",
    "LOG_LEVEL",
    "VALIDATION_EXECUTOR",
    "VALIDATION_CONCURRENCY",
    "VALIDATION_REQUEST_TIMEOUT_SEC",
    "VALIDATION_MAX_RETRIES",
    "VALIDATION_RETRY_DELAY_SEC",
    "VALID_SELLER_REVIEWS_THRESHOLD",
]
