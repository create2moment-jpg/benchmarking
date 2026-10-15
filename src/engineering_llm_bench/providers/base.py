"""Provider adapter protocol and shared types.

A provider turns a list of chat messages into one completion. Provider-specific logic (wire
formats, auth, error mapping) lives entirely in adapter modules; the benchmark core only sees
``Provider``, ``CompletionRequest``, ``CompletionResult`` and ``ProviderError``.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from engineering_llm_bench.config.models import ModelConfig
from engineering_llm_bench.control import CancelToken

Role = Literal["system", "user", "assistant"]
ErrorKind = Literal[
    "auth", "rate_limit", "server", "client", "network", "timeout", "malformed", "config", "refusal"
]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    """A provider-neutral completion request.

    ``metadata`` carries local-only hints (task id, mode, task directory) used by the fake
    provider and for logging. Real adapters must never send it to an endpoint.
    """

    messages: tuple[ChatMessage, ...]
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()
    timeout_s: float = 120.0
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    """Provider-reported token usage. ``None`` means not reported (never estimated)."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    def __add__(self, other: Usage) -> Usage:
        def add(a: int | None, b: int | None) -> int | None:
            return None if a is None and b is None else (a or 0) + (b or 0)

        return Usage(
            add(self.input_tokens, other.input_tokens),
            add(self.output_tokens, other.output_tokens),
        )


@dataclass(frozen=True)
class CompletionResult:
    text: str
    usage: Usage
    finish_reason: str | None
    latency_s: float
    model_reported: str | None = None
    request_id: str | None = None


@dataclass(frozen=True)
class ProviderCapabilities:
    streaming: bool = False
    usage_reporting: bool = False
    seed: bool = False


@dataclass(frozen=True)
class ProviderInfo:
    """Metadata recorded with every run. ``model_id`` is the provider's ID; ``alias`` the label."""

    alias: str
    provider: str
    model_id: str
    endpoint_type: str
    remote: bool
    capabilities: ProviderCapabilities
    serving_runtime: str | None = None
    quantization: str | None = None
    context_limit: int | None = None


class ProviderError(Exception):
    """A provider call failed. ``retryable`` errors may be retried with backoff."""

    def __init__(self, kind: ErrorKind, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.kind: ErrorKind = kind
        self.retryable = retryable


class Provider(Protocol):
    """Model adapter protocol. Implementations must be thread-safe for concurrent attempts."""

    @property
    def info(self) -> ProviderInfo: ...

    def complete(self, request: CompletionRequest, cancel: CancelToken) -> CompletionResult:
        """Return one completion. Raise ``ProviderError`` (or ``CancelledError``)."""
        ...

    def close(self) -> None: ...


def info_from_config(cfg: ModelConfig, capabilities: ProviderCapabilities) -> ProviderInfo:
    return ProviderInfo(
        alias=cfg.alias,
        provider=cfg.provider,
        model_id=cfg.model_id,
        endpoint_type=cfg.endpoint_type,
        remote=cfg.remote,
        capabilities=capabilities,
        serving_runtime=cfg.serving_runtime,
        quantization=cfg.quantization,
        context_limit=cfg.context_limit,
    )


class RateLimiter:
    """Bounded concurrency plus an optional requests-per-minute limit for one provider."""

    def __init__(self, max_concurrency: int, requests_per_minute: int | None) -> None:
        self._sem = threading.BoundedSemaphore(max_concurrency)
        self._interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self, cancel: CancelToken) -> None:
        while not self._sem.acquire(timeout=0.1):
            cancel.raise_if_cancelled()
        if self._interval:
            with self._lock:
                now = time.monotonic()
                wait = max(0.0, self._next_slot - now)
                self._next_slot = max(now, self._next_slot) + self._interval
            _sleep_cancellable(wait, cancel)

    def release(self) -> None:
        self._sem.release()


def _sleep_cancellable(seconds: float, cancel: CancelToken) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        cancel.raise_if_cancelled()
        time.sleep(min(0.05, max(0.0, end - time.monotonic())))


class ManagedProvider:
    """Wraps a provider with rate limiting and retry-with-backoff for retryable errors."""

    def __init__(
        self,
        inner: Provider,
        max_retries: int,
        limiter: RateLimiter,
        backoff_base_s: float = 1.0,
    ) -> None:
        self.inner = inner
        self.max_retries = max_retries
        self.limiter = limiter
        self.backoff_base_s = backoff_base_s
        self.retries_used = 0

    @property
    def info(self) -> ProviderInfo:
        return self.inner.info

    def complete(self, request: CompletionRequest, cancel: CancelToken) -> CompletionResult:
        attempt = 0
        while True:
            cancel.raise_if_cancelled()
            self.limiter.acquire(cancel)
            try:
                return self.inner.complete(request, cancel)
            except ProviderError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
            finally:
                self.limiter.release()
            attempt += 1
            self.retries_used += 1
            delay = self.backoff_base_s * 2.0 ** (attempt - 1) * (0.5 + random.random())  # noqa: S311
            _sleep_cancellable(min(30.0, delay), cancel)

    def close(self) -> None:
        self.inner.close()
