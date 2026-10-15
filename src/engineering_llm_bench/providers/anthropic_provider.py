"""Anthropic Messages API adapter using the official ``anthropic`` Python SDK.

The SDK is an optional dependency (``pip install engineering-llm-bench[anthropic]``) and is
imported lazily, so the core benchmark works without it. Current Claude models reject sampling
parameters (``temperature``/``top_p``) and control depth through ``output_config.effort``; use
``extra_body`` in the model config for such provider-specific settings. Sampling values from the
config are therefore *not* sent; this is recorded in the run metadata via the config itself.
"""

from __future__ import annotations

import os
import time
from typing import Any

from engineering_llm_bench.config.models import ModelConfig
from engineering_llm_bench.control import CancelToken
from engineering_llm_bench.providers.base import (
    CompletionRequest,
    CompletionResult,
    ProviderCapabilities,
    ProviderError,
    ProviderInfo,
    Usage,
    info_from_config,
)


def _load_sdk() -> Any:
    try:
        import anthropic  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise ProviderError(
            "config",
            "the anthropic provider needs the official SDK: "
            "pip install 'engineering-llm-bench[anthropic]'",
        ) from exc
    return anthropic


class AnthropicProvider:
    """Adapter for Claude models. ``client`` may be injected for tests."""

    def __init__(self, cfg: ModelConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        self._client = client
        self._info = info_from_config(
            cfg, ProviderCapabilities(streaming=False, usage_reporting=True, seed=False)
        )

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def _get_client(self) -> Any:
        if self._client is None:
            sdk = _load_sdk()
            env = self.cfg.api_key_env or "ANTHROPIC_API_KEY"
            key = os.environ.get(env)
            kwargs: dict[str, Any] = {"max_retries": 0}  # retries handled by ManagedProvider
            if key:
                kwargs["api_key"] = key
            if self.cfg.base_url:
                kwargs["base_url"] = self.cfg.base_url
            try:
                self._client = sdk.Anthropic(**kwargs)
            except Exception as exc:  # SDK raises on missing credentials
                raise ProviderError("auth", f"cannot create Anthropic client: {exc}") from exc
        return self._client

    def complete(self, request: CompletionRequest, cancel: CancelToken) -> CompletionResult:
        cancel.raise_if_cancelled()
        client = self._get_client()
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        messages = [
            {"role": m.role, "content": m.content} for m in request.messages if m.role != "system"
        ]
        params: dict[str, Any] = {
            "model": self.cfg.model_id,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }
        if system:
            params["system"] = system
        if request.stop:
            params["stop_sequences"] = list(request.stop)
        params.update(self.cfg.extra_body)
        start = time.monotonic()
        try:
            response = client.with_options(timeout=request.timeout_s).messages.create(**params)
        except Exception as exc:
            raise _map_error(exc) from exc
        text = "".join(
            getattr(block, "text", "")
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            raise ProviderError("refusal", "model declined the request (stop_reason=refusal)")
        return CompletionResult(
            text=text,
            usage=Usage(
                getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)
            ),
            finish_reason=stop_reason,
            latency_s=time.monotonic() - start,
            model_reported=getattr(response, "model", None),
            request_id=getattr(response, "id", None),
        )

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()


def _map_error(exc: Exception) -> ProviderError:
    """Map SDK exceptions (most specific first) onto provider-neutral errors."""
    try:
        sdk = _load_sdk()
    except ProviderError:
        return ProviderError("client", str(exc))
    if isinstance(exc, sdk.APITimeoutError):
        return ProviderError("timeout", str(exc), retryable=True)
    if isinstance(exc, sdk.AuthenticationError | sdk.PermissionDeniedError):
        return ProviderError("auth", "authentication failed")
    if isinstance(exc, sdk.RateLimitError):
        return ProviderError("rate_limit", str(exc), retryable=True)
    if isinstance(exc, sdk.APIStatusError):
        status = getattr(exc, "status_code", 0) or 0
        return ProviderError(
            "server" if status >= 500 else "client", str(exc), retryable=status >= 500
        )
    if isinstance(exc, sdk.APIConnectionError):
        return ProviderError("network", str(exc), retryable=True)
    return ProviderError("client", f"{type(exc).__name__}: {exc}")
