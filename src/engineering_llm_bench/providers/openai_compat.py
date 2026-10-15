"""OpenAI-compatible ``/chat/completions`` adapter (Ollama, vLLM, LM Studio, llama.cpp, OpenAI).

Only the documented, widely implemented subset of the Chat Completions API is used: ``model``,
``messages``, ``max_tokens``, ``temperature``, ``top_p``, ``seed``, ``stop`` and ``stream``.
Credentials come from the environment variable named by ``api_key_env``; custom header values
may reference environment variables as ``${NAME}``.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator
from typing import Any

import httpx

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

_ENV_REF = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def expand_env(value: str) -> str:
    """Replace ``${NAME}`` with the environment value; missing variables are config errors."""

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ProviderError("config", f"environment variable {name} is not set")
        return os.environ[name]

    return _ENV_REF.sub(repl, value)


def build_headers(cfg: ModelConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if cfg.api_key_env:
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise ProviderError(
                "auth", f"environment variable {cfg.api_key_env} is not set (model {cfg.alias})"
            )
        headers["Authorization"] = f"Bearer {key}"
    for name, value in cfg.headers.items():
        headers[name] = expand_env(value)
    return headers


def _map_status(status: int, body: str) -> ProviderError:
    snippet = body[:300]
    if status in (401, 403):
        return ProviderError("auth", f"HTTP {status}: authentication failed")
    if status == 429:
        return ProviderError("rate_limit", f"HTTP 429: {snippet}", retryable=True)
    if status in (408, 409) or status >= 500:
        return ProviderError("server", f"HTTP {status}: {snippet}", retryable=True)
    return ProviderError("client", f"HTTP {status}: {snippet}")


class OpenAICompatibleProvider:
    """Adapter for any endpoint implementing OpenAI Chat Completions."""

    def __init__(self, cfg: ModelConfig, transport: httpx.BaseTransport | None = None) -> None:
        if not cfg.base_url:
            raise ProviderError("config", "base_url is required")
        self.cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.base_url.rstrip("/") + "/", transport=transport, trust_env=cfg.remote
        )
        self._info = info_from_config(
            cfg, ProviderCapabilities(streaming=True, usage_reporting=True, seed=True)
        )

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def _payload(self, request: CompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.cfg.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens,
            "stream": self.cfg.stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop:
            payload["stop"] = list(request.stop)
        if self.cfg.stream:
            payload["stream_options"] = {"include_usage": True}
        payload.update(self.cfg.extra_body)
        return payload

    def complete(self, request: CompletionRequest, cancel: CancelToken) -> CompletionResult:
        cancel.raise_if_cancelled()
        headers = build_headers(self.cfg)
        payload = self._payload(request)
        timeout = httpx.Timeout(request.timeout_s, connect=min(10.0, request.timeout_s))
        start = time.monotonic()
        try:
            if self.cfg.stream:
                return self._stream(payload, headers, timeout, cancel, start)
            response = self._client.post(
                "chat/completions", json=payload, headers=headers, timeout=timeout
            )
        except httpx.TimeoutException as exc:
            raise ProviderError("timeout", f"request timed out: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError("network", f"network error: {exc}", retryable=True) from exc
        if response.status_code != 200:
            raise _map_status(response.status_code, response.text)
        return self._parse(response, start)

    def _parse(self, response: httpx.Response, start: float) -> CompletionResult:
        try:
            data = response.json()
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError("malformed", f"unexpected response shape: {exc}") from exc
        if not isinstance(text, str):
            raise ProviderError("malformed", "message content is not a string")
        usage = data.get("usage") or {}
        return CompletionResult(
            text=text,
            usage=Usage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
            finish_reason=choice.get("finish_reason"),
            latency_s=time.monotonic() - start,
            model_reported=data.get("model"),
            request_id=data.get("id"),
        )

    def _stream(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: httpx.Timeout,
        cancel: CancelToken,
        start: float,
    ) -> CompletionResult:
        parts: list[str] = []
        finish: str | None = None
        usage = Usage()
        model: str | None = None
        with self._client.stream(
            "POST", "chat/completions", json=payload, headers=headers, timeout=timeout
        ) as response:
            if response.status_code != 200:
                response.read()
                raise _map_status(response.status_code, response.text)
            for event in _sse_events(response.iter_lines()):
                cancel.raise_if_cancelled()
                model = event.get("model", model)
                if event.get("usage"):
                    u = event["usage"]
                    usage = Usage(u.get("prompt_tokens"), u.get("completion_tokens"))
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if isinstance(delta.get("content"), str):
                        parts.append(delta["content"])
                    finish = choice.get("finish_reason") or finish
        return CompletionResult(
            "".join(parts), usage, finish, time.monotonic() - start, model_reported=model
        )

    def close(self) -> None:
        self._client.close()


def _sse_events(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ProviderError("malformed", f"invalid SSE chunk: {data[:200]}") from exc
        if isinstance(event, dict):
            yield event


def probe_endpoint(cfg: ModelConfig, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Check reachability via ``GET /models``. Sends no prompts or source code."""
    if not cfg.base_url:
        return False, "no base_url"
    try:
        headers = build_headers(cfg)
    except ProviderError as exc:
        return False, str(exc)
    try:
        with httpx.Client(trust_env=cfg.remote) as client:
            resp = client.get(
                cfg.base_url.rstrip("/") + "/models", headers=headers, timeout=timeout_s
            )
    except httpx.HTTPError as exc:
        return False, f"unreachable: {type(exc).__name__}"
    if resp.status_code == 200:
        return True, "reachable (HTTP 200)"
    return False, f"HTTP {resp.status_code}"
