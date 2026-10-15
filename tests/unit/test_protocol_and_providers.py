"""Harness wire protocol and provider adapters (OpenAI-compatible via mocked HTTP, Anthropic via
an injected fake SDK client)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from engineering_llm_bench.config.models import ModelConfig
from engineering_llm_bench.control import CancelledError, CancelToken
from engineering_llm_bench.harnesses.protocol import (
    MalformedOutputError,
    parse_action,
    parse_direct,
    render_direct,
)
from engineering_llm_bench.providers.anthropic_provider import AnthropicProvider
from engineering_llm_bench.providers.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResult,
    ManagedProvider,
    ProviderError,
    RateLimiter,
    Usage,
)
from engineering_llm_bench.providers.openai_compat import OpenAICompatibleProvider

REQ = CompletionRequest(
    messages=(ChatMessage("user", "hi"),),
    max_tokens=64,
    timeout_s=5,
    metadata={"task_dir": "/secret/path"},
)


# ----------------------------------------------------------------------------- protocol
def test_parse_action_plain_and_fenced() -> None:
    assert parse_action('{"action": "list_files"}').path == "."
    fenced = 'Plan: read it\n```json\n{"action": "read_file", "path": "a.py"}\n```'
    assert parse_action(fenced).path == "a.py"
    act = parse_action('{"action": "run_command", "argv": ["pytest", "-q"]}')
    assert act.argv == ("pytest", "-q")


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        '{"action": "delete_everything"}',
        '{"action": "run_command", "argv": "pytest -q"}',
        '{"action": "write_file", "path": "a.py"}',
        '{"action": "finish", "summary": 3}',
    ],
)
def test_parse_action_malformed(text: str) -> None:
    with pytest.raises(MalformedOutputError):
        parse_action(text)


def test_direct_round_trip() -> None:
    text = render_direct({"a.py": "x = 1\n", "b/c.py": "y\n"}, "done")
    parsed = parse_direct(text)
    assert parsed.files == {"a.py": "x = 1\n", "b/c.py": "y\n"} and parsed.summary == "done"
    with pytest.raises(MalformedOutputError):
        parse_direct("nothing useful")


# ----------------------------------------------------------------------------- openai-compatible
def cfg(**over: Any) -> ModelConfig:
    base: dict[str, Any] = {
        "alias": "local",
        "provider": "openai_compatible",
        "model_id": "m",
        "base_url": "http://localhost:9/v1",
    }
    base.update(over)
    return ModelConfig.model_validate(base)


def provider_with(handler: Any, **over: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(cfg(**over), transport=httpx.MockTransport(handler))


def ok_response(text: str = "hello") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "r1",
            "model": "m-reported",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


def test_openai_success_and_no_metadata_leak() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return ok_response()

    result = provider_with(handler).complete(REQ, CancelToken())
    assert result.text == "hello"
    assert result.usage == Usage(11, 7)
    assert result.model_reported == "m-reported"
    assert "metadata" not in seen["body"] and "/secret/path" not in json.dumps(seen["body"])
    assert seen["auth"] is None


def test_openai_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "abc123")
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return ok_response()

    provider_with(handler, api_key_env="MY_KEY").complete(REQ, CancelToken())
    assert seen["auth"] == "Bearer abc123"


def test_openai_missing_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(ProviderError, match="MISSING_KEY") as info:
        provider_with(lambda r: ok_response(), api_key_env="MISSING_KEY").complete(
            REQ, CancelToken()
        )
    assert info.value.kind == "auth"


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (401, "auth", False),
        (429, "rate_limit", True),
        (503, "server", True),
        (400, "client", False),
    ],
)
def test_openai_error_mapping(status: int, kind: str, retryable: bool) -> None:
    provider = provider_with(lambda r: httpx.Response(status, text="err"))
    with pytest.raises(ProviderError) as info:
        provider.complete(REQ, CancelToken())
    assert info.value.kind == kind and info.value.retryable == retryable


def test_openai_malformed_response() -> None:
    provider = provider_with(lambda r: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ProviderError) as info:
        provider.complete(REQ, CancelToken())
    assert info.value.kind == "malformed"


def test_openai_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError) as info:
        provider_with(handler).complete(REQ, CancelToken())
    assert info.value.kind == "timeout"


def test_openai_streaming() -> None:
    chunks = [
        {"choices": [{"delta": {"content": "hel"}}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    result = provider_with(handler, stream=True).complete(REQ, CancelToken())
    assert result.text == "hello" and result.usage == Usage(3, 2) and result.finish_reason == "stop"


def test_cancelled_before_request() -> None:
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        provider_with(lambda r: ok_response()).complete(REQ, token)


# ----------------------------------------------------------------------------- retries
class Flaky:
    def __init__(self, failures: int, retryable: bool = True) -> None:
        self.failures = failures
        self.retryable = retryable
        self.calls = 0
        self.info = OpenAICompatibleProvider(cfg()).info

    def complete(self, request: CompletionRequest, cancel: CancelToken) -> CompletionResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError("server", "boom", retryable=self.retryable)
        return CompletionResult("ok", Usage(), "stop", 0.0)

    def close(self) -> None:
        return None


def test_managed_provider_retries_then_succeeds() -> None:
    inner = Flaky(2)
    managed = ManagedProvider(inner, 3, RateLimiter(1, None), backoff_base_s=0.001)
    assert managed.complete(REQ, CancelToken()).text == "ok"
    assert inner.calls == 3 and managed.retries_used == 2


def test_managed_provider_gives_up() -> None:
    managed = ManagedProvider(Flaky(5), 1, RateLimiter(1, None), backoff_base_s=0.001)
    with pytest.raises(ProviderError):
        managed.complete(REQ, CancelToken())


def test_non_retryable_not_retried() -> None:
    inner = Flaky(1, retryable=False)
    with pytest.raises(ProviderError):
        ManagedProvider(inner, 5, RateLimiter(1, None), backoff_base_s=0.001).complete(
            REQ, CancelToken()
        )
    assert inner.calls == 1


# -------------------------------------------------------------- anthropic (fake SDK client)
class FakeMessages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.params: dict[str, Any] = {}

    def create(self, **params: Any) -> Any:
        self.params = params
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeClient:
    def __init__(self, response: Any) -> None:
        self.messages = FakeMessages(response)

    def with_options(self, **_: Any) -> FakeClient:
        return self


def anthropic_cfg() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "alias": "claude",
            "provider": "anthropic",
            "model_id": "claude-opus-5-5",
            "remote": True,
            "extra_body": {"output_config": {"effort": "high"}},
        }
    )


def test_anthropic_request_shape_and_usage() -> None:
    response = SimpleNamespace(
        id="msg_1",
        model="claude-opus-5-5",
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="done"),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=4),
    )
    client = FakeClient(response)
    req = CompletionRequest(
        messages=(ChatMessage("system", "sys"), ChatMessage("user", "u")),
        max_tokens=100,
        temperature=0.0,
        top_p=0.9,
        timeout_s=5,
    )
    result = AnthropicProvider(anthropic_cfg(), client=client).complete(req, CancelToken())
    params = client.messages.params
    assert result.text == "done" and result.usage == Usage(10, 4)
    assert params["system"] == "sys" and params["messages"] == [{"role": "user", "content": "u"}]
    assert "temperature" not in params and "top_p" not in params
    assert params["output_config"] == {"effort": "high"}


def test_anthropic_refusal_is_error() -> None:
    response = SimpleNamespace(stop_reason="refusal", content=[], usage=None, model="m", id="x")
    with pytest.raises(ProviderError) as info:
        AnthropicProvider(anthropic_cfg(), client=FakeClient(response)).complete(REQ, CancelToken())
    assert info.value.kind == "refusal"


def test_anthropic_sdk_error_mapping() -> None:
    anthropic = pytest.importorskip("anthropic")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.RateLimitError(
        "slow down", response=httpx.Response(429, request=request), body=None
    )
    with pytest.raises(ProviderError) as info:
        AnthropicProvider(anthropic_cfg(), client=FakeClient(err)).complete(REQ, CancelToken())
    assert info.value.kind == "rate_limit" and info.value.retryable
