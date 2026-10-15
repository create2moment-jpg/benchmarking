"""Model provider adapters and factory."""

from __future__ import annotations

from engineering_llm_bench.config.models import ModelConfig
from engineering_llm_bench.providers.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResult,
    ManagedProvider,
    Provider,
    ProviderError,
    ProviderInfo,
    RateLimiter,
    Usage,
)


def create_provider(cfg: ModelConfig) -> ManagedProvider:
    """Build the adapter for ``cfg`` wrapped with rate limiting and retries."""
    inner: Provider
    if cfg.provider == "fake":
        from engineering_llm_bench.providers.fake import FakeProvider  # noqa: PLC0415

        inner = FakeProvider(cfg)
    elif cfg.provider == "openai_compatible":
        from engineering_llm_bench.providers.openai_compat import (  # noqa: PLC0415
            OpenAICompatibleProvider,
        )

        inner = OpenAICompatibleProvider(cfg)
    elif cfg.provider == "anthropic":
        from engineering_llm_bench.providers.anthropic_provider import (  # noqa: PLC0415
            AnthropicProvider,
        )

        inner = AnthropicProvider(cfg)
    else:  # pragma: no cover - guarded by config validation
        raise ProviderError("config", f"unsupported provider {cfg.provider!r}")
    return ManagedProvider(
        inner, cfg.max_retries, RateLimiter(cfg.max_concurrency, cfg.requests_per_minute)
    )


__all__ = [
    "ChatMessage",
    "CompletionRequest",
    "CompletionResult",
    "ManagedProvider",
    "Provider",
    "ProviderError",
    "ProviderInfo",
    "Usage",
    "create_provider",
]
