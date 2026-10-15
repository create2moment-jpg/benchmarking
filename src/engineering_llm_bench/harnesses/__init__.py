"""Execution harnesses: direct completion, built-in agent loop, external CLI."""

from __future__ import annotations

from engineering_llm_bench.config.models import HarnessConfig
from engineering_llm_bench.harnesses.agent import AgentHarness
from engineering_llm_bench.harnesses.base import Harness, HarnessContext, HarnessResult, Trace
from engineering_llm_bench.harnesses.cli_harness import CliHarness
from engineering_llm_bench.harnesses.direct import DirectHarness
from engineering_llm_bench.providers.base import ManagedProvider


def create_harness(cfg: HarnessConfig, provider: ManagedProvider | None) -> Harness:
    """Instantiate a harness. ``provider`` is required for built-in harnesses."""
    if cfg.kind == "cli":
        return CliHarness(cfg)
    if provider is None:
        raise ValueError(f"harness {cfg.name!r} requires a model provider")
    if cfg.kind == "direct":
        return DirectHarness(provider, name=cfg.name, version=cfg.version)
    return AgentHarness(provider, name=cfg.name, version=cfg.version)


__all__ = [
    "AgentHarness",
    "CliHarness",
    "DirectHarness",
    "Harness",
    "HarnessContext",
    "HarnessResult",
    "Trace",
    "create_harness",
]
