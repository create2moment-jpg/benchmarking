"""Harness protocol: how a model (or external agent) is driven through one task attempt."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from engineering_llm_bench.catalog.registry import Task
from engineering_llm_bench.config.models import LimitsConfig, SamplingConfig
from engineering_llm_bench.control import Budget, CancelToken
from engineering_llm_bench.domain import ExecutionMode, FailureClass
from engineering_llm_bench.providers.base import Usage
from engineering_llm_bench.tools.runner import CommandRunner
from engineering_llm_bench.workspace.manager import Workspace


@dataclass
class Trace:
    """Append-only event log for one attempt, persisted as JSONL."""

    events: list[dict[str, Any]] = field(default_factory=list)
    _start: float = field(default_factory=time.monotonic)

    def add(self, kind: str, **data: Any) -> None:
        self.events.append({"t": round(time.monotonic() - self._start, 3), "kind": kind, **data})

    def to_jsonl(self) -> str:
        return "".join(json.dumps(e, default=str) + "\n" for e in self.events)


@dataclass
class HarnessContext:
    task: Task
    workspace: Workspace
    runner: CommandRunner
    budget: Budget
    cancel: CancelToken
    limits: LimitsConfig
    sampling: SamplingConfig
    allowed_commands: set[str]
    request_timeout_s: float
    context_char_budget: int = 400_000


@dataclass
class HarnessResult:
    finished: bool = False
    summary: str = ""
    failure_class: FailureClass = FailureClass.NONE
    error: str | None = None
    turns: int = 0
    tool_calls: int = 0
    tool_calls_ok: int = 0
    malformed_outputs: int = 0
    usage: Usage = field(default_factory=Usage)
    usage_reliable: bool = False
    #: Seconds spent waiting for model responses (None when the harness cannot observe it).
    model_time_s: float | None = 0.0
    trace: Trace = field(default_factory=Trace)


class Harness(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def mode(self) -> ExecutionMode: ...

    def run(self, ctx: HarnessContext) -> HarnessResult: ...
