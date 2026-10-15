"""Deterministic MOCK provider for tests and pipeline smoke runs.

This adapter does not call any model. Its scores measure the benchmark pipeline, not a model,
and every run using it is recorded with ``provider = "fake"`` so reports can label it. Behaviours:

* ``reference`` - replays the task's reference solution (proves a correct patch passes);
* ``broken``    - writes the task's ``broken/`` overlay, if any (proves a bad patch fails);
* ``noop``      - changes nothing and finishes;
* ``error``     - raises a non-retryable provider error;
* ``timeout``   - blocks until the request timeout (or cancellation) then raises a timeout;
* ``malformed`` - returns text that violates the harness protocol.

It needs the local-only request metadata ``task_dir`` and ``mode`` supplied by the harness.
"""

from __future__ import annotations

import time
from pathlib import Path

from engineering_llm_bench.config.models import FakeConfig, ModelConfig
from engineering_llm_bench.control import CancelToken
from engineering_llm_bench.harnesses.protocol import render_action, render_direct
from engineering_llm_bench.providers.base import (
    CompletionRequest,
    CompletionResult,
    ProviderCapabilities,
    ProviderError,
    ProviderInfo,
    Usage,
    info_from_config,
)
from engineering_llm_bench.workspace.manager import iter_files

FAKE_SUMMARY = (
    "Applied the change described by the task. Tested by running the test suite locally. "
    "Limitations: this is a scripted mock provider output, not a model."
)


def _overlay_files(task_dir: Path, behavior: str) -> dict[str, str]:
    sub = {"reference": "reference", "broken": "broken"}.get(behavior)
    if sub is None:
        return {}
    root = task_dir / sub
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in iter_files(root)}


class FakeProvider:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.fake = cfg.fake or FakeConfig()
        # usage_reporting=False: mock token counts are estimates and must not be stored as usage.
        self._info = info_from_config(cfg, ProviderCapabilities(usage_reporting=False, seed=True))

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def complete(self, request: CompletionRequest, cancel: CancelToken) -> CompletionResult:
        start = time.monotonic()
        cancel.raise_if_cancelled()
        behavior = self.fake.behavior
        if self.fake.latency_s:
            self._sleep(self.fake.latency_s, cancel)
        if behavior == "error":
            raise ProviderError("client", "fake provider configured to fail (behavior=error)")
        if behavior == "timeout":
            self._sleep(request.timeout_s, cancel)
            raise ProviderError("timeout", f"fake provider timed out after {request.timeout_s}s")
        if behavior == "malformed":
            text = "Sure! I would change some files. (no protocol-conforming output)"
        else:
            text = self._script(request, behavior)
        prompt_chars = sum(len(m.content) for m in request.messages)
        return CompletionResult(
            text=text,
            usage=Usage(prompt_chars // 4, len(text) // 4),
            finish_reason="stop",
            latency_s=time.monotonic() - start,
            model_reported=self.cfg.model_id,
        )

    @staticmethod
    def _sleep(seconds: float, cancel: CancelToken) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            cancel.raise_if_cancelled()
            time.sleep(0.02)

    def _script(self, request: CompletionRequest, behavior: str) -> str:
        task_dir = Path(request.metadata.get("task_dir", ""))
        files = _overlay_files(task_dir, behavior) if task_dir.is_dir() else {}
        if request.metadata.get("mode") == "direct":
            return render_direct(files, FAKE_SUMMARY)
        # Agent mode: derive the step from the number of previous assistant turns.
        step = sum(1 for m in request.messages if m.role == "assistant")
        plan: list[dict[str, object]] = [{"action": "list_files", "path": "."}]
        plan += [{"action": "write_file", "path": p, "content": c} for p, c in files.items()]
        allowed = request.metadata.get("allowed_commands", "").split(",")
        if "pytest" in allowed and request.metadata.get("has_tests") == "1":
            plan.append({"action": "run_command", "argv": ["pytest", "-q"]})
        plan.append({"action": "finish", "summary": FAKE_SUMMARY})
        return render_action(plan[min(step, len(plan) - 1)])

    def close(self) -> None:
        return None
