"""Direct-completion harness: one request, the model returns complete files and a summary."""

from __future__ import annotations

from dataclasses import dataclass

from engineering_llm_bench.control import CancelledError
from engineering_llm_bench.domain import ExecutionMode, FailureClass
from engineering_llm_bench.harnesses import prompts
from engineering_llm_bench.harnesses.agent import _provider_failure
from engineering_llm_bench.harnesses.base import HarnessContext, HarnessResult
from engineering_llm_bench.harnesses.protocol import MalformedOutputError, parse_direct
from engineering_llm_bench.harnesses.tools import ToolError, write_file
from engineering_llm_bench.providers.base import (
    ChatMessage,
    CompletionRequest,
    ManagedProvider,
    ProviderError,
)


@dataclass
class DirectHarness:
    provider: ManagedProvider
    name: str = "direct"
    version: str = "1"
    mode: ExecutionMode = ExecutionMode.DIRECT

    def run(self, ctx: HarnessContext) -> HarnessResult:
        result = HarnessResult(usage_reliable=self.provider.info.capabilities.usage_reporting)
        request = CompletionRequest(
            messages=(
                ChatMessage("system", prompts.direct_system_prompt(ctx.task)),
                ChatMessage("user", prompts.direct_user_prompt(ctx.task)),
            ),
            max_tokens=ctx.sampling.max_tokens,
            temperature=ctx.sampling.temperature,
            top_p=ctx.sampling.top_p,
            seed=ctx.sampling.seed,
            timeout_s=max(1.0, min(ctx.request_timeout_s, ctx.budget.remaining_s())),
            metadata={"task_id": ctx.task.id, "task_dir": str(ctx.task.root), "mode": "direct"},
        )
        result.turns = 1
        try:
            response = self.provider.complete(request, ctx.cancel)
        except CancelledError as exc:
            result.failure_class, result.error = FailureClass.CANCELLED, str(exc) or "cancelled"
            return result
        except ProviderError as exc:
            result.failure_class, result.error = _provider_failure(exc), f"{exc.kind}: {exc}"
            return result
        result.usage = response.usage
        result.trace.add(
            "model_response",
            text=response.text,
            finish_reason=response.finish_reason,
            latency_s=round(response.latency_s, 3),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        try:
            parsed = parse_direct(response.text)
        except MalformedOutputError as exc:
            result.malformed_outputs = 1
            result.failure_class, result.error = FailureClass.MALFORMED_OUTPUT, str(exc)
            return result
        for path, content in parsed.files.items():
            result.tool_calls += 1
            try:
                rel = write_file(ctx, path, content)
            except ToolError as exc:
                result.trace.add("write_refused", path=path, error=str(exc))
                continue
            result.tool_calls_ok += 1
            result.trace.add("write_file", path=rel, chars=len(content))
        result.finished = True
        result.summary = parsed.summary
        return result
