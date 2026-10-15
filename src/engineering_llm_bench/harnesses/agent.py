"""Built-in agent harness: bounded plan -> edit -> run tools/tests -> revise -> finish loop."""

from __future__ import annotations

from dataclasses import dataclass

from engineering_llm_bench.control import (
    BudgetExceededError,
    CancelledError,
    DeadlineExceededError,
)
from engineering_llm_bench.domain import ExecutionMode, FailureClass
from engineering_llm_bench.harnesses import prompts
from engineering_llm_bench.harnesses.base import HarnessContext, HarnessResult
from engineering_llm_bench.harnesses.protocol import MalformedOutputError, parse_action
from engineering_llm_bench.harnesses.tools import execute
from engineering_llm_bench.providers.base import (
    ChatMessage,
    CompletionRequest,
    ManagedProvider,
    ProviderError,
)

MAX_CONSECUTIVE_MALFORMED = 3
_ELIDED = "[earlier observation elided to fit the context budget]"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n...[{len(text) - limit} chars truncated]...\n" + text[-half:]


def _fit_context(messages: list[ChatMessage], budget_chars: int) -> list[ChatMessage]:
    """Elide the oldest tool observations until the conversation fits ``budget_chars``.

    The system prompt, the task prompt and all assistant actions are always kept.
    """
    total = sum(len(m.content) for m in messages)
    if total <= budget_chars:
        return messages
    out = list(messages)
    for i in range(2, len(out)):
        if total <= budget_chars:
            break
        if out[i].role == "user" and out[i].content != _ELIDED:
            total -= len(out[i].content) - len(_ELIDED)
            out[i] = ChatMessage("user", _ELIDED)
    return out


def _provider_failure(exc: ProviderError) -> FailureClass:
    if exc.kind == "timeout":
        return FailureClass.TIMEOUT
    if exc.kind == "malformed":
        return FailureClass.MALFORMED_OUTPUT
    return FailureClass.PROVIDER_ERROR


@dataclass
class AgentHarness:
    provider: ManagedProvider
    name: str = "agent"
    version: str = "1"
    mode: ExecutionMode = ExecutionMode.AGENT

    def run(self, ctx: HarnessContext) -> HarnessResult:
        result = HarnessResult(usage_reliable=self.provider.info.capabilities.usage_reporting)
        trace = result.trace
        messages = [
            ChatMessage(
                "system",
                prompts.agent_system_prompt(ctx.task, ctx.allowed_commands, ctx.budget.max_turns),
            ),
            ChatMessage("user", prompts.agent_user_prompt(ctx.task)),
        ]
        has_tests = any(p.name.startswith("test") for p in ctx.task.fixture_dir.rglob("*.py"))
        metadata = {
            "task_id": ctx.task.id,
            "task_dir": str(ctx.task.root),
            "mode": "agent",
            "allowed_commands": ",".join(sorted(ctx.allowed_commands)),
            "has_tests": "1" if has_tests else "0",
        }
        consecutive_malformed = 0
        try:
            while True:
                ctx.cancel.raise_if_cancelled()
                ctx.budget.use_turn()
                result.turns += 1
                request = CompletionRequest(
                    messages=tuple(_fit_context(messages, ctx.context_char_budget)),
                    max_tokens=ctx.sampling.max_tokens,
                    temperature=ctx.sampling.temperature,
                    top_p=ctx.sampling.top_p,
                    seed=ctx.sampling.seed,
                    timeout_s=max(1.0, min(ctx.request_timeout_s, ctx.budget.remaining_s())),
                    metadata=metadata,
                )
                trace.add("model_request", turn=result.turns, messages=len(request.messages))
                response = self.provider.complete(request, ctx.cancel)
                result.usage = result.usage + response.usage
                trace.add(
                    "model_response",
                    turn=result.turns,
                    text=response.text,
                    finish_reason=response.finish_reason,
                    latency_s=round(response.latency_s, 3),
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
                ctx.budget.add_tokens(response.usage.total)
                messages.append(ChatMessage("assistant", response.text))
                try:
                    action = parse_action(response.text)
                except MalformedOutputError as exc:
                    result.malformed_outputs += 1
                    consecutive_malformed += 1
                    trace.add("malformed_output", turn=result.turns, error=str(exc))
                    if consecutive_malformed >= MAX_CONSECUTIVE_MALFORMED:
                        result.failure_class = FailureClass.MALFORMED_OUTPUT
                        result.error = f"{consecutive_malformed} consecutive malformed replies"
                        return result
                    messages.append(ChatMessage("user", f"PROTOCOL ERROR: {exc}"))
                    continue
                consecutive_malformed = 0
                if action.action == "finish":
                    result.finished = True
                    result.summary = action.summary or ""
                    trace.add("finish", summary=result.summary)
                    return result
                outcome = execute(ctx, action)
                result.tool_calls += 1
                result.tool_calls_ok += int(outcome.ok)
                trace.add(
                    "tool_call",
                    turn=result.turns,
                    action=action.action,
                    path=action.path,
                    argv=list(action.argv) or None,
                    ok=outcome.ok,
                    exit_code=outcome.exit_code,
                    observation=_truncate(outcome.observation, 4000),
                )
                messages.append(
                    ChatMessage(
                        "user", _truncate(outcome.observation, ctx.limits.max_observation_chars)
                    )
                )
        except CancelledError as exc:
            result.failure_class, result.error = FailureClass.CANCELLED, str(exc) or "cancelled"
        except DeadlineExceededError as exc:
            result.failure_class, result.error = FailureClass.TIMEOUT, str(exc)
        except BudgetExceededError as exc:
            result.failure_class, result.error = FailureClass.BUDGET_EXCEEDED, str(exc)
        except ProviderError as exc:
            result.failure_class, result.error = _provider_failure(exc), f"{exc.kind}: {exc}"
        trace.add("stopped", failure_class=result.failure_class.value, error=result.error)
        return result
