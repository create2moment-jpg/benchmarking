"""External agent-CLI harness (e.g. a coding-agent CLI driven non-interactively).

The external tool plans and edits with its own tools, so the benchmark controls only: a fresh
workspace as cwd, scrubbed environment (plus explicitly passed-through variables), the attempt
timeout, output caps and - unless the harness is marked ``remote`` - network denial. Token usage
is unknown unless the tool reports it, and is recorded as such (null), never estimated.
"""

from __future__ import annotations

from dataclasses import dataclass

from engineering_llm_bench.config.models import HarnessConfig
from engineering_llm_bench.domain import ExecutionMode, FailureClass
from engineering_llm_bench.harnesses import prompts
from engineering_llm_bench.harnesses.base import HarnessContext, HarnessResult
from engineering_llm_bench.tools.runner import CommandRefusedError

_SUMMARY_CHARS = 3000


def render_argv(argv: list[str], prompt: str, prompt_file: str, workspace: str) -> list[str]:
    out = []
    for arg in argv:
        out.append(
            arg.replace("{prompt_file}", prompt_file)
            .replace("{prompt}", prompt)
            .replace("{workspace}", workspace)
        )
    return out


@dataclass
class CliHarness:
    config: HarnessConfig
    mode: ExecutionMode = ExecutionMode.AGENT

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return self.config.version

    def run(self, ctx: HarnessContext) -> HarnessResult:
        result = HarnessResult(usage_reliable=False, turns=1, model_time_s=None)
        prompt = prompts.external_prompt(ctx.task)
        prompt_file = ctx.workspace.scratch / "prompt.md"
        prompt_file.write_text(prompt, encoding="utf-8")
        argv = render_argv(
            self.config.argv,
            prompt if self.config.prompt_via == "arg" else "",
            str(prompt_file),
            str(ctx.workspace.root),
        )
        result.trace.add(
            "external_command", argv=self.config.argv, prompt_via=self.config.prompt_via
        )
        try:
            run = ctx.runner.run_trusted(
                argv,
                cwd=ctx.workspace.root,
                scratch=ctx.workspace.scratch,
                timeout_s=max(1.0, ctx.budget.remaining_s()),
                cancel=ctx.cancel,
                stdin_text=prompt if self.config.prompt_via == "stdin" else None,
                env_passthrough=self.config.env_passthrough,
                allow_network=self.config.remote,
            )
        except CommandRefusedError as exc:
            result.failure_class, result.error = FailureClass.HARNESS_ERROR, str(exc)
            return result
        result.trace.add(
            "external_result",
            exit_code=run.exit_code,
            timed_out=run.timed_out,
            duration_s=round(run.duration_s, 3),
            stdout=run.stdout[-20000:],
            stderr=run.stderr[-5000:],
        )
        result.summary = run.stdout.strip()[-_SUMMARY_CHARS:]
        if run.cancelled:
            result.failure_class, result.error = FailureClass.CANCELLED, "cancelled"
        elif run.timed_out:
            result.failure_class, result.error = FailureClass.TIMEOUT, "external harness timed out"
        elif run.exit_code != 0:
            result.failure_class = FailureClass.HARNESS_ERROR
            result.error = f"external harness exited with {run.exit_code}"
        else:
            result.finished = True
        return result
