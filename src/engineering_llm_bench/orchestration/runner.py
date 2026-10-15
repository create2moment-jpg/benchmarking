"""Run lifecycle: resolve -> prepare -> execute -> verify -> score -> persist -> report.

One *run* = one (model, harness, profile, config) combination over a set of tasks; each
(task, repeat) is an *attempt* with its own disposable workspace and artifact directory.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engineering_llm_bench.catalog.registry import Task, TaskRegistry, check_availability
from engineering_llm_bench.config.loader import ConfigError
from engineering_llm_bench.config.models import (
    BenchmarkConfig,
    HarnessConfig,
    ModelConfig,
    ModelsFile,
    ProfileConfig,
)
from engineering_llm_bench.control import Budget, CancelToken
from engineering_llm_bench.domain import (
    AttemptStatus,
    ExecutionMode,
    FailureClass,
    Profile,
    RunStatus,
)
from engineering_llm_bench.environment import (
    collect_environment,
    environment_fingerprint,
    fingerprint,
)
from engineering_llm_bench.evaluation.evaluator import EvaluationOutcome, evaluate
from engineering_llm_bench.harnesses import Harness, HarnessContext, HarnessResult, create_harness
from engineering_llm_bench.orchestration.records import artifact_rows, evaluation_rows
from engineering_llm_bench.providers import ManagedProvider, create_provider
from engineering_llm_bench.redaction import Redactor
from engineering_llm_bench.storage.artifacts import ArtifactRef, ArtifactStore
from engineering_llm_bench.storage.db import Store, dumps, utcnow
from engineering_llm_bench.tools.runner import CommandRunner
from engineering_llm_bench.workspace.manager import Workspace, prepare_workspace, snapshot_hash

ProgressFn = Callable[[str], None]
_TERMINAL = {AttemptStatus.COMPLETED.value, AttemptStatus.SKIPPED.value}


@dataclass
class RunRequest:
    model_alias: str
    harness: str
    profile: Profile = Profile.SMOKE
    task_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    repeats: int | None = None
    concurrency: int | None = None
    allow_remote: bool = False
    allow_external_harness: bool = False
    resume_run_id: str | None = None
    notes: str | None = None


@dataclass
class ResolvedRun:
    request: RunRequest
    model: ModelConfig | None
    model_label: str
    harness_cfg: HarnessConfig
    profile_cfg: ProfileConfig
    tasks: list[Task]
    repeats: int
    concurrency: int

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.DIRECT if self.harness_cfg.kind == "direct" else ExecutionMode.AGENT

    @property
    def remote(self) -> bool:
        return bool(self.model and self.model.remote) or self.harness_cfg.remote


@dataclass
class DryRunPlan:
    run: ResolvedRun
    rows: list[dict[str, Any]]
    attempts: int
    estimated_s: float
    max_model_requests: int | None
    max_tokens: int | None
    cost_upper_bound: float | None
    warnings: list[str]


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)[:80]


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


class BenchmarkRunner:
    def __init__(
        self,
        bench: BenchmarkConfig,
        models: ModelsFile,
        registry: TaskRegistry,
        store: Store,
        artifacts: ArtifactStore,
        progress: ProgressFn | None = None,
    ) -> None:
        self.bench = bench
        self.models = models
        self.registry = registry
        self.store = store
        self.artifacts = artifacts
        self.progress = progress or (lambda _msg: None)
        self._eval_sem = threading.BoundedSemaphore(bench.concurrency.max_parallel_evaluations)

    # ------------------------------------------------------------------ resolve
    def resolve(self, req: RunRequest) -> ResolvedRun:
        """Validate everything up front and fail early with actionable messages."""
        try:
            harness_cfg = self.models.harness(req.harness)
        except KeyError as exc:
            raise ConfigError(str(exc.args[0])) from None
        model: ModelConfig | None = None
        if harness_cfg.kind == "cli":
            if not req.allow_external_harness:
                raise ConfigError(
                    f"harness {harness_cfg.name!r} runs an external agent CLI; "
                    "pass --allow-external-harness to enable it"
                )
            label = req.model_alias or harness_cfg.name
        else:
            try:
                model = self.models.model(req.model_alias)
            except KeyError as exc:
                raise ConfigError(str(exc.args[0])) from None
            label = model.alias
        remote = bool(model and model.remote) or harness_cfg.remote
        if remote and not (req.allow_remote or self.bench.allow_remote_providers):
            raise ConfigError(
                f"{label!r} is a REMOTE provider/harness: task prompts and source code will be "
                "sent to it. Re-run with --allow-remote (or set allow_remote_providers: true) "
                "to consent."
            )
        profile_cfg = self.bench.profiles[req.profile]
        tasks = self.registry.select(
            req.profile, profile_cfg, req.task_ids or None, req.tags or None, req.domains or None
        )
        if not tasks:
            raise ConfigError("no tasks selected (check --task/--tag/--domain filters)")
        unsafe = sorted(set(harness_cfg.argv[:1]) & {"sh", "bash", "sudo"})
        if unsafe:
            raise ConfigError(f"harness argv must not start with a shell: {unsafe}")
        return ResolvedRun(
            request=req,
            model=model,
            model_label=label,
            harness_cfg=harness_cfg,
            profile_cfg=profile_cfg,
            tasks=tasks,
            repeats=req.repeats or profile_cfg.repeats,
            concurrency=req.concurrency or self.bench.concurrency.max_parallel_attempts,
        )

    # ------------------------------------------------------------------ dry run
    def dry_run(self, req: RunRequest) -> DryRunPlan:
        """Estimate work without invoking any provider or harness."""
        run = self.resolve(req)
        rows: list[dict[str, Any]] = []
        total_s = 0.0
        attempts = 0
        for task in run.tasks:
            avail = check_availability(task)
            timeout = min(task.spec.timeout_s, run.profile_cfg.max_task_timeout_s)
            est = min(task.spec.estimated_runtime_s, timeout)
            if avail.available:
                total_s += est * run.repeats
                attempts += run.repeats
            rows.append(
                {
                    "task": task.id,
                    "domain": task.spec.domain.value,
                    "evaluation": task.spec.evaluation.value,
                    "available": avail.available,
                    "reason": avail.reason or "",
                    "est_s": est,
                    "timeout_s": timeout,
                }
            )
        turns = 1 if run.mode == ExecutionMode.DIRECT else run.profile_cfg.max_turns
        max_requests = None if run.harness_cfg.kind == "cli" else turns * attempts
        max_tokens = (
            run.profile_cfg.max_total_tokens * attempts
            if run.profile_cfg.max_total_tokens
            else None
        )
        cost = None
        if max_tokens and run.model and run.model.pricing:
            price = max(run.model.pricing.input_per_mtok, run.model.pricing.output_per_mtok)
            cost = max_tokens / 1e6 * price
        warnings: list[str] = []
        if run.remote:
            warnings.append("remote provider: task prompts and fixture source will be sent")
        if req.profile == Profile.EXTENDED or run.profile_cfg.warn_cost:
            warnings.append(
                f"extended profile: up to ~{total_s / 60 / run.concurrency:.0f} min wall time"
                + (
                    f" and up to ~{cost:.2f} {run.model.pricing.currency} (upper bound)"
                    if cost is not None and run.model and run.model.pricing
                    else ""
                )
            )
        if max_tokens is None and run.remote:
            warnings.append("no max_total_tokens budget set for this profile; cost is unbounded")
        return DryRunPlan(
            run, rows, attempts, total_s / run.concurrency, max_requests, max_tokens, cost, warnings
        )

    # ------------------------------------------------------------------ run
    def _fingerprint(self, run: ResolvedRun) -> str:
        return fingerprint(
            {
                "model": run.model.model_dump(exclude={"alias", "notes"}) if run.model else None,
                "harness": run.harness_cfg.model_dump(exclude={"description"}),
                "profile": run.profile_cfg.model_dump(exclude={"description"}),
                "weights": self.bench.weights,
                "limits": self.bench.limits.model_dump(),
                "network": self.bench.isolation.network,
                "suite_version": self.bench.suite_version,
            }
        )

    def _start_run(self, run: ResolvedRun) -> str:
        req = run.request
        config_fp = self._fingerprint(run)
        if req.resume_run_id:
            existing = self.store.get_run(req.resume_run_id)
            if existing is None:
                raise ConfigError(f"cannot resume: unknown run id {req.resume_run_id}")
            if existing["config_fingerprint"] != config_fp:
                raise ConfigError(
                    "cannot resume: model/harness/profile configuration changed since the run "
                    "started (config fingerprint differs); start a new run instead"
                )
            self.store.update_run(req.resume_run_id, status=RunStatus.RUNNING.value, error=None)
            return req.resume_run_id
        env = collect_environment()
        run_id = new_run_id()
        self.store.insert_run(
            {
                "id": run_id,
                "created_at": utcnow(),
                "status": RunStatus.RUNNING.value,
                "source": "executed",
                "profile": req.profile.value,
                "suite_name": self.bench.suite_name,
                "suite_version": self.bench.suite_version,
                "model_alias": run.model_label,
                "provider": run.model.provider if run.model else "external",
                "provider_model_id": run.model.model_id if run.model else None,
                "endpoint_type": run.model.endpoint_type if run.model else None,
                "remote": int(run.remote),
                "harness": run.harness_cfg.name,
                "harness_version": run.harness_cfg.version,
                "mode": run.mode.value,
                "model_config_json": dumps(run.model.model_dump()) if run.model else None,
                "harness_config_json": dumps(run.harness_cfg.model_dump()),
                "run_config_json": dumps(
                    {
                        "profile": run.profile_cfg.model_dump(),
                        "repeats": run.repeats,
                        "concurrency": run.concurrency,
                        "tasks": [t.id for t in run.tasks],
                        "weights": self.bench.weights,
                        "limits": self.bench.limits.model_dump(),
                        "isolation": self.bench.isolation.model_dump(),
                    }
                ),
                "config_fingerprint": config_fp,
                "env_json": dumps(env),
                "env_fingerprint": environment_fingerprint(env),
                "task_count": len(run.tasks),
                "notes": req.notes,
            }
        )
        return run_id

    def run(self, req: RunRequest, cancel: CancelToken | None = None) -> str:
        """Execute a run and return its id. Interruptions leave inspectable, resumable state."""
        resolved = self.resolve(req)
        cancel = cancel or CancelToken()
        run_id = self._start_run(resolved)
        command_runner = CommandRunner(self.bench.limits, self.bench.isolation.network)
        provider = create_provider(resolved.model) if resolved.model else None
        harness = create_harness(resolved.harness_cfg, provider)
        existing = {
            (a["task_id"], a["repeat_index"]): a for a in self.store.attempts_for_run(run_id)
        }
        planned: list[tuple[Task, int, str]] = []
        for task in resolved.tasks:
            for rep in range(resolved.repeats):
                prior = existing.get((task.id, rep))
                if prior and prior["status"] in _TERMINAL:
                    continue
                aid = prior["id"] if prior else f"{run_id}--{_slug(task.id)}--r{rep}"
                planned.append((task, rep, aid))
                self.store.upsert_attempt(
                    {
                        "id": aid,
                        "run_id": run_id,
                        "task_id": task.id,
                        "task_version": task.spec.version,
                        "task_hash": task.content_hash,
                        "task_domain": task.spec.domain.value,
                        "evaluation_kind": task.spec.evaluation.value,
                        "repeat_index": rep,
                        "status": AttemptStatus.PENDING.value,
                    }
                )
        self.progress(
            f"run {run_id}: {len(planned)} attempt(s) for {resolved.model_label} "
            f"[{resolved.harness_cfg.name}, {req.profile.value}]"
        )
        timer = threading.Timer(
            resolved.profile_cfg.run_timeout_s, lambda: cancel.cancel("whole-run timeout reached")
        )
        timer.daemon = True
        timer.start()
        run_error: str | None = None
        try:
            with ThreadPoolExecutor(max_workers=resolved.concurrency) as pool:
                futures: list[Future[None]] = [
                    pool.submit(
                        self._attempt,
                        run_id,
                        resolved,
                        harness,
                        provider,
                        command_runner,
                        task,
                        rep,
                        aid,
                        cancel,
                    )
                    for task, rep, aid in planned
                ]
                try:
                    for fut in futures:
                        while not fut.done():
                            time.sleep(0.05)
                        fut.result()
                except KeyboardInterrupt:
                    cancel.cancel("interrupted by user")
                    self.progress("interrupt received: cancelling running attempts...")
                    for fut in futures:
                        fut.cancel()
        except Exception as exc:  # unexpected orchestration error: record, then re-raise
            run_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            timer.cancel()
            if provider is not None:
                provider.close()
            self._finalize(run_id, cancel, run_error)
        return run_id

    def _finalize(self, run_id: str, cancel: CancelToken, error: str | None) -> None:
        attempts = self.store.attempts_for_run(run_id)
        for a in attempts:
            if a["status"] in (AttemptStatus.PENDING.value, AttemptStatus.RUNNING.value):
                self.store.upsert_attempt(
                    {
                        "id": a["id"],
                        "status": AttemptStatus.INTERRUPTED.value,
                        "failure_class": FailureClass.CANCELLED.value,
                        "error": cancel.reason or error or "process stopped before completion",
                        "finished_at": utcnow(),
                    }
                )
        attempts = self.store.attempts_for_run(run_id)
        if error:
            status = RunStatus.FAILED
        elif any(a["status"] == AttemptStatus.INTERRUPTED.value for a in attempts):
            status = RunStatus.INTERRUPTED
        else:
            status = RunStatus.COMPLETED
        self.store.update_run(
            run_id, status=status.value, finished_at=utcnow(), error=error or cancel.reason
        )
        self.progress(f"run {run_id}: {status.value}")

    # ------------------------------------------------------------------ attempt
    def _attempt(
        self,
        run_id: str,
        run: ResolvedRun,
        harness: Harness,
        provider: ManagedProvider | None,
        command_runner: CommandRunner,
        task: Task,
        rep: int,
        aid: str,
        run_cancel: CancelToken,
    ) -> None:
        if run_cancel.cancelled:
            return
        avail = check_availability(task)
        base = {"id": aid, "run_id": run_id, "task_id": task.id}
        if not avail.available:
            self.store.upsert_attempt(
                {
                    **base,
                    "status": AttemptStatus.SKIPPED.value,
                    "failure_class": FailureClass.UNAVAILABLE.value,
                    "skip_reason": avail.reason,
                    "finished_at": utcnow(),
                }
            )
            self.progress(f"  SKIP  {task.id} r{rep}: {avail.reason}")
            return
        started = utcnow()
        self.store.upsert_attempt(
            {**base, "status": AttemptStatus.RUNNING.value, "started_at": started}
        )
        cancel = run_cancel.child()
        workspace = prepare_workspace(task, keep=self.bench.isolation.keep_workspaces)
        try:
            self._execute_attempt(
                run_id,
                run,
                harness,
                provider,
                command_runner,
                task,
                rep,
                aid,
                cancel,
                workspace,
                started,
            )
        finally:
            workspace.cleanup()

    def _execute_attempt(
        self,
        run_id: str,
        run: ResolvedRun,
        harness: Harness,
        provider: ManagedProvider | None,
        command_runner: CommandRunner,
        task: Task,
        rep: int,
        aid: str,
        cancel: CancelToken,
        workspace: Workspace,
        started: str,
    ) -> None:
        timeout = min(task.spec.timeout_s, run.profile_cfg.max_task_timeout_s)
        budget = Budget.from_timeout(
            timeout, run.profile_cfg.max_turns, run.profile_cfg.max_total_tokens
        )
        allowed = set(task.spec.allowed_commands) & set(self.bench.command_allowlist)
        sampling = run.model.sampling if run.model else None
        ctx = HarnessContext(
            task=task,
            workspace=workspace,
            runner=command_runner,
            budget=budget,
            cancel=cancel,
            limits=self.bench.limits,
            sampling=sampling or _default_sampling(),
            allowed_commands=allowed,
            request_timeout_s=run.model.timeout_s if run.model else timeout,
            context_char_budget=(run.model.context_limit * 3)
            if run.model and run.model.context_limit
            else 400_000,
        )
        t0 = time.monotonic()
        try:
            result = harness.run(ctx)
        except Exception as exc:  # harness bug: classify as harness error, keep evidence
            result = HarnessResult(
                failure_class=FailureClass.HARNESS_ERROR, error=f"{type(exc).__name__}: {exc}"
            )
        agent_wall = time.monotonic() - t0
        refs: list[ArtifactRef] = [
            self.artifacts.write_text(run_id, aid, "trace.jsonl", result.trace.to_jsonl()),
            self.artifacts.write_text(run_id, aid, "diff.patch", workspace.diff()),
            self.artifacts.write_text(run_id, aid, "summary.md", result.summary or ""),
        ]
        final_hash = snapshot_hash(workspace.current())
        outcome: EvaluationOutcome | None = None
        eval_wall = None
        eval_error: str | None = None
        if not cancel.cancelled:
            t1 = time.monotonic()
            with self._eval_sem:
                try:
                    outcome = evaluate(
                        task,
                        workspace,
                        result,
                        run.mode,
                        command_runner,
                        self.bench.limits,
                        self.bench.command_allowlist,
                        self.bench.weights,
                        cancel,
                    )
                except Exception as exc:
                    eval_error = f"{type(exc).__name__}: {exc}"
            eval_wall = time.monotonic() - t1
        status, failure = _classify(result, outcome, cancel, eval_error)
        checks_rows: list[dict[str, Any]] = []
        dims_rows: list[dict[str, Any]] = []
        notes: list[str] = []
        if outcome is not None:
            dims_rows, checks_rows = evaluation_rows(outcome, self.artifacts, run_id, aid, refs)
            notes = outcome.scorecard.notes
        usage = result.usage
        cost = None
        if (
            run.model
            and run.model.pricing
            and result.usage_reliable
            and usage.input_tokens is not None
            and usage.output_tokens is not None
            and run.model.provider != "fake"
        ):
            cost = (
                usage.input_tokens * run.model.pricing.input_per_mtok
                + usage.output_tokens * run.model.pricing.output_per_mtok
            ) / 1e6
        card = outcome.scorecard if outcome else None
        attempt_row = {
            "id": aid,
            "run_id": run_id,
            "task_id": task.id,
            "task_version": task.spec.version,
            "task_hash": task.content_hash,
            "task_domain": task.spec.domain.value,
            "evaluation_kind": task.spec.evaluation.value,
            "repeat_index": rep,
            "status": status.value,
            "failure_class": failure.value,
            "error": self.artifacts.redactor.text(result.error or eval_error or "") or None,
            "started_at": started,
            "finished_at": utcnow(),
            "agent_wall_s": round(agent_wall, 3),
            "eval_wall_s": round(eval_wall, 3) if eval_wall is not None else None,
            "turns": result.turns,
            "tool_calls": result.tool_calls,
            "tool_calls_ok": result.tool_calls_ok,
            "malformed_outputs": result.malformed_outputs,
            "input_tokens": usage.input_tokens if result.usage_reliable else None,
            "output_tokens": usage.output_tokens if result.usage_reliable else None,
            "usage_reliable": int(result.usage_reliable),
            "cost_usd": cost,
            "score": round(card.score, 3) if card and card.score is not None else None,
            "coverage": round(card.coverage, 3) if card else None,
            "passed": None
            if not card or card.passed is None or status != AttemptStatus.COMPLETED
            else int(card.passed),
            "regression_failed": None
            if not card or card.regression_failed is None
            else int(card.regression_failed),
            "summary": self.artifacts.redactor.text(result.summary)[:5000],
            "baseline_hash": workspace.baseline_hash,
            "final_hash": final_hash,
            "artifact_dir": self.artifacts.attempt_dir(run_id, aid)
            .relative_to(self.artifacts.root)
            .as_posix(),
            "notes_json": dumps(notes),
        }
        self.store.save_evaluation(
            attempt_row,
            dims_rows,
            checks_rows,
            artifact_rows(refs),
        )
        score_txt = f"{card.score:.1f}" if card and card.score is not None else "-"
        self.progress(
            f"  {status.value.upper():<11} {task.id} r{rep}: score={score_txt} "
            f"failure={failure.value} ({agent_wall:.1f}s)"
        )


def _default_sampling() -> Any:
    from engineering_llm_bench.config.models import SamplingConfig  # noqa: PLC0415

    return SamplingConfig()


def _classify(
    result: HarnessResult,
    outcome: EvaluationOutcome | None,
    cancel: CancelToken,
    eval_error: str | None,
) -> tuple[AttemptStatus, FailureClass]:
    if cancel.cancelled and outcome is None:
        return AttemptStatus.INTERRUPTED, FailureClass.CANCELLED
    if eval_error is not None or outcome is None:
        return AttemptStatus.FAILED, FailureClass.EVALUATION_ERROR
    if result.failure_class != FailureClass.NONE:
        return AttemptStatus.COMPLETED, result.failure_class
    if outcome.scorecard.passed is False:
        return AttemptStatus.COMPLETED, FailureClass.TESTS_FAILED
    return AttemptStatus.COMPLETED, FailureClass.NONE


def default_paths(bench: BenchmarkConfig, base: Path) -> tuple[Path, Path, Path]:
    """Resolve tasks dir, artifacts dir and database path relative to ``base``."""

    def res(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (base / path)

    return res(bench.tasks_dir), res(bench.artifacts_dir), res(bench.database)


def build_redactor(models: ModelsFile, bench: BenchmarkConfig) -> Redactor:
    names = [m.api_key_env for m in models.models if m.api_key_env]
    names += ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    for m in models.models:
        names += re.findall(r"\$\{([A-Z_][A-Z0-9_]*)\}", " ".join(m.headers.values()))
    for h in models.harnesses:
        names += h.env_passthrough
    return Redactor(secret_env_names=names, extra_patterns=bench.redact_patterns)
