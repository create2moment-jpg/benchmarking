"""``bench`` command-line interface.

Exit codes: 0 success; 1 validation found problems; 2 configuration/usage error;
3 run interrupted (resumable with ``--resume``); 4 run failed (infrastructure error).
Paths in config files are resolved relative to the current working directory.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Annotated, Any

import typer

from engineering_llm_bench import __version__
from engineering_llm_bench.catalog.registry import TaskRegistry, check_availability, load_task
from engineering_llm_bench.config.loader import ConfigError, load_benchmark, load_models
from engineering_llm_bench.config.models import BenchmarkConfig, ModelsFile
from engineering_llm_bench.domain import Dimension, Profile, RunStatus
from engineering_llm_bench.orchestration.runner import (
    BenchmarkRunner,
    RunRequest,
    build_redactor,
    default_paths,
)
from engineering_llm_bench.redaction import Redactor
from engineering_llm_bench.storage.artifacts import ArtifactStore, atomic_write_bytes
from engineering_llm_bench.storage.db import SchemaVersionError, Store

app = typer.Typer(
    help="Benchmark local and paid LLMs / agent harnesses on reproducible engineering tasks.",
    no_args_is_help=True,
    add_completion=False,
)
task_app = typer.Typer(help="Task catalogue tools.", no_args_is_help=True)
review_app = typer.Typer(help="Human / judge review of rubric dimensions.", no_args_is_help=True)
app.add_typer(task_app, name="task")
byok_app = typer.Typer(
    help="Copilot BYOK / IDE-agent sessions: starting_point.md + result collection.",
    no_args_is_help=True,
)
app.add_typer(review_app, name="review")
app.add_typer(byok_app, name="byok")

EXIT_PROBLEMS, EXIT_CONFIG, EXIT_INTERRUPTED, EXIT_FAILED = 1, 2, 3, 4


def _base_dir() -> Path:
    """Directory that relative config paths resolve against (env BENCH_BASE_DIR, else cwd)."""
    return Path(os.environ.get("BENCH_BASE_DIR") or Path.cwd())


def _default_config(env: str, name: str) -> Path:
    if os.environ.get(env):
        return Path(os.environ[env])
    configs = _base_dir() / "configs"
    local = configs / f"{name}.yaml"
    return local if local.exists() else configs / f"{name}.example.yaml"


ConfigOpt = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Benchmark config (default: configs/benchmark.yaml, "
        "else benchmark.example.yaml; env BENCH_CONFIG).",
    ),
]
ModelsOpt = Annotated[
    Path | None,
    typer.Option(
        "--models",
        "-m",
        help="Models config (default: configs/models.yaml, else "
        "models.example.yaml; env BENCH_MODELS).",
    ),
]


@dataclass
class Ctx:
    bench: BenchmarkConfig
    models: ModelsFile
    registry: TaskRegistry
    tasks_dir: Path
    artifacts_dir: Path
    db_path: Path
    redactor: Redactor

    def store(self) -> Store:
        return Store(self.db_path)

    def artifacts(self) -> ArtifactStore:
        return ArtifactStore(self.artifacts_dir, self.redactor)


def _fail(message: str, code: int = EXIT_CONFIG) -> typer.Exit:
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(code)


def load_ctx(config: Path | None, models: Path | None, need_tasks: bool = True) -> Ctx:
    try:
        bench = load_benchmark(config or _default_config("BENCH_CONFIG", "benchmark"))
        models_file = load_models(models or _default_config("BENCH_MODELS", "models"))
        tasks_dir, artifacts_dir, db_path = default_paths(bench, _base_dir())
        registry = TaskRegistry.discover(tasks_dir) if need_tasks else TaskRegistry(tasks_dir)
    except ConfigError as exc:
        raise _fail(str(exc)) from None
    return Ctx(
        bench,
        models_file,
        registry,
        tasks_dir,
        artifacts_dir,
        db_path,
        build_redactor(models_file, bench),
    )


def _table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        typer.echo("(none)")
        return
    text = [[("-" if r.get(c) is None else str(r.get(c))) for c in columns] for r in rows]
    widths = [max(len(c), *(len(t[i]) for t in text)) for i, c in enumerate(columns)]
    typer.echo("  ".join(c.upper().ljust(w) for c, w in zip(columns, widths, strict=True)))
    for t in text:
        typer.echo("  ".join(v.ljust(w) for v, w in zip(t, widths, strict=True)))


def _csv(value: str | None) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


# ---------------------------------------------------------------------------------- config


@app.command("version")
def version() -> None:
    """Print the tool version."""
    typer.echo(__version__)


@app.command("validate-config")
def validate_config(config: ConfigOpt = None, models: ModelsOpt = None) -> None:
    """Validate benchmark + models config and the task catalogue (no side effects)."""
    ctx = load_ctx(config, models)
    problems: list[str] = []
    try:
        ctx.models.harness(ctx.bench.default_harness)
    except KeyError as exc:
        problems.append(str(exc.args[0]))
    for profile, cfg in ctx.bench.profiles.items():
        unknown = ctx.registry.unknown_ids(cfg.include_tasks + cfg.exclude_tasks)
        if unknown:
            problems.append(f"profile {profile.value}: unknown task ids {unknown}")
    for task in ctx.registry.tasks.values():
        missing = set(task.spec.allowed_commands) - set(ctx.bench.command_allowlist)
        if missing:
            problems.append(
                f"{task.id}: allowed_commands not in global allowlist: {sorted(missing)}"
            )
    if problems:
        for p in problems:
            typer.secho(f"  - {p}", fg=typer.colors.RED)
        raise typer.Exit(EXIT_PROBLEMS)
    typer.secho(
        f"OK: {len(ctx.models.models)} models, {len(ctx.models.all_harnesses())} harnesses, "
        f"{len(ctx.registry.tasks)} tasks, suite {ctx.bench.suite_version}",
        fg=typer.colors.GREEN,
    )


@app.command("list-models")
def list_models(config: ConfigOpt = None, models: ModelsOpt = None) -> None:
    """List configured model aliases and harnesses."""
    ctx = load_ctx(config, models, need_tasks=False)
    _table(
        [
            {
                "alias": m.alias,
                "provider": m.provider,
                "model_id": m.model_id,
                "endpoint": m.endpoint_type,
                "remote": m.remote,
                "runtime": m.serving_runtime,
                "quant": m.quantization,
                "ctx": m.context_limit,
                "key_env": m.api_key_env,
            }
            for m in ctx.models.models
        ],
        [
            "alias",
            "provider",
            "model_id",
            "endpoint",
            "remote",
            "runtime",
            "quant",
            "ctx",
            "key_env",
        ],
    )
    typer.echo("")
    _table(
        [
            {
                "harness": h.name,
                "kind": h.kind,
                "version": h.version,
                "remote": h.remote,
                "description": h.description,
            }
            for h in ctx.models.all_harnesses()
        ],
        ["harness", "kind", "version", "remote", "description"],
    )


@app.command("list-tasks")
def list_tasks(
    config: ConfigOpt = None,
    models: ModelsOpt = None,
    profile: Annotated[Profile | None, typer.Option(help="Only tasks in this profile.")] = None,
    domain: Annotated[str | None, typer.Option(help="Comma-separated domains.")] = None,
) -> None:
    """List tasks with domain, evaluation kind and runtime availability."""
    ctx = load_ctx(config, models)
    tasks = (
        ctx.registry.select(profile, ctx.bench.profiles[profile], domains=_csv(domain) or None)
        if profile
        else sorted(ctx.registry.tasks.values(), key=lambda t: t.id)
    )
    if domain and not profile:
        tasks = [t for t in tasks if t.spec.domain.value in _csv(domain)]
    rows = []
    for t in tasks:
        avail = check_availability(t)
        rows.append(
            {
                "id": t.id,
                "version": t.spec.version,
                "domain": t.spec.domain.value,
                "difficulty": t.spec.difficulty.value,
                "evaluation": t.spec.evaluation.value,
                "profiles": ",".join(p.value for p in t.spec.profiles),
                "est_s": t.spec.estimated_runtime_s,
                "available": "yes" if avail.available else f"no ({avail.reason})",
            }
        )
    _table(
        rows,
        ["id", "version", "domain", "difficulty", "evaluation", "profiles", "est_s", "available"],
    )


# ---------------------------------------------------------------------------------- doctor


@app.command("doctor")
def doctor(
    config: ConfigOpt = None,
    models: ModelsOpt = None,
    model: Annotated[str | None, typer.Option(help="Comma-separated aliases to probe.")] = None,
    check_endpoints: Annotated[
        bool, typer.Option(help="Probe endpoints (GET /models only).")
    ] = True,
) -> None:
    """Check runtime, tools, storage and endpoint reachability. Sends no prompts or code."""
    from engineering_llm_bench.environment import collect_environment  # noqa: PLC0415
    from engineering_llm_bench.providers.openai_compat import probe_endpoint  # noqa: PLC0415
    from engineering_llm_bench.tools.runner import isolation_method  # noqa: PLC0415

    ctx = load_ctx(config, models)
    env = collect_environment()
    ok = True
    typer.echo(f"python {env['python']} on {env['platform']}")
    for name, ver in {**env["tools"], **env["python_modules"]}.items():
        typer.echo(f"  {name:<10} {ver or 'not found'}")
    typer.echo(f"network isolation: {isolation_method()} [policy: {ctx.bench.isolation.network}]")
    try:
        store = ctx.store()
        typer.echo(
            f"database {ctx.db_path}: schema v{store.schema_version()}, "
            f"integrity {store.integrity_check()}"
        )
        store.close()
    except (SchemaVersionError, OSError) as exc:
        typer.secho(f"database problem: {exc}", fg=typer.colors.RED)
        ok = False
    unavailable = [
        (t.id, a.reason)
        for t in ctx.registry.tasks.values()
        if not (a := check_availability(t)).available
    ]
    typer.echo(
        f"tasks: {len(ctx.registry.tasks)} total, {len(unavailable)} unavailable (will be skipped)"
    )
    for tid, reason in unavailable:
        typer.echo(f"  - {tid}: {reason}")
    if check_endpoints:
        aliases = _csv(model) or [m.alias for m in ctx.models.models]
        for alias in aliases:
            try:
                cfg = ctx.models.model(alias)
            except KeyError as exc:
                raise _fail(str(exc.args[0])) from None
            if cfg.provider == "fake":
                typer.echo(f"  {alias:<16} mock provider (no endpoint)")
                continue
            if cfg.provider == "anthropic":
                key_set = bool(os.environ.get(cfg.api_key_env or "ANTHROPIC_API_KEY"))
                typer.echo(
                    f"  {alias:<16} remote API; key env {'set' if key_set else 'NOT set'}"
                    " (not probed)"
                )
                continue
            reachable, detail = probe_endpoint(cfg)
            color = typer.colors.GREEN if reachable else typer.colors.YELLOW
            typer.secho(f"  {alias:<16} {cfg.base_url} -> {detail}", fg=color)
    raise typer.Exit(0 if ok else EXIT_PROBLEMS)


# ---------------------------------------------------------------------------------- run


def _request(
    model: str,
    harness: str | None,
    profile: Profile,
    task: str | None,
    tag: str | None,
    domain: str | None,
    repeats: int | None,
    concurrency: int | None,
    allow_remote: bool,
    allow_external_harness: bool,
    resume: str | None,
    notes: str | None,
    default_harness: str,
) -> RunRequest:
    return RunRequest(
        model_alias=model,
        harness=harness or default_harness,
        profile=profile,
        task_ids=_csv(task),
        tags=_csv(tag),
        domains=_csv(domain),
        repeats=repeats,
        concurrency=concurrency,
        allow_remote=allow_remote,
        allow_external_harness=allow_external_harness,
        resume_run_id=resume,
        notes=notes,
    )


ModelOpt = Annotated[str, typer.Option("--model", help="Model alias (label for CLI harnesses).")]
HarnessOpt = Annotated[str | None, typer.Option(help="direct | agent | <cli harness name>.")]
ProfileOpt = Annotated[Profile, typer.Option(help="smoke | standard | extended.")]
TaskOpt = Annotated[str | None, typer.Option("--task", help="Comma-separated task ids.")]
TagOpt = Annotated[str | None, typer.Option(help="Comma-separated tags.")]
DomainOpt = Annotated[str | None, typer.Option(help="Comma-separated domains.")]


@app.command("dry-run")
def dry_run(
    model: ModelOpt,
    config: ConfigOpt = None,
    models: ModelsOpt = None,
    harness: HarnessOpt = None,
    profile: ProfileOpt = Profile.SMOKE,
    task: TaskOpt = None,
    tag: TagOpt = None,
    domain: DomainOpt = None,
    repeats: Annotated[int | None, typer.Option(min=1, max=20)] = None,
    allow_remote: Annotated[bool, typer.Option(help="Consent to remote providers.")] = False,
    allow_external_harness: bool = False,
) -> None:
    """Estimate a run (tasks, availability, time, request/token budget). Calls no provider."""
    ctx = load_ctx(config, models)
    runner = BenchmarkRunner(
        ctx.bench, ctx.models, ctx.registry, Store(Path(":memory:")), ctx.artifacts()
    )
    req = _request(
        model,
        harness,
        profile,
        task,
        tag,
        domain,
        repeats,
        None,
        allow_remote,
        allow_external_harness,
        None,
        None,
        ctx.bench.default_harness,
    )
    try:
        plan = runner.dry_run(req)
    except ConfigError as exc:
        raise _fail(str(exc)) from None
    _table(plan.rows, ["task", "domain", "evaluation", "available", "reason", "est_s", "timeout_s"])
    typer.echo(
        f"\nmodel={plan.run.model_label} harness={plan.run.harness_cfg.name} "
        f"mode={plan.run.mode.value} repeats={plan.run.repeats} "
        f"concurrency={plan.run.concurrency}"
    )
    typer.echo(
        f"attempts to execute: {plan.attempts}; "
        f"estimated wall time ~{plan.estimated_s / 60:.1f} min"
    )
    requests = "n/a" if plan.max_model_requests is None else plan.max_model_requests
    tokens = "unbounded" if plan.max_tokens is None else plan.max_tokens
    typer.echo(f"max model requests: {requests}; token budget: {tokens}")
    if plan.cost_upper_bound is not None:
        typer.echo(f"cost upper bound: {plan.cost_upper_bound:.2f}")
    for w in plan.warnings:
        typer.secho(f"WARNING: {w}", fg=typer.colors.YELLOW)


def _raise_interrupt(_sig: int, _frame: FrameType | None) -> None:
    raise KeyboardInterrupt


@app.command("run")
def run(
    model: ModelOpt,
    config: ConfigOpt = None,
    models: ModelsOpt = None,
    harness: HarnessOpt = None,
    profile: ProfileOpt = Profile.SMOKE,
    task: TaskOpt = None,
    tag: TagOpt = None,
    domain: DomainOpt = None,
    repeats: Annotated[int | None, typer.Option(min=1, max=20)] = None,
    concurrency: Annotated[int | None, typer.Option(min=1, max=32)] = None,
    allow_remote: Annotated[
        bool, typer.Option(help="Consent to sending prompts/code to remote providers.")
    ] = False,
    allow_external_harness: Annotated[
        bool, typer.Option(help="Enable external agent-CLI harnesses.")
    ] = False,
    resume: Annotated[str | None, typer.Option(help="Resume an interrupted run id.")] = None,
    notes: Annotated[str | None, typer.Option(help="Free-text note stored with the run.")] = None,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the extended-profile confirmation.")
    ] = False,
) -> None:
    """Run the benchmark for one model/harness and persist scored results."""
    ctx = load_ctx(config, models)
    store = ctx.store()
    runner = BenchmarkRunner(
        ctx.bench, ctx.models, ctx.registry, store, ctx.artifacts(), progress=typer.echo
    )
    req = _request(
        model,
        harness,
        profile,
        task,
        tag,
        domain,
        repeats,
        concurrency,
        allow_remote,
        allow_external_harness,
        resume,
        notes,
        ctx.bench.default_harness,
    )
    try:
        plan = runner.dry_run(req)
    except ConfigError as exc:
        raise _fail(str(exc)) from None
    for w in plan.warnings:
        typer.secho(f"WARNING: {w}", fg=typer.colors.YELLOW)
    if (profile == Profile.EXTENDED or plan.run.profile_cfg.warn_cost) and not yes:
        typer.confirm("Extended runs can take long and cost money. Continue?", abort=True)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        run_id = runner.run(req)
    except ConfigError as exc:
        raise _fail(str(exc)) from None
    run_row = store.get_run(run_id) or {}
    attempts = store.attempts_for_run(run_id)
    _table(
        attempts,
        ["task_id", "repeat_index", "status", "failure_class", "score", "passed", "agent_wall_s"],
    )
    typer.echo(f"\nrun id: {run_id}  status: {run_row.get('status')}")
    typer.echo(f"report: bench report --run-id {run_id} --format html")
    store.close()
    if run_row.get("status") == RunStatus.INTERRUPTED.value:
        typer.secho(
            f"interrupted; resume with: bench run --model {model} --profile "
            f"{profile.value} --resume {run_id}",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(EXIT_INTERRUPTED)
    if run_row.get("status") == RunStatus.FAILED.value:
        raise typer.Exit(EXIT_FAILED)


@app.command("runs")
def runs(
    config: ConfigOpt = None,
    models: ModelsOpt = None,
    limit: Annotated[int, typer.Option(min=1)] = 30,
) -> None:
    """List recorded runs (newest first)."""
    ctx = load_ctx(config, models, need_tasks=False)
    store = ctx.store()
    _table(
        store.list_runs(limit),
        [
            "id",
            "created_at",
            "status",
            "source",
            "model_alias",
            "harness",
            "mode",
            "profile",
            "task_count",
        ],
    )


@app.command("show")
def show(
    run_id: Annotated[str, typer.Option("--run-id")],
    config: ConfigOpt = None,
    models: ModelsOpt = None,
    checks: Annotated[bool, typer.Option(help="Also show evaluator checks.")] = False,
) -> None:
    """Show a run's attempts (and optionally evaluator checks)."""
    ctx = load_ctx(config, models, need_tasks=False)
    store = ctx.store()
    run_row = store.get_run(run_id)
    if run_row is None:
        raise _fail(f"unknown run id {run_id}")
    typer.echo(f"{run_id}: {run_row['model_alias']} [{run_row['harness']}] {run_row['status']}")
    attempts = store.attempts_for_run(run_id)
    _table(
        attempts,
        [
            "task_id",
            "repeat_index",
            "status",
            "failure_class",
            "score",
            "coverage",
            "passed",
            "agent_wall_s",
            "skip_reason",
        ],
    )
    if checks:
        for a in attempts:
            typer.echo(f"\n{a['task_id']} r{a['repeat_index']}")
            _table(
                store.check_results(a["id"]),
                ["check_id", "dimension", "status", "score", "message"],
            )


@app.command("compare")
def compare(
    run_ids: Annotated[str, typer.Option("--run-ids", help="Comma-separated run ids.")],
    config: ConfigOpt = None,
    models: ModelsOpt = None,
) -> None:
    """Side-by-side comparison of runs (aggregates, per task, comparability warnings)."""
    from engineering_llm_bench.reporting import ReportOptions, build_report  # noqa: PLC0415

    ctx = load_ctx(config, models, need_tasks=False)
    try:
        report = build_report(
            ctx.store(), ctx.artifacts(), ctx.redactor, ReportOptions(run_ids=_csv(run_ids))
        )
    except KeyError as exc:
        raise _fail(str(exc.args[0])) from None
    rows = []
    for key, g in report["groups"].items():
        rows.append(
            {
                "model|harness": key,
                "evaluated": g["evaluated"],
                "skipped": g["skipped"],
                "pass_rate": None if g["pass_rate"] is None else f"{g['pass_rate']:.0%}",
                "score_mean": None if g["score"]["mean"] is None else f"{g['score']['mean']:.1f}",
                "sd": None if g["score"]["stdev"] is None else f"{g['score']['stdev']:.1f}",
                "wall_p50": g["wall_time_s"]["median"] and round(g["wall_time_s"]["median"], 1),
                "timeouts": g["timeout_rate"],
                "tool_ok": g["tool_success_rate"] and round(g["tool_success_rate"], 2),
            }
        )
    _table(rows, list(rows[0]) if rows else ["model|harness"])
    typer.echo("")
    groups = list(report["groups"])
    matrix_rows = []
    for task_id, cells in report["task_matrix"].items():
        row: dict[str, Any] = {"task": task_id}
        for g in groups:
            c = cells.get(g)
            row[g] = (
                "-"
                if not c
                else (
                    ",".join(c["status"])
                    if c["mean"] is None
                    else f"{c['mean']:.0f} ({c['passed']}/{c['n']})"
                )
            )
        matrix_rows.append(row)
    _table(matrix_rows, ["task", *groups])
    for issue in report["comparability"]:
        typer.secho(f"NON-COMPARABLE: {issue}", fg=typer.colors.YELLOW)
    for key, g in report["groups"].items():
        for w in g["warnings"]:
            typer.secho(f"{key}: {w}", fg=typer.colors.YELLOW)


@app.command("report")
def report(
    config: ConfigOpt = None,
    models: ModelsOpt = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Comma-separated; default all runs.")
    ] = None,
    fmt: Annotated[str, typer.Option("--format", help="json | csv | html")] = "html",
    out: Annotated[
        Path | None, typer.Option(help="Output file (default reports/report.<fmt>).")
    ] = None,
    include_transcripts: Annotated[
        bool, typer.Option(help="Opt-in: include model traces.")
    ] = False,
    include_source: Annotated[bool, typer.Option(help="Opt-in: include diffs.")] = False,
) -> None:
    """Export JSON / CSV / static HTML report from persisted results (redacted)."""
    from engineering_llm_bench.reporting import (  # noqa: PLC0415
        ReportOptions,
        build_report,
        to_csv,
        to_html,
        to_json,
    )

    if fmt not in {"json", "csv", "html"}:
        raise _fail("--format must be json, csv or html")
    ctx = load_ctx(config, models, need_tasks=False)
    try:
        data = build_report(
            ctx.store(),
            ctx.artifacts(),
            ctx.redactor,
            ReportOptions(
                run_ids=_csv(run_id) or None,
                include_transcripts=include_transcripts,
                include_source=include_source,
            ),
        )
    except KeyError as exc:
        raise _fail(str(exc.args[0])) from None
    text = {"json": to_json, "csv": to_csv, "html": to_html}[fmt](data)
    target = out or Path("reports") / f"report.{fmt}"
    atomic_write_bytes(target, text.encode("utf-8"))
    typer.echo(f"wrote {target} ({len(data['attempts'])} attempts)")


@app.command("dashboard")
def dashboard(
    config: ConfigOpt = None,
    models: ModelsOpt = None,
    host: Annotated[str, typer.Option(help="Bind address (default loopback only).")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
) -> None:
    """Serve the local dashboard from the results database."""
    from engineering_llm_bench.dashboard.server import serve  # noqa: PLC0415

    ctx = load_ctx(config, models, need_tasks=False)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        typer.secho(
            "WARNING: binding beyond loopback exposes results to your network",
            fg=typer.colors.YELLOW,
        )
    serve(ctx.store(), ctx.artifacts(), ctx.redactor, host, port)


# ---------------------------------------------------------------------------------- import / export


@app.command("import-run")
def import_run(
    file: Annotated[Path, typer.Option("--file", help="Import JSON (see docs/provider-setup.md).")],
    config: ConfigOpt = None,
    models: ModelsOpt = None,
    evaluate: Annotated[
        bool, typer.Option(help="Apply patches and run the task evaluators.")
    ] = True,
    keep_transcripts: Annotated[bool, typer.Option(help="Store transcripts (opt-in).")] = False,
) -> None:
    """Import a manually captured run (e.g. Copilot BYOK). Imported commands are never executed."""
    from engineering_llm_bench.importing import RunImporter  # noqa: PLC0415

    ctx = load_ctx(config, models)
    importer = RunImporter(ctx.bench, ctx.registry, ctx.store(), ctx.artifacts())
    try:
        result = importer.import_file(
            file, evaluate_patches=evaluate, keep_transcripts=keep_transcripts
        )
    except ConfigError as exc:
        raise _fail(str(exc)) from None
    for w in result.warnings:
        typer.secho(f"WARNING: {w}", fg=typer.colors.YELLOW)
    typer.echo(
        f"imported run {result.run_id}: {result.attempts} attempt(s), {result.evaluated} evaluated"
    )


@app.command("export-run")
def export_run(
    run_id: Annotated[str, typer.Option("--run-id")],
    out: Annotated[Path, typer.Option(help="Output JSON in import format.")],
    config: ConfigOpt = None,
    models: ModelsOpt = None,
) -> None:
    """Export a run in the import format (round-trip; unknown fields stay null)."""
    from engineering_llm_bench.importing import export_import_format  # noqa: PLC0415

    ctx = load_ctx(config, models, need_tasks=False)
    try:
        doc = export_import_format(ctx.store(), ctx.artifacts(), run_id)
    except ConfigError as exc:
        raise _fail(str(exc)) from None
    atomic_write_bytes(out, json.dumps(ctx.redactor.obj(doc), indent=2).encode())
    typer.echo(f"wrote {out}")


# ---------------------------------------------------------------------------------- tasks


@task_app.command("validate")
def task_validate(
    path: Annotated[Path, typer.Argument(help="task.yaml or task directory.")],
    config: ConfigOpt = None,
    models: ModelsOpt = None,
    run_evaluators: Annotated[
        bool, typer.Option(help="Check fixture fails and reference passes.")
    ] = True,
) -> None:
    """Validate a task: schema, and that evaluators reject the fixture and accept the reference."""
    from engineering_llm_bench.catalog.validation import validate_task_behaviour  # noqa: PLC0415

    ctx = load_ctx(config, models, need_tasks=False)
    target = path / "task.yaml" if path.is_dir() else path
    try:
        task = load_task(target)
    except ConfigError as exc:
        raise _fail(str(exc), EXIT_PROBLEMS) from None
    typer.secho(
        f"schema OK: {task.id} v{task.spec.version} ({len(task.spec.checks)} checks)",
        fg=typer.colors.GREEN,
    )
    if not run_evaluators:
        return
    report = validate_task_behaviour(task, ctx.bench)
    for line in report.lines:
        typer.echo(line)
    if report.skipped:
        typer.secho(f"SKIPPED: {report.skipped}", fg=typer.colors.YELLOW)
        return
    if not report.ok:
        raise _fail("task validation FAILED", EXIT_PROBLEMS)
    typer.secho("behaviour OK", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------------- reviews


@review_app.command("export")
def review_export(
    run_id: Annotated[str, typer.Option("--run-id", help="Comma-separated run ids.")],
    out: Annotated[Path, typer.Option(help="Output directory for packet/key/template.")],
    dimension: Annotated[
        str, typer.Option(help="Comma-separated dimensions.")
    ] = "explanation,quality",
    config: ConfigOpt = None,
    models: ModelsOpt = None,
) -> None:
    """Write a blind review packet (model identity removed) and a separate key file."""
    from engineering_llm_bench.evaluation.review import export_blind_packet  # noqa: PLC0415

    ctx = load_ctx(config, models)
    dims = [Dimension(d) for d in _csv(dimension)]
    packet, key = export_blind_packet(
        ctx.store(), ctx.artifacts(), ctx.registry, _csv(run_id), out, dims
    )
    typer.echo(
        f"packet: {packet}\nkey (keep private): {key}\ntemplate: {out / 'reviews_template.csv'}"
    )


@review_app.command("import")
def review_import(
    key: Annotated[Path, typer.Option(help="key.json from `review export`.")],
    file: Annotated[
        Path, typer.Option("--file", help="CSV: blind_id,reviewer,dimension,score,rationale")
    ],
    config: ConfigOpt = None,
    models: ModelsOpt = None,
) -> None:
    """Record human review scores (0-1) from a CSV."""
    from engineering_llm_bench.evaluation.review import import_reviews  # noqa: PLC0415

    ctx = load_ctx(config, models, need_tasks=False)
    result = import_reviews(ctx.store(), key, file)
    for e in result.errors:
        typer.secho(e, fg=typer.colors.RED)
    typer.echo(f"recorded {result.recorded} review score(s)")
    if result.errors:
        raise typer.Exit(EXIT_PROBLEMS)


@review_app.command("judge")
def review_judge(
    run_id: Annotated[str, typer.Option("--run-id")],
    judge_model: Annotated[str, typer.Option("--judge-model", help="Model alias used as judge.")],
    dimension: Annotated[str, typer.Option()] = "explanation",
    allow_remote: bool = False,
    config: ConfigOpt = None,
    models: ModelsOpt = None,
) -> None:
    """Score a rubric dimension with a judge model (stored separately as source=judge)."""
    from engineering_llm_bench.evaluation.review import judge_attempts  # noqa: PLC0415
    from engineering_llm_bench.providers import create_provider  # noqa: PLC0415

    ctx = load_ctx(config, models)
    try:
        cfg = ctx.models.model(judge_model)
    except KeyError as exc:
        raise _fail(str(exc.args[0])) from None
    if cfg.remote and not (allow_remote or ctx.bench.allow_remote_providers):
        raise _fail("judge model is remote: pass --allow-remote to send diffs/summaries to it")
    result = judge_attempts(
        ctx.store(),
        ctx.artifacts(),
        ctx.registry,
        create_provider(cfg),
        run_id,
        Dimension(dimension),
    )
    for e in result.errors:
        typer.secho(e, fg=typer.colors.YELLOW)
    typer.echo(f"recorded {result.recorded} judge score(s) for {dimension}")


# ---------------------------------------------------------------------------------- BYOK sessions


def _default_session_dir() -> Path:
    return Path.home() / "elb-byok-sessions" / time.strftime("session-%Y%m%d-%H%M%S")


@byok_app.command("prepare")
def byok_prepare(
    out: Annotated[
        Path | None,
        typer.Option(help="Session folder to create (default ~/elb-byok-sessions/session-<time>)."),
    ] = None,
    profile: ProfileOpt = Profile.STANDARD,
    task: TaskOpt = None,
    allow_inside_repo: Annotated[
        bool, typer.Option(help="Testing only: allow a session folder inside this repository.")
    ] = False,
    config: ConfigOpt = None,
    models: ModelsOpt = None,
) -> None:
    """Create a session folder with starting_point.md and one folder per task for an IDE agent."""
    from engineering_llm_bench.byok import prepare_session  # noqa: PLC0415

    ctx = load_ctx(config, models)
    target = (out or _default_session_dir()).expanduser()
    try:
        session = prepare_session(
            ctx.registry,
            ctx.bench,
            ctx.artifacts_dir,
            target,
            profile,
            _csv(task),
            allow_inside_repo=allow_inside_repo,
        )
    except ConfigError as exc:
        raise _fail(str(exc)) from None
    typer.secho(
        f"session {session.session_id}: {len(session.tasks)} task(s)", fg=typer.colors.GREEN
    )
    typer.echo(f"  folder:         {session.path}")
    typer.echo(f"  starting point: {session.starting_point}")
    typer.echo("Next steps:")
    typer.echo("  1. Open ONLY the session folder in VS Code (File > Open Folder), not this repo.")
    typer.echo("  2. In Copilot Chat choose Agent mode and your BYOK model, attach")
    typer.echo("     starting_point.md and say: 'Follow starting_point.md and complete all tasks.'")
    typer.echo("  3. When it is done, collect and score the results:")
    typer.echo(f'     python byok/collect_results.py "{session.path}" --model-label "<your model>"')


@byok_app.command("collect")
def byok_collect(
    session_dir: Annotated[Path, typer.Argument(help="Session folder created by `byok prepare`.")],
    model_label: Annotated[
        str, typer.Option(help="Label for the model in the dashboard, e.g. 'strata/qwen-3.8'.")
    ],
    harness_label: Annotated[str, typer.Option(help="Harness label.")] = "copilot-byok",
    copilot_version: Annotated[
        str | None, typer.Option(help="Copilot extension / VS Code version (recorded).")
    ] = None,
    provider_label: Annotated[
        str | None, typer.Option(help="Optional serving label, e.g. 'ollama' (recorded only).")
    ] = None,
    remote: Annotated[
        bool | None, typer.Option("--remote/--local", help="Was the model remote? (recorded)")
    ] = None,
    session_minutes: Annotated[
        float | None, typer.Option(min=0, help="Total time the agent took, if you measured it.")
    ] = None,
    notes: Annotated[str | None, typer.Option(help="Free-text note stored with the run.")] = None,
    only_attempted: Annotated[
        bool,
        typer.Option(help="Mark untouched tasks as skipped instead of evaluating them as failed."),
    ] = False,
    dry_run: Annotated[bool, typer.Option(help="Validate and show findings only.")] = False,
    force: Annotated[bool, typer.Option(help="Allow collecting the same session again.")] = False,
    report: Annotated[Path | None, typer.Option(help="Also write an HTML report here.")] = None,
    serve: Annotated[bool, typer.Option(help="Start the dashboard after collecting.")] = False,
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
    config: ConfigOpt = None,
    models: ModelsOpt = None,
) -> None:
    """Validate an agent's session output, run the hidden evaluators and update the dashboard."""
    from engineering_llm_bench.byok import collect_session  # noqa: PLC0415

    ctx = load_ctx(config, models)
    store = None if dry_run else ctx.store()
    artifacts = None if dry_run else ctx.artifacts()
    try:
        result = collect_session(
            session_dir.expanduser(),
            ctx.registry,
            ctx.bench,
            store,
            artifacts,
            ctx.artifacts_dir,
            model_label=model_label,
            harness_label=harness_label,
            harness_version=copilot_version,
            provider_label=provider_label,
            remote=remote,
            session_minutes=session_minutes,
            notes=notes,
            only_attempted=only_attempted,
            dry_run=dry_run,
            force=force,
        )
    except ConfigError as exc:
        raise _fail(str(exc)) from None
    typer.echo(f"session {result.session_id}: validation")
    _table(
        [
            {
                "task": s.task_id,
                "attempted": "yes" if s.attempted else "no",
                "files_changed": len(s.overlay),
                "result_md": "yes" if s.result_md is not None else "MISSING",
                "self_status": s.self_reported_status,
                "rejected_edits": len(s.rejected),
            }
            for s in result.submissions
        ],
        ["task", "attempted", "files_changed", "result_md", "self_status", "rejected_edits"],
    )
    for s in result.submissions:
        for n in s.notes:
            typer.secho(f"  {s.task_id}: {n}", fg=typer.colors.YELLOW)
    for w in result.warnings:
        typer.secho(f"WARNING: {w}", fg=typer.colors.YELLOW)
    if dry_run or result.run_id is None or store is None:
        typer.echo("dry run: nothing was evaluated or stored")
        return
    typer.echo(f"\nscored run {result.run_id} ({model_label} via {harness_label}):")
    attempts = store.attempts_for_run(result.run_id)
    _table(attempts, ["task_id", "status", "passed", "score", "failure_class", "skip_reason"])
    passed = sum(1 for a in attempts if a["passed"] == 1)
    evaluated = sum(1 for a in attempts if a["passed"] is not None)
    typer.secho(f"passed {passed}/{evaluated} evaluated task(s)", fg=typer.colors.GREEN)
    if report is not None:
        from engineering_llm_bench.reporting import ReportOptions, build_report, to_html  # noqa: PLC0415

        data = build_report(store, ctx.artifacts(), ctx.redactor, ReportOptions())
        atomic_write_bytes(report, to_html(data).encode("utf-8"))
        typer.echo(f"wrote {report}")
    if serve:
        from engineering_llm_bench.dashboard.server import serve as serve_dashboard  # noqa: PLC0415

        serve_dashboard(store, ctx.artifacts(), ctx.redactor, "127.0.0.1", port)
    else:
        typer.echo("dashboard: `bench dashboard` (or refresh it if already running)")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        typer.secho("interrupted", fg=typer.colors.YELLOW, err=True)
        sys.exit(EXIT_INTERRUPTED)


if __name__ == "__main__":  # pragma: no cover
    main()
