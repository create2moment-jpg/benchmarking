"""Evaluator check implementations.

Every check runs *after* the harness finishes, in a fresh evaluation copy of the workspace with
hidden evaluator files under ``_hidden/``. Checks never trust the model's self-reported results.

Status semantics:

* ``pass`` / ``partial`` / ``fail`` - the candidate was evaluated; ``score`` is in [0, 1];
* ``error``  - the *evaluator* could not run (e.g. tool missing); excluded from scoring and
  surfaced in reports, never silently counted as a model failure;
* ``skipped`` - not applicable (e.g. optional runtime missing); excluded from scoring.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sqlite3
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engineering_llm_bench.catalog.registry import Task
from engineering_llm_bench.catalog.schema import (
    AddedTestsCheck,
    CheckSpec,
    CommandCheck,
    FilesExistCheck,
    PytestCheck,
    RuffCheck,
    SqlResultCheck,
    SummaryCheck,
    TestsDetectBugCheck,
    TextRubricCheck,
)
from engineering_llm_bench.config.models import LimitsConfig
from engineering_llm_bench.control import CancelToken
from engineering_llm_bench.domain import CheckStatus, Dimension, ScoreSource
from engineering_llm_bench.tools.runner import CommandRefusedError, CommandResult, CommandRunner
from engineering_llm_bench.workspace.manager import HIDDEN_DIRNAME, Workspace, iter_files

_MAX_JUNIT_BYTES = 5_000_000
_MAX_SQL_ROWS = 10_000
_CHECK_LINE = re.compile(r"^CHECK\s+(\S+)\s+(PASS|FAIL)\s*$", re.MULTILINE)


@dataclass
class CheckResult:
    check_id: str
    kind: str
    dimension: Dimension
    weight: float
    status: CheckStatus
    score: float | None
    source: ScoreSource
    regression_guard: bool = False
    message: str = ""
    duration_s: float = 0.0
    evidence: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def scored(self) -> bool:
        return (
            self.status not in (CheckStatus.ERROR, CheckStatus.SKIPPED) and self.score is not None
        )


@dataclass
class EvalContext:
    task: Task
    workspace: Workspace
    eval_dir: Path
    runner: CommandRunner
    limits: LimitsConfig
    cancel: CancelToken
    allowlist: set[str]
    summary: str
    memory_mb: int | None = -1
    _baseline_dir: Path | None = None

    def baseline_dir(self) -> Path:
        if self._baseline_dir is None:
            self._baseline_dir = self.workspace.make_baseline_copy()
        return self._baseline_dir

    def run(self, argv: list[str], cwd: Path, timeout_s: float | None) -> CommandResult:
        return self.runner.run(
            argv,
            cwd=cwd,
            root=cwd,
            scratch=self.workspace.scratch,
            allowed=self.allowlist,
            timeout_s=timeout_s,
            cancel=self.cancel,
            memory_mb=self.memory_mb,
        )


def _status(score: float) -> CheckStatus:
    if score >= 1.0 - 1e-9:
        return CheckStatus.PASS
    return CheckStatus.PARTIAL if score > 0 else CheckStatus.FAIL


def _result(
    spec: CheckSpec, score: float, message: str, evidence: str = "", **details: Any
) -> CheckResult:
    score = max(0.0, min(1.0, score))
    return CheckResult(
        check_id=spec.id,
        kind=spec.kind,
        dimension=spec.dimension,
        weight=spec.weight,
        status=_status(score),
        score=score,
        source=spec.source,
        regression_guard=spec.regression_guard,
        message=message,
        evidence=evidence,
        details=details,
    )


def _error(spec: CheckSpec, message: str, status: CheckStatus = CheckStatus.ERROR) -> CheckResult:
    return CheckResult(
        check_id=spec.id,
        kind=spec.kind,
        dimension=spec.dimension,
        weight=spec.weight,
        status=status,
        score=None,
        source=spec.source,
        regression_guard=spec.regression_guard,
        message=message,
    )


# --------------------------------------------------------------------------- pytest


@dataclass(frozen=True)
class PytestOutcome:
    total: int
    passed: int
    failed: int
    errors: int
    skipped: int
    exit_code: int | None
    log: str
    timed_out: bool


def run_pytest(
    ctx: EvalContext, cwd: Path, targets: list[str], tag: str, timeout: float | None
) -> PytestOutcome:
    junit = f".elb_junit_{tag}.xml"
    argv = ["pytest", "-q", "-p", "no:cacheprovider", "--color=no", f"--junitxml={junit}", *targets]
    res = ctx.run(argv, cwd, timeout)
    junit_path = cwd / junit
    total = failed = errors = skipped = 0
    if junit_path.is_file() and junit_path.stat().st_size <= _MAX_JUNIT_BYTES:
        try:
            root = ET.fromstring(junit_path.read_bytes())  # noqa: S314
            suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
            for suite in suites:
                total += int(suite.get("tests", 0))
                failed += int(suite.get("failures", 0))
                errors += int(suite.get("errors", 0))
                skipped += int(suite.get("skipped", 0))
        except (ET.ParseError, ValueError):
            total = 0
        junit_path.unlink(missing_ok=True)
    passed = max(0, total - failed - errors - skipped)
    # Consistency guard: a non-zero exit with "all passed" junit means tampering or a crash.
    if res.exit_code != 0 and passed == total - skipped and total > 0:
        passed = 0
        failed = total - skipped
    return PytestOutcome(
        total, passed, failed, errors, skipped, res.exit_code, res.summary(20_000), res.timed_out
    )


def check_pytest(spec: PytestCheck, ctx: EvalContext) -> CheckResult:
    missing = [t for t in spec.targets if not (ctx.eval_dir / t).exists()]
    if missing:
        return _result(spec, 0.0, f"test targets missing: {missing}")
    out = run_pytest(ctx, ctx.eval_dir, spec.targets, spec.id, spec.timeout_s)
    if out.timed_out:
        return _result(spec, 0.0, "pytest timed out", out.log)
    runnable = out.total - out.skipped
    if runnable <= 0:
        # Usually a collection error caused by the candidate (syntax/import error).
        return _result(spec, 0.0, f"no tests ran (exit={out.exit_code})", out.log)
    score = out.passed / runnable if spec.partial_credit else float(out.passed == runnable)
    msg = f"{out.passed}/{runnable} passed, {out.failed} failed, {out.errors} errors"
    return _result(
        spec, score, msg, out.log, total=out.total, passed=out.passed, skipped=out.skipped
    )


# --------------------------------------------------------------------------- command


def expand_argv(argv: list[str]) -> list[str]:
    return [
        a.replace("{python}", "python")
        .replace("{hidden}", HIDDEN_DIRNAME)
        .replace("{workspace}", ".")
        for a in argv
    ]


def check_command(spec: CommandCheck, ctx: EvalContext) -> CheckResult:
    argv = expand_argv(spec.argv)
    try:
        res = ctx.run(argv, ctx.eval_dir, spec.timeout_s)
    except CommandRefusedError as exc:
        return _error(spec, f"evaluator command refused: {exc}")
    log = res.summary(20_000)
    if res.timed_out:
        return _result(spec, 0.0, "command timed out", log)
    if spec.parse == "exit":
        ok = res.exit_code == spec.expect_exit
        return _result(spec, float(ok), f"exit={res.exit_code} (expected {spec.expect_exit})", log)
    outcomes: dict[str, bool] = {}
    for name, verdict in _CHECK_LINE.findall(res.stdout):
        outcomes[name] = outcomes.get(name, True) and verdict == "PASS"
    if not outcomes:
        return _result(spec, 0.0, f"no CHECK lines (exit={res.exit_code})", log)
    passed = sum(outcomes.values())
    score = passed / len(outcomes)
    if res.exit_code != spec.expect_exit:
        score = min(score, 0.5 * score)  # crashed part-way; do not grant full credit
    return _result(
        spec,
        score,
        f"{passed}/{len(outcomes)} checks passed, exit={res.exit_code}",
        log,
        checks=outcomes,
    )


# --------------------------------------------------------------------------- ruff


def check_ruff(spec: RuffCheck, ctx: EvalContext) -> CheckResult:
    targets = [t for t in spec.targets if (ctx.eval_dir / t).exists()]
    if not targets:
        return _result(spec, 0.0, "no lint targets exist")
    argv = [
        "ruff",
        "check",
        "--isolated",
        "--no-cache",
        "--output-format",
        "json",
        "--select",
        ",".join(spec.select),
        *targets,
    ]
    try:
        res = ctx.run(argv, ctx.eval_dir, spec.timeout_s)
    except CommandRefusedError as exc:
        return _error(spec, f"ruff unavailable: {exc}", CheckStatus.SKIPPED)
    try:
        findings = json.loads(res.stdout or "[]")
    except json.JSONDecodeError:
        return _error(spec, f"could not parse ruff output (exit={res.exit_code})")
    count = len(findings) if isinstance(findings, list) else 0
    codes = Counter(str(f.get("code")) for f in findings if isinstance(f, dict))
    evidence = "\n".join(
        f"{Path(str(f.get('filename', ''))).name}:{(f.get('location') or {}).get('row')}: "
        f"{f.get('code')} {f.get('message')}"
        for f in findings
        if isinstance(f, dict)
    )
    return _result(
        spec, 1.0 / (1.0 + 0.25 * count), f"{count} lint findings", evidence, codes=dict(codes)
    )


# --------------------------------------------------------------------------- SQL


def _norm(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return repr(round(value, 9))
    return str(value).strip()


def _cell_equal(a: str, b: str, tol: float) -> bool:
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False
    return math.isclose(fa, fb, rel_tol=tol, abs_tol=tol)


def _rows_match(a: tuple[str, ...], b: tuple[str, ...], tol: float) -> bool:
    return len(a) == len(b) and all(_cell_equal(x, y, tol) for x, y in zip(a, b, strict=True))


def _sqlite_authorizer(action: int, *_: Any) -> int:
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def run_sqlite(setup: str, query: str, timeout_s: float) -> tuple[list[str], list[tuple[Any, ...]]]:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(setup)
        conn.set_authorizer(_sqlite_authorizer)  # candidate query: read-only SELECT only
        deadline = time.monotonic() + timeout_s
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        cur = conn.execute(query)
        cols = [d[0] for d in cur.description or []]
        return cols, cur.fetchmany(_MAX_SQL_ROWS + 1)
    finally:
        conn.close()


def run_duckdb(setup: str, query: str, timeout_s: float) -> tuple[list[str], list[tuple[Any, ...]]]:
    import duckdb  # noqa: PLC0415 - optional runtime

    conn = duckdb.connect(":memory:")
    timer = threading.Timer(timeout_s, conn.interrupt)
    try:
        conn.execute(setup)
        conn.execute("SET enable_external_access = false")
        conn.execute("SET lock_configuration = true")
        timer.start()
        cur = conn.execute(query)
        cols = [d[0] for d in cur.description or []]
        return cols, cur.fetchmany(_MAX_SQL_ROWS + 1)
    finally:
        timer.cancel()
        conn.close()


def check_sql(spec: SqlResultCheck, ctx: EvalContext) -> CheckResult:
    query_path = ctx.eval_dir / spec.query_file
    if not query_path.is_file():
        return _result(spec, 0.0, f"query file {spec.query_file} missing")
    query = query_path.read_text(encoding="utf-8").strip().rstrip(";").strip()
    if not query:
        return _result(spec, 0.0, "query file is empty")
    hidden = ctx.eval_dir / HIDDEN_DIRNAME
    setup = (hidden / spec.setup_sql).read_text(encoding="utf-8")
    with (hidden / spec.expected_csv).open(newline="", encoding="utf-8") as fh:
        reader = list(csv.reader(fh))
    exp_cols, expected = [c.strip().lower() for c in reader[0]], [tuple(r) for r in reader[1:]]
    timeout = spec.timeout_s or 30.0
    try:
        if spec.engine == "duckdb":
            cols, rows = run_duckdb(setup, query, timeout)
        else:
            cols, rows = run_sqlite(setup, query, timeout)
    except ImportError:
        return _error(spec, "duckdb not installed", CheckStatus.SKIPPED)
    except Exception as exc:  # candidate SQL error: evaluated as a failure
        return _result(spec, 0.0, f"query failed: {type(exc).__name__}: {exc}", query)
    if len(rows) > _MAX_SQL_ROWS:
        return _result(spec, 0.0, f"query returned more than {_MAX_SQL_ROWS} rows")
    actual = [tuple(_norm(v) for v in r) for r in rows]
    got_cols = [c.lower() for c in cols]
    preview = "\n".join([",".join(cols), *(",".join(r) for r in actual[:50])])
    if got_cols != exp_cols:
        return _result(spec, 0.0, f"columns {got_cols} != expected {exp_cols}", preview)
    tol = spec.float_tolerance
    remaining = list(actual)
    matched = 0
    for exp in expected:
        for i, act in enumerate(remaining):
            if _rows_match(exp, act, tol):
                matched += 1
                del remaining[i]
                break
    denom = max(len(expected), len(actual)) or 1
    score = matched / denom
    if score >= 1.0 and spec.ordered:
        in_order = all(_rows_match(e, a, tol) for e, a in zip(expected, actual, strict=True))
        if not in_order:
            return _result(spec, 0.5, "correct rows in wrong order", preview)
    return _result(
        spec,
        score,
        f"{matched}/{len(expected)} expected rows matched; {len(actual)} returned",
        preview,
    )


# --------------------------------------------------------------------------- regression tests


def check_tests_detect_bug(spec: TestsDetectBugCheck, ctx: EvalContext) -> CheckResult:
    missing = [t for t in spec.test_targets if not (ctx.eval_dir / t).exists()]
    if missing:
        return _result(spec, 0.0, f"test targets missing: {missing}")
    # Separate copy: the pristine baseline (used by added_tests) must stay untouched.
    baseline = ctx.workspace.make_baseline_copy(f"baseline-{spec.id}")
    for target in spec.test_targets:  # candidate's tests on top of the ORIGINAL code
        src = ctx.eval_dir / target
        dst = baseline / target
        if src.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
            dst.mkdir(parents=True, exist_ok=True)
            for f in iter_files(src):
                out = dst / f.relative_to(src)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, out)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    on_original = run_pytest(ctx, baseline, spec.test_targets, f"{spec.id}_orig", spec.timeout_s)
    on_final = run_pytest(ctx, ctx.eval_dir, spec.test_targets, f"{spec.id}_final", spec.timeout_s)
    detects = on_original.failed + on_original.errors > 0 and on_original.total > 0
    passes = on_final.total > 0 and on_final.passed == on_final.total - on_final.skipped
    evidence = f"== on original code ==\n{on_original.log}\n\n== on final code ==\n{on_final.log}"
    msg = f"fails on original: {detects}; passes on final: {passes}"
    return _result(spec, float(detects and passes), msg, evidence)


def _count_tests(root: Path, glob: str, pattern: re.Pattern[str]) -> int:
    count = 0
    for path in root.glob(glob):
        if path.is_file() and HIDDEN_DIRNAME not in path.relative_to(root).parts:
            count += len(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))
    return count


def check_added_tests(spec: AddedTestsCheck, ctx: EvalContext) -> CheckResult:
    pattern = re.compile(spec.pattern, re.MULTILINE)
    before = _count_tests(ctx.baseline_dir(), spec.glob, pattern)
    after = _count_tests(ctx.eval_dir, spec.glob, pattern)
    added = after - before
    return _result(
        spec,
        min(1.0, max(0, added) / spec.min_added),
        f"{added} test(s) added ({before} -> {after})",
    )


# --------------------------------------------------------------- text rubric / summary / files


def check_text_rubric(spec: TextRubricCheck, ctx: EvalContext) -> CheckResult:
    path = ctx.eval_dir / spec.file
    if not path.is_file():
        return _result(spec, 0.0, f"{spec.file} not found")
    text = path.read_bytes()[: spec.max_bytes].decode("utf-8", errors="replace")
    total = sum(c.weight for c in spec.criteria)
    got = 0.0
    lines: list[str] = []
    met_ids: list[str] = []
    for c in spec.criteria:
        any_ok = not c.any_of or any(
            re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in c.any_of
        )
        all_ok = all(re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in c.all_of)
        none_ok = not any(re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in c.none_of)
        met = any_ok and all_ok and none_ok
        got += c.weight if met else 0.0
        if met:
            met_ids.append(c.id)
        lines.append(f"[{'x' if met else ' '}] {c.id} (w={c.weight}): {c.description}")
    return _result(
        spec,
        got / total,
        f"{len(met_ids)}/{len(spec.criteria)} criteria met",
        "\n".join(lines),
        met=met_ids,
    )


def check_summary(spec: SummaryCheck, ctx: EvalContext) -> CheckResult:
    summary = ctx.summary.strip()
    if not summary:
        return _result(spec, 0.0, "no final summary")
    words = len(summary.split())
    parts = {
        "present": 0.4,
        "concise": 0.2 if words <= spec.max_words else 0.0,
        "verification": 0.2
        if re.search(r"\b(test|tested|verified|ran|pytest|ctest)\b", summary, re.I)
        else 0.0,
        "limitations": 0.2
        if re.search(r"\b(limitation|assum|caveat|not handled|todo|risk)", summary, re.I)
        else 0.0,
    }
    return _result(spec, sum(parts.values()), f"{words} words; components={parts}", summary[:2000])


def check_files_exist(spec: FilesExistCheck, ctx: EvalContext) -> CheckResult:
    present = [p for p in spec.paths if (ctx.eval_dir / p).is_file()]
    return _result(
        spec, len(present) / len(spec.paths), f"{len(present)}/{len(spec.paths)} files present"
    )


CHECKS: dict[str, Callable[[Any, EvalContext], CheckResult]] = {
    "pytest": check_pytest,
    "command": check_command,
    "ruff": check_ruff,
    "sql_result": check_sql,
    "tests_detect_bug": check_tests_detect_bug,
    "added_tests": check_added_tests,
    "text_rubric": check_text_rubric,
    "summary": check_summary,
    "files_exist": check_files_exist,
}


def run_check(spec: CheckSpec, ctx: EvalContext) -> CheckResult:
    """Run one check, converting evaluator exceptions into ``error`` results."""
    start = time.monotonic()
    try:
        result = CHECKS[spec.kind](spec, ctx)
    except CommandRefusedError as exc:
        result = _error(spec, f"evaluator command refused: {exc}")
    except Exception as exc:  # evaluator bug or broken task: surface, never crash the run
        result = _error(spec, f"evaluator error: {type(exc).__name__}: {exc}")
    result.duration_s = time.monotonic() - start
    return result
