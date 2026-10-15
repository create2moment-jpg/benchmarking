"""Aggregate statistics over persisted attempts.

Rules: skipped attempts never count as pass or fail; usage/cost are summed only from attempts
that reported them and the coverage (n of N) is shown; small samples carry warnings; groups
whose task versions or environments differ are flagged as non-comparable.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from engineering_llm_bench.domain import AttemptStatus, FailureClass

SMALL_SAMPLE = 5


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 100]); None for empty input."""
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def group_key(a: dict[str, Any]) -> str:
    return f"{a['model_alias']} | {a['harness']}"


def summarize_group(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics for one (model, harness) group."""
    executed = [a for a in attempts if a["status"] != AttemptStatus.SKIPPED.value]
    evaluated = [a for a in executed if a["status"] == AttemptStatus.COMPLETED.value]
    with_pass = [a for a in evaluated if a["passed"] is not None]
    scores = [float(a["score"]) for a in evaluated if a["score"] is not None]
    walls = [float(a["agent_wall_s"]) for a in executed if a["agent_wall_s"] is not None]
    failures: dict[str, int] = defaultdict(int)
    for a in executed:
        failures[a["failure_class"] or "unknown"] += 1
    tool_calls = sum(a["tool_calls"] or 0 for a in executed)
    tool_ok = sum(a["tool_calls_ok"] or 0 for a in executed)
    guarded = [a for a in evaluated if a["regression_failed"] is not None]
    usage_rows = [a for a in executed if a["usage_reliable"] and a["input_tokens"] is not None]
    cost_rows = [a for a in executed if a["cost_usd"] is not None]
    domains: dict[str, list[float]] = defaultdict(list)
    for a in evaluated:
        if a["score"] is not None:
            domains[a["task_domain"] or "unknown"].append(float(a["score"]))
    warnings = []
    if len(scores) < SMALL_SAMPLE:
        warnings.append(f"small sample (n={len(scores)}): do not rank on this alone")
    if any(a["provider"] == "fake" for a in attempts):
        warnings.append("MOCK provider: scores validate the pipeline, not a model")
    return {
        "attempts": len(attempts),
        "executed": len(executed),
        "evaluated": len(evaluated),
        "skipped": len(attempts) - len(executed),
        "interrupted": sum(a["status"] == AttemptStatus.INTERRUPTED.value for a in attempts),
        "infra_failed": sum(a["status"] == AttemptStatus.FAILED.value for a in attempts),
        "passed": sum(bool(a["passed"]) for a in with_pass),
        "pass_rate": _rate(sum(bool(a["passed"]) for a in with_pass), len(with_pass)),
        "score": describe(scores),
        "wall_time_s": {
            "median": percentile(walls, 50),
            "p90": percentile(walls, 90),
            "n": len(walls),
        },
        "timeout_rate": _rate(failures.get(FailureClass.TIMEOUT.value, 0), len(executed)),
        "failure_rate": _rate(
            sum(v for k, v in failures.items() if k != FailureClass.NONE.value), len(executed)
        ),
        "failure_classes": dict(sorted(failures.items())),
        "tool_success_rate": _rate(tool_ok, tool_calls),
        "tool_calls": tool_calls,
        "regression_rate": _rate(sum(bool(a["regression_failed"]) for a in guarded), len(guarded)),
        "malformed_outputs": sum(a["malformed_outputs"] or 0 for a in executed),
        "usage": {
            "input_tokens": sum(a["input_tokens"] or 0 for a in usage_rows) if usage_rows else None,
            "output_tokens": sum(a["output_tokens"] or 0 for a in usage_rows)
            if usage_rows
            else None,
            "reported_attempts": len(usage_rows),
            "of": len(executed),
        },
        "cost_usd": {
            "total": sum(a["cost_usd"] for a in cost_rows) if cost_rows else None,
            "reported_attempts": len(cost_rows),
            "of": len(executed),
        },
        "domains": {d: describe(v) for d, v in sorted(domains.items())},
        "warnings": warnings,
    }


def dimension_means(
    store_rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    by_dim: dict[str, list[float]] = defaultdict(list)
    for r in store_rows:
        if r["score"] is not None:
            by_dim[r["dimension"]].append(float(r["score"]))
    return {d: describe(v) for d, v in sorted(by_dim.items())}


def comparability(attempts: list[dict[str, Any]]) -> list[str]:
    """Explain why groups may not be comparable (task version/content, environment)."""
    issues: list[str] = []
    hashes: dict[str, set[str]] = defaultdict(set)
    envs: set[str] = set()
    suites: set[str] = set()
    for a in attempts:
        if a["status"] == AttemptStatus.SKIPPED.value:
            continue
        hashes[a["task_id"]].add(f"v{a['task_version']}:{(a['task_hash'] or 'unknown')[:8]}")
        if a.get("env_fingerprint"):
            envs.add(a["env_fingerprint"])
        if a.get("suite_version"):
            suites.add(a["suite_version"])
    for task_id, versions in sorted(hashes.items()):
        if len(versions) > 1:
            issues.append(f"task {task_id} differs across runs: {sorted(versions)}")
    if len(envs) > 1:
        issues.append(f"{len(envs)} different environment fingerprints (tool/runtime versions)")
    if len(suites) > 1:
        issues.append(f"multiple suite versions: {sorted(suites)}")
    return issues


def review_summary(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Human/judge review aggregation with inter-reviewer disagreement (max - min)."""
    by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in reviews:
        by_key[(r["attempt_id"], r["dimension"], r["source"])].append(float(r["score"]))
    items: list[dict[str, Any]] = []
    for (attempt_id, dim, source), scores in sorted(by_key.items()):
        items.append(
            {
                "attempt_id": attempt_id,
                "dimension": dim,
                "source": source,
                "reviewers": len(scores),
                "mean": statistics.fmean(scores),
                "disagreement": max(scores) - min(scores) if len(scores) > 1 else None,
            }
        )
    return {
        "items": items,
        "high_disagreement": [i for i in items if (i["disagreement"] or 0) > 0.3],
    }


def notes(a: dict[str, Any]) -> list[str]:
    try:
        value = json.loads(a.get("notes_json") or "[]")
    except json.JSONDecodeError:
        return []
    return [str(v) for v in value] if isinstance(value, list) else []
