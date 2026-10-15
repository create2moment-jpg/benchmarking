"""Machine-readable (JSON/CSV) and shareable (static HTML) reports built from persisted data.

Exports are redacted. Transcripts (model traces) and source (diffs) are excluded unless the
caller opts in with ``include_transcripts`` / ``include_source``.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from jinja2 import Environment, select_autoescape

from engineering_llm_bench import DATA_SCHEMA_VERSION, __version__
from engineering_llm_bench.domain import Dimension
from engineering_llm_bench.redaction import Redactor
from engineering_llm_bench.reporting.stats import (
    comparability,
    dimension_means,
    group_key,
    review_summary,
    summarize_group,
)
from engineering_llm_bench.storage.artifacts import ArtifactStore
from engineering_llm_bench.storage.db import Store, utcnow

_RUN_FIELDS_DROP = {"env_json", "model_config_json", "harness_config_json", "run_config_json"}


@dataclass
class ReportOptions:
    run_ids: list[str] | None = None
    include_transcripts: bool = False
    include_source: bool = False


def _json_field(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def build_report(
    store: Store, artifacts: ArtifactStore, redactor: Redactor, options: ReportOptions
) -> dict[str, Any]:
    """Assemble a report dict from the database (the single source of truth for exports)."""
    runs = store.list_runs(limit=10_000)
    if options.run_ids:
        wanted = set(options.run_ids)
        missing = wanted - {r["id"] for r in runs}
        if missing:
            raise KeyError(f"unknown run ids: {sorted(missing)}")
        runs = [r for r in runs if r["id"] in wanted]
    run_ids = {r["id"] for r in runs}
    joined = [a for a in store.all_attempts() if a["run_id"] in run_ids]
    attempts_out = []
    for a in joined:
        dims = store.dimension_scores(a["id"])
        checks = store.check_results(a["id"])
        item = {
            **{k: v for k, v in a.items() if k != "notes_json"},
            "notes": _json_field(a["notes_json"]) or [],
            "dimensions": {d["dimension"]: d["score"] for d in dims},
            "dimension_sources": {d["dimension"]: d["source"] for d in dims},
            "checks": [
                {
                    k: c[k]
                    for k in (
                        "check_id",
                        "kind",
                        "dimension",
                        "status",
                        "score",
                        "source",
                        "weight",
                        "message",
                    )
                }
                for c in checks
            ],
            "reviews": store.reviews(a["id"]),
        }
        names = {r["name"]: r["path"] for r in store.artifacts(a["id"])}
        if options.include_source and "diff.patch" in names:
            item["diff"] = artifacts.resolve(names["diff.patch"]).read_text(encoding="utf-8")
        if options.include_transcripts and "trace.jsonl" in names:
            item["trace"] = artifacts.resolve(names["trace.jsonl"]).read_text(encoding="utf-8")
        attempts_out.append(item)
    groups: dict[str, list[dict[str, Any]]] = {}
    for a in joined:
        groups.setdefault(group_key(a), []).append(a)
    group_out = {}
    for key, rows in sorted(groups.items()):
        summary = summarize_group(rows)
        dim_rows = [d for a in rows for d in store.dimension_scores(a["id"])]
        summary["dimensions"] = dimension_means(dim_rows)
        summary["model_alias"], summary["harness"] = rows[0]["model_alias"], rows[0]["harness"]
        summary["provider"] = rows[0]["provider"]
        summary["mode"] = rows[0]["mode"]
        group_out[key] = summary
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for a in joined:
        cell = matrix.setdefault(a["task_id"], {}).setdefault(
            group_key(a), {"scores": [], "passed": 0, "n": 0, "status": []}
        )
        cell["status"].append(a["status"])
        if a["score"] is not None:
            cell["scores"].append(a["score"])
        if a["passed"] is not None:
            cell["n"] += 1
            cell["passed"] += int(a["passed"])
    for cells in matrix.values():
        for cell in cells.values():
            sc = cell.pop("scores")
            cell["mean"] = sum(sc) / len(sc) if sc else None
            cell["status"] = sorted(set(cell["status"]))
    report = {
        "schema_version": DATA_SCHEMA_VERSION,
        "generator": f"engineering-llm-bench {__version__}",
        "generated_at": utcnow(),
        "includes": {
            "transcripts": options.include_transcripts,
            "source": options.include_source,
        },
        "runs": [
            {
                **{k: v for k, v in r.items() if k not in _RUN_FIELDS_DROP},
                "environment": _json_field(r["env_json"]),
                "model_config": _json_field(r["model_config_json"]),
                "harness_config": _json_field(r["harness_config_json"]),
                "run_config": _json_field(r["run_config_json"]),
            }
            for r in runs
        ],
        "groups": group_out,
        "comparability": comparability(joined),
        "task_matrix": dict(sorted(matrix.items())),
        "reviews": review_summary([rv for a in attempts_out for rv in a["reviews"]]),
        "attempts": attempts_out,
    }
    redacted: dict[str, Any] = redactor.obj(report)
    return redacted


CSV_COLUMNS = [
    "run_id",
    "model_alias",
    "harness",
    "mode",
    "provider",
    "profile",
    "task_id",
    "task_version",
    "task_domain",
    "evaluation_kind",
    "repeat_index",
    "status",
    "failure_class",
    "skip_reason",
    "score",
    "coverage",
    "passed",
    "regression_failed",
    "agent_wall_s",
    "eval_wall_s",
    "turns",
    "tool_calls",
    "tool_calls_ok",
    "malformed_outputs",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "config_fingerprint",
    "env_fingerprint",
    "task_hash",
]


def to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=False, default=str)


def to_csv(report: dict[str, Any]) -> str:
    """One row per attempt; dimension scores as ``dim_<name>`` columns; nulls as empty."""
    buf = io.StringIO()
    dims = [f"dim_{d.value}" for d in Dimension]
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS + dims, extrasaction="ignore")
    writer.writeheader()
    for a in report["attempts"]:
        row = {k: a.get(k) for k in CSV_COLUMNS}
        for d in Dimension:
            row[f"dim_{d.value}"] = a["dimensions"].get(d.value)
        writer.writerow(row)
    return buf.getvalue()


def _fmt(value: Any, digits: int = 1, pct: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, float | int) and not isinstance(value, bool):
        return f"{value * 100:.0f}%" if pct else f"{value:.{digits}f}"
    return str(value)


def to_html(report: dict[str, Any]) -> str:
    env = Environment(autoescape=select_autoescape(default=True, default_for_string=True))
    env.filters["fmt"] = _fmt
    template_text = (
        resources.files("engineering_llm_bench.reporting")
        .joinpath("report.html.j2")
        .read_text(encoding="utf-8")
    )
    return env.from_string(template_text).render(r=report, dimensions=[d.value for d in Dimension])
