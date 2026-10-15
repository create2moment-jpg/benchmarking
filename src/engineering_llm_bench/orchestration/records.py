"""Convert evaluation outcomes into persisted rows (shared by executed and imported runs)."""

from __future__ import annotations

from typing import Any

from engineering_llm_bench.evaluation.evaluator import EvaluationOutcome
from engineering_llm_bench.storage.artifacts import ArtifactRef, ArtifactStore
from engineering_llm_bench.storage.db import dumps


def evaluation_rows(
    outcome: EvaluationOutcome,
    artifacts: ArtifactStore,
    run_id: str,
    attempt_id: str,
    refs: list[ArtifactRef],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Write check evidence files (appending to ``refs``) and return (dimension, check) rows."""
    checks: list[dict[str, Any]] = []
    for c in outcome.checks:
        evidence_path = None
        if c.evidence:
            ref = artifacts.write_text(run_id, attempt_id, f"checks/{c.check_id}.log", c.evidence)
            refs.append(ref)
            evidence_path = ref.path
        checks.append(
            {
                "check_id": c.check_id,
                "kind": c.kind,
                "dimension": c.dimension.value,
                "weight": c.weight,
                "status": c.status.value,
                "score": c.score,
                "source": c.source.value,
                "regression_guard": int(c.regression_guard),
                "message": artifacts.redactor.text(c.message)[:2000],
                "duration_s": round(c.duration_s, 3),
                "evidence_path": evidence_path,
                "details_json": dumps(c.details),
            }
        )
    dims = [
        {
            "dimension": d.dimension.value,
            "score": d.score,
            "weight": d.weight,
            "source": d.source.value,
            "n_checks": d.n_checks,
            "details": d.details,
        }
        for d in outcome.scorecard.dimensions
    ]
    return dims, checks


def artifact_rows(refs: list[ArtifactRef]) -> list[dict[str, Any]]:
    return [{"name": r.name, "path": r.path, "bytes": r.bytes, "sha256": r.sha256} for r in refs]
