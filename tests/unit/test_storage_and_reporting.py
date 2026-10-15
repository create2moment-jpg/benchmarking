"""Storage (migrations, transactions, atomic artifacts), redaction and statistics."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from engineering_llm_bench import DATA_SCHEMA_VERSION
from engineering_llm_bench.redaction import REDACTED, Redactor
from engineering_llm_bench.reporting.stats import comparability, percentile, summarize_group
from engineering_llm_bench.storage.artifacts import ArtifactStore
from engineering_llm_bench.storage.db import SchemaVersionError, Store


def run_row(run_id: str = "r1") -> dict[str, Any]:
    return {
        "id": run_id,
        "created_at": "2026-01-01T00:00:00",
        "status": "running",
        "source": "executed",
        "model_alias": "m",
        "harness": "agent",
        "mode": "agent",
    }


def test_migrations_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "b.db"
    Store(db).close()
    store = Store(db)
    assert store.schema_version() == DATA_SCHEMA_VERSION
    assert store.integrity_check() == "ok"


def test_newer_schema_refused(tmp_path: Path) -> None:
    db = tmp_path / "b.db"
    Store(db).close()
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO schema_migrations VALUES (999, 'future')")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaVersionError):
        Store(db)


def test_save_evaluation_is_atomic(tmp_path: Path) -> None:
    store = Store(tmp_path / "b.db")
    store.insert_run(run_row())
    attempt = {"id": "a1", "run_id": "r1", "task_id": "t", "repeat_index": 0, "status": "completed"}
    bad_dim = [
        {
            "dimension": "functional",
            "score": 1.0,
            "weight": None,
            "source": "deterministic",
            "n_checks": 1,
        }
    ]
    with pytest.raises(sqlite3.IntegrityError):
        store.save_evaluation(attempt, bad_dim, [], [])
    assert store.get_attempt("a1") is None  # rolled back together


def test_unknown_attempt_column_rejected(tmp_path: Path) -> None:
    store = Store(tmp_path / "b.db")
    with pytest.raises(KeyError):
        store.upsert_attempt({"id": "a", "evil; DROP TABLE runs": 1})


def test_review_score_bounds(tmp_path: Path) -> None:
    store = Store(tmp_path / "b.db")
    store.insert_run(run_row())
    store.upsert_attempt(
        {"id": "a1", "run_id": "r1", "task_id": "t", "repeat_index": 0, "status": "completed"}
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.add_review("a1", "explanation", "alice", "human", 1.5, None)


def test_artifacts_atomic_and_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TOKEN", "super-secret-value-123")
    store = ArtifactStore(tmp_path, Redactor(secret_env_names=["MY_TOKEN"]))
    ref = store.write_text("r1", "a1", "logs/out.txt", "token=super-secret-value-123 ok")
    text = store.resolve(ref.path).read_text()
    assert "super-secret" not in text and REDACTED in text
    assert not list((tmp_path / "runs" / "r1" / "a1" / "logs").glob(".tmp-*"))
    with pytest.raises(ValueError):
        store.write_text("r1", "a1", "../escape.txt", "x")
    with pytest.raises(ValueError):
        store.resolve("../../etc/passwd")


@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
        "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "AKIAABCDEFGHIJKLMNOP",
        "Bearer abcdefghijklmnopqrstuvwxyz",
    ],
)
def test_redaction_patterns(secret: str) -> None:
    out = Redactor().text(f"value: {secret} end")
    assert secret not in out and REDACTED in out


def test_redaction_keeps_key_names() -> None:
    assert Redactor().text("api_key=abcdefgh12345678") == f"api_key={REDACTED}"


def test_percentile() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0], 90) == 5.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 90) == pytest.approx(4.6)


def attempt(**kw: Any) -> dict[str, Any]:
    base = {
        "status": "completed",
        "passed": 1,
        "score": 80.0,
        "agent_wall_s": 10.0,
        "failure_class": "none",
        "tool_calls": 4,
        "tool_calls_ok": 3,
        "regression_failed": 0,
        "usage_reliable": 1,
        "input_tokens": 100,
        "output_tokens": 10,
        "cost_usd": None,
        "malformed_outputs": 0,
        "task_domain": "python",
        "provider": "openai_compatible",
        "task_id": "t1",
        "task_version": "1.0.0",
        "task_hash": "h1",
        "env_fingerprint": "e1",
    }
    base.update(kw)
    return base


def test_skipped_never_counts_as_pass_or_fail() -> None:
    rows = [
        attempt(),
        attempt(status="skipped", passed=None, score=None, failure_class="unavailable"),
    ]
    s = summarize_group(rows)
    assert s["skipped"] == 1 and s["evaluated"] == 1 and s["pass_rate"] == 1.0
    assert s["failure_rate"] == 0.0


def test_cost_unknown_unless_reported() -> None:
    s = summarize_group([attempt(), attempt()])
    assert s["cost_usd"]["total"] is None
    s = summarize_group([attempt(cost_usd=0.5), attempt()])
    assert s["cost_usd"] == {"total": 0.5, "reported_attempts": 1, "of": 2}


def test_small_sample_and_mock_warnings() -> None:
    s = summarize_group([attempt(provider="fake")])
    assert any("small sample" in w for w in s["warnings"])
    assert any("MOCK" in w for w in s["warnings"])


def test_rates() -> None:
    rows = [attempt(failure_class="timeout", passed=0), attempt(regression_failed=1), attempt()]
    s = summarize_group(rows)
    assert s["timeout_rate"] == pytest.approx(1 / 3)
    assert s["regression_rate"] == pytest.approx(1 / 3)
    assert s["tool_success_rate"] == pytest.approx(9 / 12)


def test_comparability_flags_task_and_env_changes() -> None:
    issues = comparability([attempt(), attempt(task_hash="h2", env_fingerprint="e2")])
    assert any("t1 differs" in i for i in issues)
    assert any("environment" in i for i in issues)
