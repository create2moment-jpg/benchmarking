"""SQLite persistence: versioned schema, migrations, transactional writes, typed queries.

SQLite runs in WAL mode with ``synchronous=FULL`` and foreign keys on. Every logical write is
one transaction, so an interrupted process leaves either the old or the new state. Large data
(logs, diffs, traces) is stored as artifact files and referenced here.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engineering_llm_bench import DATA_SCHEMA_VERSION

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE runs (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        finished_at TEXT,
        status TEXT NOT NULL,
        source TEXT NOT NULL,
        profile TEXT,
        suite_name TEXT,
        suite_version TEXT,
        model_alias TEXT NOT NULL,
        provider TEXT,
        provider_model_id TEXT,
        endpoint_type TEXT,
        remote INTEGER,
        harness TEXT NOT NULL,
        harness_version TEXT,
        mode TEXT NOT NULL,
        model_config_json TEXT,
        harness_config_json TEXT,
        run_config_json TEXT,
        config_fingerprint TEXT,
        env_json TEXT,
        env_fingerprint TEXT,
        task_count INTEGER,
        notes TEXT,
        error TEXT
    );
    CREATE INDEX idx_runs_model ON runs(model_alias);
    CREATE INDEX idx_runs_created ON runs(created_at);

    CREATE TABLE attempts (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        task_id TEXT NOT NULL,
        task_version TEXT,
        task_hash TEXT,
        task_domain TEXT,
        evaluation_kind TEXT,
        repeat_index INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        failure_class TEXT,
        skip_reason TEXT,
        error TEXT,
        started_at TEXT,
        finished_at TEXT,
        agent_wall_s REAL,
        eval_wall_s REAL,
        turns INTEGER,
        tool_calls INTEGER,
        tool_calls_ok INTEGER,
        malformed_outputs INTEGER,
        input_tokens INTEGER,
        output_tokens INTEGER,
        usage_reliable INTEGER,
        cost_usd REAL,
        score REAL,
        coverage REAL,
        passed INTEGER,
        regression_failed INTEGER,
        summary TEXT,
        baseline_hash TEXT,
        final_hash TEXT,
        artifact_dir TEXT,
        notes_json TEXT,
        UNIQUE (run_id, task_id, repeat_index)
    );
    CREATE INDEX idx_attempts_run ON attempts(run_id);
    CREATE INDEX idx_attempts_task ON attempts(task_id);

    CREATE TABLE dimension_scores (
        attempt_id TEXT NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
        dimension TEXT NOT NULL,
        score REAL,
        weight REAL NOT NULL,
        source TEXT NOT NULL,
        n_checks INTEGER NOT NULL,
        details TEXT,
        PRIMARY KEY (attempt_id, dimension)
    );

    CREATE TABLE check_results (
        attempt_id TEXT NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
        check_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        dimension TEXT NOT NULL,
        weight REAL NOT NULL,
        status TEXT NOT NULL,
        score REAL,
        source TEXT NOT NULL,
        regression_guard INTEGER NOT NULL,
        message TEXT,
        duration_s REAL,
        evidence_path TEXT,
        details_json TEXT,
        PRIMARY KEY (attempt_id, check_id)
    );

    CREATE TABLE artifacts (
        attempt_id TEXT NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        path TEXT NOT NULL,
        bytes INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        PRIMARY KEY (attempt_id, name)
    );

    CREATE TABLE reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        attempt_id TEXT NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
        dimension TEXT NOT NULL,
        reviewer TEXT NOT NULL,
        source TEXT NOT NULL CHECK (source IN ('human', 'judge')),
        score REAL NOT NULL CHECK (score >= 0 AND score <= 1),
        rationale TEXT,
        judge_model TEXT,
        blind INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        UNIQUE (attempt_id, dimension, reviewer)
    );
    """,
    # v2: model-server performance (endpoint info, throughput calibration, counters, timing).
    2: """
    ALTER TABLE runs ADD COLUMN perf_json TEXT;
    ALTER TABLE attempts ADD COLUMN model_time_s REAL;
    ALTER TABLE attempts ADD COLUMN eval_cpu_s REAL;
    ALTER TABLE attempts ADD COLUMN eval_max_rss_mb REAL;
    """,
}

ATTEMPT_COLUMNS = (
    "id",
    "run_id",
    "task_id",
    "task_version",
    "task_hash",
    "task_domain",
    "evaluation_kind",
    "repeat_index",
    "status",
    "failure_class",
    "skip_reason",
    "error",
    "started_at",
    "finished_at",
    "agent_wall_s",
    "eval_wall_s",
    "turns",
    "tool_calls",
    "tool_calls_ok",
    "malformed_outputs",
    "input_tokens",
    "output_tokens",
    "usage_reliable",
    "cost_usd",
    "score",
    "coverage",
    "passed",
    "regression_failed",
    "summary",
    "baseline_hash",
    "final_hash",
    "artifact_dir",
    "notes_json",
    "model_time_s",
    "eval_cpu_s",
    "eval_max_rss_mb",
)


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SchemaVersionError(RuntimeError):
    """The database was written by a newer, incompatible version of the tool."""


class Store:
    """Thread-safe SQLite store. One connection guarded by a lock (writes are short)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.execute("PRAGMA busy_timeout = 10000")
        self.migrate()

    # ------------------------------------------------------------------ infrastructure
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def schema_version(self) -> int:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            row = self._conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        return int(row[0] or 0)

    def migrate(self) -> None:
        current = self.schema_version()
        if current > DATA_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"database schema v{current} is newer than supported v{DATA_SCHEMA_VERSION}"
            )
        for version in sorted(v for v in MIGRATIONS if v > current):
            with self.transaction() as conn:
                for statement in MIGRATIONS[version].split(";"):
                    if statement.strip():
                        conn.execute(statement)
                conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, utcnow()),
                )

    def integrity_check(self) -> str:
        return str(self.query("PRAGMA integrity_check")[0]["integrity_check"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ runs
    def insert_run(self, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self.transaction() as conn:
            conn.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", list(row.values()))  # noqa: S608

    def update_run(self, run_id: str, **fields: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self.transaction() as conn:
            conn.execute(f"UPDATE runs SET {sets} WHERE id = ?", [*fields.values(), run_id])  # noqa: S608

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM runs WHERE id = ?", (run_id,))
        return rows[0] if rows else None

    def list_runs(self, limit: int = 500) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM runs ORDER BY created_at DESC, id DESC LIMIT ?", (limit,))

    # ------------------------------------------------------------------ attempts
    def upsert_attempt(self, row: dict[str, Any]) -> None:
        """Update the attempt's given columns, inserting the row if it does not exist yet.

        (UPDATE-then-INSERT rather than INSERT ... ON CONFLICT, so partial updates of existing
        rows do not trip NOT NULL constraints on columns they leave out.)
        """
        unknown = set(row) - set(ATTEMPT_COLUMNS)
        if unknown:
            raise KeyError(f"unknown attempt columns {sorted(unknown)}")
        fields = {k: v for k, v in row.items() if k != "id"}
        with self.transaction() as conn:
            updated = 0
            if fields:
                sets = ", ".join(f"{k} = ?" for k in fields)
                cur = conn.execute(
                    f"UPDATE attempts SET {sets} WHERE id = ?",  # noqa: S608 - columns allowlisted
                    [*fields.values(), row["id"]],
                )
                updated = cur.rowcount
            if not updated:
                cols = ", ".join(row)
                marks = ", ".join("?" for _ in row)
                conn.execute(
                    f"INSERT INTO attempts ({cols}) VALUES ({marks})",  # noqa: S608
                    list(row.values()),
                )

    def save_evaluation(
        self,
        attempt: dict[str, Any],
        dimensions: list[dict[str, Any]],
        checks: list[dict[str, Any]],
        artifacts: list[dict[str, Any]],
    ) -> None:
        """Write an attempt's final row and all evaluation records in ONE transaction."""
        cols = ", ".join(attempt)
        marks = ", ".join("?" for _ in attempt)
        updates = ", ".join(f"{k} = excluded.{k}" for k in attempt if k != "id")
        with self.transaction() as conn:
            conn.execute(
                f"INSERT INTO attempts ({cols}) VALUES ({marks}) "  # noqa: S608
                f"ON CONFLICT(id) DO UPDATE SET {updates}",
                list(attempt.values()),
            )
            aid = attempt["id"]
            for table in ("dimension_scores", "check_results", "artifacts"):
                conn.execute(f"DELETE FROM {table} WHERE attempt_id = ?", (aid,))  # noqa: S608
            for table, rows in (
                ("dimension_scores", dimensions),
                ("check_results", checks),
                ("artifacts", artifacts),
            ):
                for raw in rows:
                    r = {"attempt_id": aid, **raw}
                    cols, marks = ", ".join(r), ", ".join("?" for _ in r)
                    conn.execute(
                        f"INSERT INTO {table} ({cols}) VALUES ({marks})",  # noqa: S608
                        list(r.values()),
                    )

    def attempts_for_run(self, run_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM attempts WHERE run_id = ? ORDER BY task_id, repeat_index", (run_id,)
        )

    def get_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM attempts WHERE id = ?", (attempt_id,))
        return rows[0] if rows else None

    def dimension_scores(self, attempt_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM dimension_scores WHERE attempt_id = ? ORDER BY dimension", (attempt_id,)
        )

    def check_results(self, attempt_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM check_results WHERE attempt_id = ? ORDER BY check_id", (attempt_id,)
        )

    def artifacts(self, attempt_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM artifacts WHERE attempt_id = ? ORDER BY name", (attempt_id,)
        )

    def all_attempts(self) -> list[dict[str, Any]]:
        """Attempts joined with their run's comparison dimensions (for reports/dashboard)."""
        return self.query(
            """SELECT a.*, r.model_alias, r.provider, r.provider_model_id, r.harness,
                      r.harness_version, r.mode, r.profile, r.source AS run_source,
                      r.created_at AS run_created_at, r.config_fingerprint, r.env_fingerprint,
                      r.endpoint_type, r.remote, r.suite_version
               FROM attempts a JOIN runs r ON r.id = a.run_id
               ORDER BY r.created_at DESC, a.task_id, a.repeat_index"""
        )

    # ------------------------------------------------------------------ reviews
    def add_review(
        self,
        attempt_id: str,
        dimension: str,
        reviewer: str,
        source: str,
        score: float,
        rationale: str | None,
        judge_model: str | None = None,
        blind: bool = False,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO reviews
                   (attempt_id, dimension, reviewer, source, score, rationale, judge_model,
                    blind, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(attempt_id, dimension, reviewer) DO UPDATE SET
                     score = excluded.score, rationale = excluded.rationale,
                     created_at = excluded.created_at""",
                (
                    attempt_id,
                    dimension,
                    reviewer,
                    source,
                    score,
                    rationale,
                    judge_model,
                    int(blind),
                    utcnow(),
                ),
            )

    def reviews(self, attempt_id: str | None = None) -> list[dict[str, Any]]:
        if attempt_id is None:
            return self.query("SELECT * FROM reviews ORDER BY attempt_id, dimension, reviewer")
        return self.query(
            "SELECT * FROM reviews WHERE attempt_id = ? ORDER BY dimension, reviewer", (attempt_id,)
        )


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)
