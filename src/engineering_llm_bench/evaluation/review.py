"""Human and judge review for rubric dimensions that cannot be scored deterministically.

* ``export_blind_packet`` writes a review packet with model identity removed (blind ids) and a
  separate key file mapping blind ids back to attempts. Give reviewers only the packet.
* ``import_reviews`` reads a CSV ``blind_id,reviewer,dimension,score,rationale`` (score 0-1)
  using the key file. Two or more reviewers per item give a disagreement measure.
* ``judge_attempts`` asks a configured judge model for a 0-1 score with rationale. Judge scores
  are stored with ``source = judge`` and the judge identity, never mixed with deterministic ones.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engineering_llm_bench.catalog.registry import TaskRegistry
from engineering_llm_bench.control import CancelToken
from engineering_llm_bench.domain import AttemptStatus, Dimension
from engineering_llm_bench.providers.base import (
    ChatMessage,
    CompletionRequest,
    ManagedProvider,
    ProviderError,
)
from engineering_llm_bench.storage.artifacts import ArtifactStore, atomic_write_bytes
from engineering_llm_bench.storage.db import Store

_MAX_DIFF = 60_000


@dataclass
class ReviewImportResult:
    recorded: int
    errors: list[str]


def _diff_for(store: Store, artifacts: ArtifactStore, attempt_id: str) -> str:
    names = {r["name"]: r["path"] for r in store.artifacts(attempt_id)}
    if "diff.patch" not in names:
        return ""
    return artifacts.resolve(names["diff.patch"]).read_text(encoding="utf-8")[:_MAX_DIFF]


def export_blind_packet(
    store: Store,
    artifacts: ArtifactStore,
    registry: TaskRegistry,
    run_ids: list[str],
    out_dir: Path,
    dimensions: list[Dimension],
) -> tuple[Path, Path]:
    """Write ``packet.json`` (blind) and ``key.json`` (secret mapping). Items are shuffled."""
    salt = secrets.token_hex(8)
    items: list[dict[str, Any]] = []
    key: dict[str, str] = {}
    for run_id in run_ids:
        for a in store.attempts_for_run(run_id):
            if a["status"] != AttemptStatus.COMPLETED.value:
                continue
            blind = hashlib.sha256(f"{salt}:{a['id']}".encode()).hexdigest()[:12]
            key[blind] = a["id"]
            task = registry.tasks.get(a["task_id"])
            items.append(
                {
                    "blind_id": blind,
                    "task_id": a["task_id"],
                    "task_prompt": task.spec.prompt if task else None,
                    "acceptance_criteria": task.spec.acceptance_criteria if task else None,
                    "dimensions": [d.value for d in dimensions],
                    "summary": a["summary"],
                    "diff": _diff_for(store, artifacts, a["id"]),
                }
            )
    items.sort(key=lambda i: i["blind_id"])  # order independent of model/run
    out_dir.mkdir(parents=True, exist_ok=True)
    packet = out_dir / "packet.json"
    key_file = out_dir / "key.json"
    atomic_write_bytes(packet, json.dumps({"items": items}, indent=2).encode())
    atomic_write_bytes(key_file, json.dumps({"key": key}, indent=2).encode())
    template = out_dir / "reviews_template.csv"
    rows = ["blind_id,reviewer,dimension,score,rationale"]
    rows += [f"{i['blind_id']},,{d},," for i in items for d in i["dimensions"]]
    atomic_write_bytes(template, ("\n".join(rows) + "\n").encode())
    return packet, key_file


def import_reviews(store: Store, key_file: Path, csv_file: Path) -> ReviewImportResult:
    key = json.loads(key_file.read_text(encoding="utf-8")).get("key", {})
    valid_dims = {d.value for d in Dimension}
    recorded, errors = 0, []
    with csv_file.open(newline="", encoding="utf-8") as fh:
        for lineno, row in enumerate(csv.DictReader(fh), start=2):
            blind = (row.get("blind_id") or "").strip()
            reviewer = (row.get("reviewer") or "").strip()
            dim = (row.get("dimension") or "").strip()
            raw = (row.get("score") or "").strip()
            if not reviewer or not raw:
                continue  # unfilled template row
            if blind not in key:
                errors.append(f"line {lineno}: unknown blind_id {blind!r}")
                continue
            if dim not in valid_dims:
                errors.append(f"line {lineno}: unknown dimension {dim!r}")
                continue
            try:
                score = float(raw)
            except ValueError:
                errors.append(f"line {lineno}: score {raw!r} is not a number")
                continue
            if not 0.0 <= score <= 1.0:
                errors.append(f"line {lineno}: score must be within [0, 1]")
                continue
            store.add_review(
                key[blind],
                dim,
                reviewer,
                "human",
                score,
                (row.get("rationale") or "").strip() or None,
                blind=True,
            )
            recorded += 1
    return ReviewImportResult(recorded, errors)


JUDGE_PROMPT = """You are reviewing an engineering change against an explicit rubric.
Score the dimension "{dimension}" from 0.0 (unacceptable) to 1.0 (excellent).
Judge only what the evidence shows. Reply with ONLY a JSON object:
{{"score": <number 0-1>, "rationale": "<two or three sentences citing evidence>"}}

## Task
{prompt}

## Acceptance criteria
{criteria}

## Engineer's summary
{summary}

## Diff
{diff}
"""


def parse_judge(text: str) -> tuple[float, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in judge reply")
    obj = json.loads(match.group(0))
    score = float(obj["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError("judge score out of range")
    return score, str(obj.get("rationale", ""))[:4000]


def judge_attempts(
    store: Store,
    artifacts: ArtifactStore,
    registry: TaskRegistry,
    judge: ManagedProvider,
    run_id: str,
    dimension: Dimension,
    cancel: CancelToken | None = None,
) -> ReviewImportResult:
    cancel = cancel or CancelToken()
    label = f"judge:{judge.info.alias}:{judge.info.model_id}"
    recorded, errors = 0, []
    for a in store.attempts_for_run(run_id):
        if a["status"] != AttemptStatus.COMPLETED.value or a["task_id"] not in registry.tasks:
            continue
        task = registry.tasks[a["task_id"]]
        prompt = JUDGE_PROMPT.format(
            dimension=dimension.value,
            prompt=task.spec.prompt,
            criteria="\n".join(f"- {c}" for c in task.spec.acceptance_criteria),
            summary=a["summary"] or "(none)",
            diff=_diff_for(store, artifacts, a["id"]) or "(no changes)",
        )
        request = CompletionRequest(
            messages=(ChatMessage("user", prompt),),
            max_tokens=1024,
            temperature=0.0,
            timeout_s=120,
            metadata={"mode": "judge"},
        )
        try:
            reply = judge.complete(request, cancel)
            score, rationale = parse_judge(reply.text)
        except (ProviderError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"{a['id']}: {exc}")
            continue
        store.add_review(
            a["id"],
            dimension.value,
            label,
            "judge",
            score,
            rationale,
            judge_model=judge.info.model_id,
        )
        recorded += 1
    return ReviewImportResult(recorded, errors)
