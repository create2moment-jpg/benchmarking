"""Import format for manually captured runs (IDE agents such as Copilot BYOK, chat UIs, ...).

Missing data stays ``null``: the importer never estimates tokens, cost, timings or outcomes.
Imported command logs are stored as evidence only and are NEVER executed. Self-reported test
results are stored but not trusted; scores come from re-running the task's own evaluators on
the imported patch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IMPORT_FORMAT = "engineering-llm-bench/import"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CommandLog(_Strict):
    command: str | None = Field(default=None, max_length=10_000)
    exit_code: int | None = None
    output: str | None = Field(default=None, max_length=2_000_000)


class ImportedAttempt(_Strict):
    task_id: str
    task_version: str | None = None
    repeat_index: int = Field(default=0, ge=0, le=1000)
    patch: str | None = Field(default=None, max_length=5_000_000)
    patch_file: str | None = None
    transcript: str | None = Field(default=None, max_length=10_000_000)
    command_logs: list[CommandLog] = Field(default_factory=list, max_length=10_000)
    elapsed_s: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    summary: str | None = Field(default=None, max_length=200_000)
    human_notes: str | None = Field(default=None, max_length=200_000)
    self_reported_passed: bool | None = None

    @model_validator(mode="after")
    def _one_patch(self) -> ImportedAttempt:
        if self.patch is not None and self.patch_file is not None:
            raise ValueError("give either patch or patch_file, not both")
        return self


class ImportedRunMeta(_Strict):
    model_label: str = Field(min_length=1, max_length=128)
    provider_label: str | None = None
    provider_model_id: str | None = None
    harness: str = Field(min_length=1, max_length=64)
    harness_version: str | None = None
    mode: Literal["manual"] = "manual"
    captured_at: datetime | None = None
    remote: bool | None = None
    notes: str | None = Field(default=None, max_length=50_000)


class ImportFile(_Strict):
    schema_version: Literal[1] = 1
    format: Literal["engineering-llm-bench/import"] = "engineering-llm-bench/import"
    run: ImportedRunMeta
    attempts: list[ImportedAttempt] = Field(min_length=1, max_length=5000)
