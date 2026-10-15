# Architecture

## Goals in one paragraph

`engineering-llm-bench` compares **models**, **harnesses** (how a model is driven), **tool
configurations** and **task difficulty** on realistic engineering work, while keeping those
dimensions separate in every stored record. A run executes versioned tasks in disposable
workspaces, verifies the result with *independent* evaluators (never the model's own claims),
scores it with a transparent weighted rubric, and persists evidence so every number in a report
can be traced back to logs, diffs and traces.

## Implementation plan (as executed)

1. Data contracts and threat model (this document, `docs/security.md`).
2. Validated configuration, task schema/registry, domain types, fake (mock) provider.
3. Isolated workspaces, sandboxed command runner, evaluator checks, deterministic scoring.
4. SQLite persistence + minimal CLI; prove one task end-to-end before adding breadth.
5. OpenAI-compatible and Anthropic adapters, external CLI harness, manual evidence import.
6. Starter task catalogue, each task validated independently (fixture fails, reference passes).
7. Dashboard and exports built only from persisted data.
8. Tests, CI, docs, final smoke run.

## Data flow

```
 configs/*.yaml ──► config (pydantic)  ─┐
 tasks/**/task.yaml ─► catalog (registry, hashes, availability)
                                        ▼
                        orchestration.BenchmarkRunner
   resolve ─► prepare ─► execute ─────────► verify ─────────► score ─► persist ─► report
   (config,   (fresh     (harness:          (evaluation:       (rubric   (SQLite +   (JSON/CSV/HTML,
    consent,   workspace  direct | agent |   fresh eval copy    weights,  atomic      dashboard)
    budget)    from       cli; provider      + hidden files,    coverage) artifact
               fixture)   adapter; tools     checks in the                files)
                          via sandboxed      sandbox)
                          runner)
```

```
src/engineering_llm_bench/
  domain.py            enums shared by every layer (dimensions, statuses, failure classes)
  control.py           CancelToken, Budget (deadline / turns / tokens)
  environment.py       environment + configuration fingerprints
  redaction.py         secret redaction for logs and exports
  config/              ModelsFile, BenchmarkConfig (validation rules live here)
  catalog/             TaskSpec schema, TaskRegistry, behavioural task validation
  providers/           Provider protocol, ManagedProvider (retry/rate limit), fake,
                       openai_compat (httpx), anthropic_provider (official SDK, optional)
  harnesses/           protocol (JSON actions / <file> blocks), agent loop, direct, cli
  tools/runner.py      CommandRunner: allowlist, arg checks, rlimits, timeouts, netns
  workspace/           disposable workspaces, safe paths, snapshots, diffs
  evaluation/          checks (pytest, cpp_test, command, ruff, sql_result, text_rubric, ...),
                       scoring, evaluator, review (blind human review, judge)
  storage/             SQLite Store (migrations, transactions), ArtifactStore (atomic files)
  orchestration/       run lifecycle, row conversion
  importing/           manual / IDE run import (Copilot BYOK etc.)
  reporting/           statistics, exports, static HTML template
  dashboard/           stdlib HTTP server + vanilla JS single page
  cli/                 `bench` (typer)
```

## Separation of dimensions

| Dimension      | Where it lives                                  | Recorded as |
|----------------|-------------------------------------------------|-------------|
| Model          | `ModelConfig` (alias, provider, model_id, runtime, quantization, sampling) | `runs.model_alias`, `provider_model_id`, `model_config_json` |
| Harness        | `HarnessConfig` (`direct`, `agent`, `cli:*`, `manual`) | `runs.harness`, `harness_version`, `mode` |
| Tools          | task `allowed_tools`/`allowed_commands` ∩ global allowlist | `run_config_json`, trace |
| Task           | `TaskSpec` + content hash of the task directory | `attempts.task_version`, `task_hash` |
| Environment    | `collect_environment()`                         | `runs.env_json`, `env_fingerprint` |

Failure classes distinguish model outcomes (`tests_failed`, `malformed_output`) from harness
(`harness_error`, `budget_exceeded`), provider (`provider_error`, `timeout`) and infrastructure
(`evaluation_error`, `unavailable`, `cancelled`) causes, so a harness or tooling failure is not
silently attributed to the model.

## Data contracts (schema v1)

### Task directory
`task.yaml` (see `catalog/schema.py`) + `fixture/` (workspace) + `evaluator/` (hidden) +
`reference/` (proves solvable) + optional `broken/` (proves the evaluator discriminates).

### Persistent schema (`storage/db.py`)

* `runs` - one model × harness × profile × configuration execution (or import).
* `attempts` - one (task, repeat) inside a run: status, failure class, timings, turns, tool
  calls, provider-reported usage (`NULL` when unknown), cost (only with reliable usage and
  user-supplied pricing), score, coverage, pass, regression flag, hashes, artifact dir.
* `dimension_scores` - per attempt × dimension: score (or `NULL` = unassessed), weight, source.
* `check_results` - per evaluator check: status, score, message, evidence file reference.
* `artifacts` - name → relative path, size, SHA-256 (trace, diff, summary, check logs).
* `reviews` - human or judge scores, kept separate (`source`, reviewer/judge identity).
* `schema_migrations` - applied versions; newer-than-supported databases are refused.

### Artifact layout
```
artifacts/                         (gitignored)
  bench.db
  runs/<run_id>/<attempt_id>/trace.jsonl | diff.patch | summary.md | checks/<check>.log
           | code_location.txt
  byok_sessions/<session_id>.json  (trusted BYOK manifests)
  model_outputs/                   (outputs_dir; chmod 000 unless being written - outputs.py)
    <model>/MODEL.json
    <model>/runs/<run_id>/<task_id>__r<N>/code/ | diff.patch | meta.json
    <model>/byok_sessions/<session_id>/   (archived BYOK session)
```
All of `artifacts/`, `tasks/`, `.git`, BYOK session folders and other attempts' workspaces are
hidden from model-run commands by a private mount namespace (`tools/runner.py`, `isolation.py`).

### Import format
`importing/schema.py` - see `docs/provider-setup.md#manual-import`.

## Run lifecycle details

1. **Resolve**: model/harness lookup, remote-consent check (`--allow-remote`), external harness
   opt-in, profile budget, task selection (unknown ids are errors), availability.
2. **Prepare**: `prepare_workspace` copies the fixture into a new temp dir; baseline snapshot hash.
3. **Execute**: harness under a `Budget` (wall clock, turns, tokens) and `CancelToken`.
4. **Verify**: `make_eval_copy` = final workspace + hidden files under `_hidden/` (any `_hidden`
   created by the model is deleted first); checks run through the sandboxed runner.
5. **Score**: `score_attempt` → per-dimension scores, aggregate over assessed dimensions,
   coverage, pass, regression flag.
6. **Persist**: the final code is saved to the model's output folder, evidence files are
   written atomically, then one DB transaction per attempt.
   `PENDING`/`RUNNING` rows left by an interrupted process become `INTERRUPTED`.
7. **Report**: exports and the dashboard read only the database and artifact files.

Resume (`--resume RUN_ID`) re-runs only non-terminal attempts and refuses if the configuration
fingerprint changed.

## Extension points

* **Provider**: implement `providers.base.Provider` (`info`, `complete`, `close`), register it in
  `providers.create_provider`, add a `provider` literal to `ModelConfig`.
* **Harness**: implement `harnesses.base.Harness.run(ctx) -> HarnessResult`; register in
  `harnesses.create_harness`.
* **Evaluator check**: add a pydantic spec to `catalog/schema.py` (`CheckSpec` union) and a
  function to `evaluation/checks.CHECKS`.
* **Task**: add a directory under `tasks/` (see `docs/task-authoring.md`).

## Design decisions and alternatives

| Decision | Why | Alternative considered |
|---|---|---|
| Own minimal JSON-action agent loop | identical, model-agnostic conditions for local and paid models; no framework lock-in | provider-native tool calling (better for some models, but not comparable across endpoints) |
| SQLite + files | local-first, zero ops, transactional | Postgres (unneeded), JSON files (no transactions/queries) |
| stdlib HTTP + vanilla JS dashboard | no build step, no CDN, tiny attack surface | Streamlit/React (heavier deps, network fetches) |
| Linux network namespace + rlimits | strong-enough default without Docker | Docker/gVisor per attempt (stronger; documented as recommended for untrusted models) |
| Deterministic checks first; judge optional & separate | transparent, reproducible | single LLM judge (opaque, drifts) |
| Regex text rubrics for review/reasoning tasks | deterministic proxy, inspectable evidence | human-only scoring (slow); judge-only (opaque) - both supported on top |
