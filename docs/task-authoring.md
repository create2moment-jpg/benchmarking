# Authoring tasks, evaluators, providers and harnesses

## Task layout

```
tasks/<domain>/<slug>/
  task.yaml      metadata, prompt, acceptance criteria, checks (schema: catalog/schema.py)
  fixture/       the starting workspace (copied fresh for every attempt)
  evaluator/     hidden inputs; available to checks as _hidden/<file>
  reference/     overlay with a correct solution (required to prove solvability)
  broken/        optional overlay with a plausible-but-wrong solution
```

Rules of thumb:
* Small and fast: 1-5 minutes for a capable model; evaluators should run in seconds.
* Synthetic fixtures only - no proprietary code or data.
* Precise contract in the prompt; acceptance criteria restate what checks verify.
* Hidden tests should check behaviour the prompt specifies, including edge cases. Put robustness
  cases in a separate check so they land in the `robustness` dimension.
* Keep public tests in the fixture passing; mark them `regression_guard: true`.
* Bump `version` whenever the prompt, fixture or evaluator changes (results are tagged with
  version and a content hash; changed tasks are flagged as non-comparable).

## `task.yaml` fields

| Field | Notes |
|---|---|
| `id` | `<domain>.<slug>`, unique |
| `version` | semver |
| `domain`, `difficulty`, `skills`, `tags` | catalogue metadata |
| `evaluation` | `executable`, `rubric` or `human_review` (shown in `bench list-tasks`) |
| `profiles` | `smoke` (must also be `standard`), `standard`, `extended` (always includes all) |
| `estimated_runtime_s`, `timeout_s` | timeout is clipped by the profile's `max_task_timeout_s` |
| `resource_profile` | `small` (memory_mb), `medium` (2x), `large` (no address-space limit; JVM/torch) |
| `requires` | `executables`, `python_modules`; missing → attempt `skipped` with reason |
| `allowed_tools` | subset of `list_files, read_file, write_file, run_command` |
| `allowed_commands` | logical names ∩ global `command_allowlist`; `local-binaries` permits `./prog` |
| `editable_paths` | glob patterns the model may write |
| `expected_artifacts` | files expected to exist (counted in the workflow dimension) |
| `weights` | optional per-dimension override |
| `human_review_dimensions` | dimensions intended for blind human review |
| `checks` | list, see below; at least one `functional` check |

## Check kinds

| kind | Purpose | Key fields |
|---|---|---|
| `pytest` | run pytest on targets; partial credit = passed / collected | `targets` (e.g. `_hidden/test_x.py`, `tests`) |
| `cpp_test` | compile sources with g++/clang++ and run; parse `CHECK <name> PASS\|FAIL` | `sources`, `include_dirs`, `std`, `flags` (`-O/-W/-D/-g/-f` only), `expected_checks` |
| `command` | allowlisted command; exit status or CHECK lines | `argv` (`{python}`, `{hidden}`, `{workspace}`), `expect_exit`, `parse` |
| `ruff` | lint score `1/(1+0.25·findings)` | `targets`, `select`, `line_length` |
| `sql_result` | run candidate SELECT on hidden seed DB; compare to CSV (`NULL` token) | `engine` (sqlite/duckdb), `query_file`, `setup_sql`, `expected_csv`, `ordered` |
| `tests_detect_bug` | candidate tests must fail on the original code and pass on the final code | `test_targets` |
| `added_tests` | count test definitions added | `glob`, `pattern`, `min_added` |
| `text_rubric` | regex criteria over a written file, with evidence | `file`, `criteria[any_of/all_of/none_of, weight]` |
| `summary` | heuristic handoff-quality check on the final summary | `max_words` |
| `files_exist` | presence of files | `paths` |

Hidden test file names must not collide with public test module names (e.g. use
`test_hidden_*.py`). Common fields: `id`, `dimension`, `weight`, `regression_guard`,
`timeout_s`, `description`.

## Validate a task

```bash
bench task validate tasks/<domain>/<slug>
```
This proves: the schema is valid, the untouched fixture **fails** functional checks, the
reference **passes** all functional checks with no evaluator errors, and `broken/` (if present)
**fails**. `tests/integration/test_task_catalogue.py` runs this for every task in CI.

## Adding an evaluator check

1. Add a pydantic model with a unique `kind` literal to `catalog/schema.py` and to `CheckSpec`.
2. Implement `def check_x(spec, ctx: EvalContext) -> CheckResult` in `evaluation/checks.py`
   using `ctx.run(...)` for commands (sandboxed); return `_error(...)` for evaluator problems and
   `_result(spec, score, message, evidence)` for candidate outcomes. Register it in `CHECKS`.
3. Add unit tests and use it in a task.

## Adding a provider

1. Implement the `Provider` protocol (`providers/base.py`): `info`, `complete(request, cancel)`,
   `close()`. Raise `ProviderError(kind, message, retryable=...)`; never send `request.metadata`.
   Report usage only when the API reports it.
2. Add the provider literal to `ModelConfig.provider` and a branch in `providers.create_provider`.
3. Test against the documented API with mocked transports (see
   `tests/unit/test_protocol_and_providers.py`).

## Adding a harness

Implement `Harness.run(ctx: HarnessContext) -> HarnessResult` (`harnesses/base.py`), using
`harnesses.tools.execute` for workspace tools so path/allowlist policies stay centralised, and
classify failures with `FailureClass`. Register it in `harnesses.create_harness` and give it a
`HarnessConfig.kind`. Record what the harness controls vs. what it cannot observe (e.g. token
usage) - unknowns stay null.
