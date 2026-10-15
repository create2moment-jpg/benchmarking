# engineering-llm-bench

Compare **local and paid LLMs and agent harnesses** on realistic engineering tasks - Python, C++
sensor fusion, perception, SQL, Spark, PyTorch, build tooling, debugging, code review and design
reasoning - with reproducible, independently verified, transparently scored runs.

* **Engineering outcomes, not throughput.** Each task is a small repository with hidden
  evaluators. The model plans, edits, runs tools and tests; *independent* checks decide the result.
* **Separate dimensions.** Model, harness (direct / agent loop / external CLI / manual import),
  tool configuration, task version and environment are recorded separately, so harness or
  tooling failures are not blamed on the model.
* **Auditable.** Every score links to traces, diffs, evaluator logs and fingerprints in SQLite +
  artifact files; a local dashboard and JSON/CSV/HTML exports read only persisted data.
* **Local-first and safe by default.** Nothing leaves the machine unless you mark a model remote
  *and* pass `--allow-remote`. Commands run in disposable workspaces with an allowlist, rlimits,
  timeouts, a scrubbed environment and (on Linux) no network.

> Not a model-training or GPU-throughput benchmark. Wall time and provider-reported usage are
> recorded as context only.

## Benchmarking a model through Copilot BYOK (the easy way)

```bash
./setup.sh      # builds/installs everything (first run), creates a new session, opens it in VS Code
```
In the VS Code window that opens (inside a Docker Dev Container with every tool installed and
commands auto-approved): **Copilot Chat → Agent → pick your model → paste (Ctrl+V) → Enter.**
```bash
./finish.sh     # when the agent says ALL TASKS COMPLETE: scores it and opens the dashboard
```
Model on another machine? `cp server.example.yaml server.yaml` and fill in the SSH login: the
run then also records GPU usage, LLM requests and their durations on the server. The dashboard
shows how long each run took (overall and per task) next to the scores.
Details: [instructions/README.md](instructions/README.md) and [byok/README.md](byok/README.md).

## Per-model output folders (invisible to other models)

Every attempt's final code is saved per model:

```
artifacts/model_outputs/<model>/
  MODEL.json
  runs/<run_id>/<task_id>__r<N>/{code/, diff.patch, meta.json}
  byok_sessions/<session_id>/       # Copilot BYOK sessions, moved here by `byok collect`
```

* Commands run for a model (agent tools, external agent CLIs, evaluators) see this folder,
  `artifacts/`, `tasks/`, `.git`, BYOK sessions and other attempts' workspaces as **empty**
  (Linux mount namespaces, Python ≥ 3.12; `isolation.filesystem` in the config).
* The folder is **locked** (`chmod 000`) whenever the benchmark is not writing to it. Use
  `bench outputs list | path --model M | export --model M --to DIR | unlock | lock` to inspect it.
* Each attempt's `code_location.txt` artifact in the dashboard points to its folder.
* See [docs/security.md](docs/security.md#controls) for the guarantees and limits.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate      # Python >= 3.11
pip install -e ".[dev]"                             # or: uv sync --extra dev  (uv.lock)
bench validate-config
bench doctor --no-check-endpoints
bench list-tasks
bench run --model fake-reference --profile smoke    # MOCK adapter: validates the pipeline
bench run --model fake-noop --profile smoke         # MOCK: changes nothing -> must fail
bench report --format html                          # reports/report.html
bench dashboard                                     # http://127.0.0.1:8765
```

Then add a real model (see [docs/provider-setup.md](docs/provider-setup.md)):

```bash
ollama serve & ollama pull qwen2.5-coder:14b
bench dry-run --model local-qwen --profile standard  # estimate; calls nothing
bench run --model local-qwen --profile smoke
bench run --model local-qwen --profile standard --harness direct
bench compare --run-ids RUN_A,RUN_B
```

Optional runtimes: `pip install -e ".[torch]"`, `".[spark]"` (needs Java 17+), `".[anthropic]"`.
Tasks whose runtime is missing are reported as **skipped** with the reason - never passed.

## Commands

| Command | Purpose |
|---|---|
| `bench validate-config [--config --models]` | validate configs and the task catalogue (no side effects) |
| `bench list-models` / `bench list-tasks [--profile --domain]` | inventory, availability |
| `bench doctor [--model a,b]` | runtimes, tools, DB integrity, isolation, endpoint reachability (`GET /models` only) |
| `bench dry-run --model M [--profile --task --harness]` | tasks, time, request/token budget, cost bound; calls no provider |
| `bench run --model M [--profile] [--task ids] [--tag] [--domain] [--harness] [--repeats] [--concurrency] [--allow-remote] [--allow-external-harness] [--resume RUN] [-y]` | execute and persist |
| `bench runs` / `bench show --run-id R [--checks]` | history and per-check evidence |
| `bench compare --run-ids A,B` | side-by-side aggregates, per task, comparability warnings |
| `bench report [--run-id R] --format json\|csv\|html [--include-transcripts] [--include-source]` | redacted exports |
| `bench dashboard [--host --port]` | local dashboard (loopback by default) |
| `bench import-run --file F [--no-evaluate] [--keep-transcripts]` / `bench export-run` | manual/IDE runs (e.g. Copilot BYOK) |
| `bench byok start` / `bench byok finish` | one-command Copilot BYOK session (used by `setup.sh`, `finish.sh`) |
| `bench byok prepare` / `bench byok collect` | the same in individual steps with all options (`byok/*.py`) |
| `bench outputs list\|path\|export\|unlock\|lock` | per-model output folders (locked by default) |
| `bench perf discover\|probe` / `bench run --perf-probe` | model-server info, tokens/s, TTFT, GPU share, session token counters |
| `bench task validate PATH` | prove fixture fails, reference passes, broken fails |
| `bench review export\|import\|judge` | blind human review, judge models (stored separately) |

Exit codes: `0` ok, `1` validation problems, `2` config/usage error, `3` run interrupted
(resume with `--resume`), `4` run failed. Ctrl-C/SIGTERM mark running attempts `interrupted`.

## Profiles

| Profile | Tasks | Intent |
|---|---|---|
| `smoke` | 4 (Python, SQL, debugging, C++) | adapter + repo sanity check in minutes |
| `standard` | 12 (adds config validation, perception x2, C++ association, DuckDB, PyTorch, CMake, code review) | representative; ~20-45 min per model on ordinary tasks |
| `extended` | all 14 (+ Spark, design reasoning), 2 repeats | deeper; asks for confirmation (time/cost) |

Budgets per profile (task timeout, whole-run timeout, max turns, token budget, repeats) live in
`configs/benchmark.example.yaml`. Concurrency defaults to 1 attempt at a time, with per-provider
concurrency and requests-per-minute limits.

## Task catalogue

| Task | Domain | Evaluation |
|---|---|---|
| python.fix-interval-merge | python | hidden pytest, regression-test detection, ruff |
| python.typed-config-loader | production | API contract with error paths, malformed-input fuzz |
| debugging.timestamp-sync | debugging | reproduce seeded defect, root cause, regression test |
| perception.track-lifecycle | perception | tentative/confirmed/deleted lifecycle, global NN association |
| perception.frame-transform | perception | SE(2) transforms, yaw wrap interpolation (numpy) |
| cpp_sensor_fusion.kalman-cv | C++17 | CV Kalman filter, Joseph form, input validation |
| cpp_sensor_fusion.gated-association | C++17 | gated global-greedy assignment, NaN/inf, ties |
| sql.latest-sensor-reading | SQL (SQLite) | dedup + NULL handling, exact result set |
| sql.sensor-gaps | SQL (DuckDB) | window functions, time-series gaps |
| pytorch.masked-metrics | PyTorch (CPU) | masked mean/loss, IoU, NaN-safe gradients |
| spark.dedup-partition | PySpark | deterministic dedup, schema checks (extended) |
| build.cmake-ring-buffer | build | linker error, CTest registration, wrap-around bug |
| code_review.log-upload-patch | code review | text rubric (+ blind human review) |
| reasoning.fusion-tradeoff | reasoning | text rubric proxy; intended for human review (extended) |

## Documentation

* [docs/architecture.md](docs/architecture.md) - plan, data flow, data contracts, extension points, design trade-offs
* [docs/security.md](docs/security.md) - threat model, controls, residual risks
* [docs/evaluation-methodology.md](docs/evaluation-methodology.md) - scoring, fairness, statistics, interpretation
* [docs/provider-setup.md](docs/provider-setup.md) - Ollama, vLLM, OpenAI-compatible, Anthropic, CLI harnesses, manual import
* [docs/task-authoring.md](docs/task-authoring.md) - adding tasks, checks, providers, harnesses

## Development

```bash
ruff check src tests && ruff format --check src tests
mypy                     # strict, src/
pytest -q                # unit + integration (no credentials or GPU needed)
pytest -q -m "not integration"   # fast unit tests only
```
Dependencies are pinned in `uv.lock` (`uv sync --extra dev`); `pyproject.toml` keeps compatible
ranges for library use. All runtime dependencies are permissively licensed (MIT/BSD/Apache-2.0).
CI: `.github/workflows/ci.yml` (lint, types, tests on 3.11/3.12, smoke run, HTML report).

## Known limitations

* Hiding other models' folders needs Linux and Python ≥ 3.12 (mount namespaces); elsewhere
  only the `chmod 000` lock applies. Copilot BYOK agents run outside the sandbox (see byok/README.md).
* The command sandbox is defence in depth, not a security boundary; run untrusted models in a
  container or VM (see docs/security.md).
* GitHub Copilot BYOK cannot be driven programmatically; use the `byok/` session workflow
  (or the lower-level `bench import-run`).
* External CLI harness support is generic (argv template); token usage from such tools is not
  parsed and is recorded as unknown.
* The OpenAI-compatible and Anthropic adapters are tested against mocked transports and a local
  scripted server; no paid API was called while building this repository.
* Starter catalogue is small (14 tasks); treat differences on a few tasks as noise.
