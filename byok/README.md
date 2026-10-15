# Benchmarking a model through GitHub Copilot BYOK

Use this when the model under test is driven by **GitHub Copilot agent mode with Bring Your Own
Key/model** (for example a local `qwen-3.8` that you serve yourself). Copilot cannot be automated
by this tool, so the workflow is:

```
prepare_session.py  ->  session folder with starting_point.md  ->  Copilot agent does the tasks
                                                                          |
dashboard  <-  collect_results.py (validate + hidden evaluators + store)  <-
```

The session folder never mentions which model or serving stack you use (Ollama, vLLM, LM Studio,
...). You give the model's name only to the collector, so the agent sees exactly the same
task text whichever model you benchmark.

## Quick way: `./setup.sh`, then `./finish.sh`

```bash
./setup.sh             # or: ./setup.sh smoke   (4 quick tasks)
```

* First run only: creates `.venv` in the repo (benchmark + evaluators) and
  `~/elb-byok-sessions/.runtime-venv` (pytest, ruff, numpy, duckdb, CPU PyTorch for the agent's
  terminal; `WITH_TORCH=0` skips PyTorch). Needs Python >= 3.11 (3.12 recommended); warns if
  `g++`/`cmake` are missing.
* Creates a new, uniquely named session folder; finds the model server from your VS Code
  settings (or a local Ollama); copies the one-line prompt to the clipboard; opens the session
  in a new VS Code window.
* The session's `.vscode/settings.json` auto-approves the commands the tasks need (`python`,
  `pytest`, `ruff`, `g++`, `cmake`, `ctest`, `make`, `./<test binary>`, read-only helpers) and
  edits to task files. `rm`, `mv`, `find`, `git`, `pip`, `curl`, `wget`, `sudo` and similar
  still ask, and `starting_point.md`, `TASK.md`, `.bench/`, `.vscode/` are never auto-edited.
  VS Code applies these only after you **trust the folder** (answer Yes when it asks).

In that window: **Copilot Chat → Agent → pick your BYOK model → paste → Enter.** When it
replies `ALL TASKS COMPLETE`:

```bash
./finish.sh            # or: ./finish.sh --model-label "qwen3.8-next"
```

It detects the model that ran (the one loaded on the server; asks you if it cannot tell),
scores the session, moves the code into the model's private folder, measures the server and
opens the dashboard. For the next model, run `./setup.sh` again.

**Auto-approval note:** an approved `python` can run any Python code, so auto-approval means
the agent runs code as your user without asking. Do it only for models you are willing to let
do that, or use `bench byok start` without `--auto-approve` and click Allow yourself.

The sections below are the same workflow in individual steps, with all options.

## 1. Prepare a session (in the repo)

```bash
pip install -e ".[dev]"            # once; the scripts also work without installing
python byok/prepare_session.py --profile standard --out ~/elb-byok-sessions/qwen38-run1 \
    --byok-config ~/.config/Code/User/settings.json      # where your BYOK endpoint is configured
# or: --endpoint http://localhost:11434 (Ollama) / http://localhost:8000/v1 (vLLM, llama.cpp)
# or: --profile smoke (4 tasks), --task python.fix-interval-merge,sql.sensor-gaps
```

`--byok-config` reads the JSON (comments allowed) where Copilot's BYOK endpoint is configured
and finds the model server's URL and model ids; `bench perf discover --byok-config FILE` shows
what it found. If several endpoints are configured, pick one with `--endpoint` / `--model-id`.
The endpoint is stored with the benchmark's private session record, **not** in the session folder
the agent sees. Endpoint options are optional: without them you still get task scores and
time estimates, just no server statistics.

This creates (outside the repository, so the agent cannot read hidden tests or reference
solutions):

```
qwen38-run1/
  starting_point.md            <- give this to the agent
  AGENTS.md, .github/copilot-instructions.md   <- always-on rules (stay in this folder, ...)
  tasks/01_<task>/TASK.md      <- one folder per task: code + full task contract
  tasks/02_<task>/...
  .bench/session.json          <- manifest (a trusted copy is kept in artifacts/byok_sessions/)
```

See [starting_point.example.md](starting_point.example.md) for what the agent receives
(standard profile).

## 2. Let the agent work (in VS Code)

Right before you hand over `starting_point.md`, record the start (start time for per-task time
estimates, and a fresh "before" snapshot of the server's token counters):

```bash
bench byok mark-start ~/elb-byok-sessions/qwen38-run1
```

1. **File > Open Folder…** and open the *session folder* (not this repository - if the agent
   says there is no starting_point.md, the wrong folder is open). Troubleshooting, including
   the per-command **Allow** prompts: [instructions/README.md](../instructions/README.md).
2. Open Copilot Chat, switch to **Agent** mode, and select your BYOK model in the model picker
   (`Manage Models…` if it is not listed yet).
3. Attach `starting_point.md` (drag it into the chat or use `#file:starting_point.md`) and send
   the prompt in [instructions/command.md](../instructions/command.md) (also printed by
   `prepare`). Use the same prompt for every model.
4. Let it run. Allow it to run terminal commands (pytest, g++, cmake) when asked; its terminal
   needs the tools listed in each `TASK.md`. If it stops early, say "continue with the next
   task in starting_point.md".
5. Optionally time the session (start to finish) for `--session-minutes`.

Each task folder should end with a `RESULT.md` written by the agent.

## 3. Collect, validate and score (in the repo)

```bash
python byok/collect_results.py ~/elb-byok-sessions/qwen38-run1 \
    --model-label "strata/qwen-3.8" --provider-label ollama --local \
    --copilot-version "1.x" --session-minutes 35 --dry-run      # check first
python byok/collect_results.py ~/elb-byok-sessions/qwen38-run1 \
    --model-label "strata/qwen-3.8" --provider-label ollama --local \
    --copilot-version "1.x" --session-minutes 35 --serve        # score + open dashboard
```

What the collector does:

* **Validates** the session: the manifest must match the trusted copy, every task folder is
  compared with the pristine task fixture, and a table shows what the agent changed, whether
  `RESULT.md` exists, and the status the agent reported.
* **Applies only allowed changes**: edits to files outside each task's editable list (for
  example a planted `conftest.py`, edited tests that are not editable, or a modified `TASK.md`)
  are ignored and reported. Build outputs and binaries are ignored.
* **Scores independently**: the changes are copied into a fresh copy of the task and the
  hidden tests / evaluators run there. What the agent claims in `RESULT.md` is stored as its
  summary but never trusted for pass/fail.
* **Stores a run** (`mode = manual`, harness `copilot-byok`, model label as given) in
  `artifacts/bench.db`, so `bench dashboard` shows it next to your other models. Refresh the
  dashboard if it is already running; `--serve` starts it.
* Tasks the agent did not touch count as **failed** (it was asked to do them). Use
  `--only-attempted` to mark them skipped instead.
* Collecting the same session twice is refused (it would double-count); use `--force` if you
  really mean it.

* **Moves the agent's code into the model's private folder**:
  `artifacts/model_outputs/<model>/byok_sessions/<session_id>/` (raw session) and
  `.../runs/<run_id>/<task>__r0/` (final code, diff, meta). It then deletes the session's
  `tasks/` and `starting_point.md` (a `COLLECTED.md` note stays), so the next model's agent
  cannot find this model's code. Pass `--keep-session` to keep the working copy. A forced
  re-collection (`--force`) reads from the archive.
* The output folder is locked (`chmod 000`) while the benchmark is not writing to it; browse
  it with `bench outputs list`, `bench outputs export --model "strata/qwen-3.8" --to DIR`.

Each new benchmark of a model (or of a different model) needs a **fresh session**: run
`prepare_session.py` again with a new `--out` folder. **Collect one session before preparing the
next one.** Copilot's agent runs in VS Code outside the benchmark's sandbox, so it is only kept
away from other models' code because finished sessions are removed and the output folder is
locked. `prepare` warns when a sibling session still holds an uncollected working copy. For
stronger separation use a separate OS user (or container/VM) per model.

## Performance statistics (secondary to task outcomes)

Copilot does not expose token usage, so the collector measures the **model server** directly
(only GET requests plus a fixed calibration prompt set; never your code):

| Statistic | Source | Ollama | vLLM | llama.cpp |
|---|---|---|---|---|
| model, quantization, parameter count, context length | `/api/show`, `/api/tags`, `/v1/models`, `/props` | yes | id, max len | ctx |
| share of the model resident on GPU | `/api/ps` (`size_vram / size`) | yes | - | - |
| generation tokens/s, prompt tokens/s, cold load time | calibration; Ollama reports server-side ns timings | server-side | client-timed | client-timed |
| time to first token | calibration (streaming) | yes | yes | yes |
| **tokens used during the session** | `/metrics` counters, before vs after (calibration excluded) | not available | yes | with `--metrics` |
| GPU utilisation / memory peak | `nvidia-smi` sampled during calibration (local server) | if available | if available | if available |
| per-task active time | file modification times of each task's edits and RESULT.md | estimate | estimate | estimate |

Calibration runs with `--calibration-runs N` (default 3, `0` = server info only) when you
collect, after the agent is done, so it never competes with the session. Everything not
measured is shown as "unknown" - nothing is estimated except the clearly labelled time estimates.
Results appear in the dashboard's **Compare** tab ("Model server, throughput and time profile")
and in each attempt's detail view. Probe any endpoint on its own with
`bench perf probe --endpoint URL [--model-id M] [--out probe.json]`.

## Comparing models

Run one session per model (same profile), collect each with its own `--model-label`, then use
the dashboard's **Compare** tab or:

```bash
bench runs
bench compare --run-ids RUN_A,RUN_B
bench report --format html --out reports/byok.html
```

## Notes and limitations

* Copilot does not expose usage; session token counts come from the server's `/metrics`
  (vLLM, llama.cpp) and are unknown for Ollama. Counters include any other traffic the server
  handled during the session. Per-task times are estimates from file timestamps;
  `--session-minutes` records your own measurement of the whole session.
* Calibration tokens/s measures the server under a fixed load, not every Copilot request.
* Copilot's own system prompt, context management and tool behaviour are part of what you
  measure. Results are a **model + Copilot harness** combination; compare them with other
  models through the same harness.
* The collecting machine needs the evaluator runtimes (Python with pytest, g++, cmake, numpy,
  duckdb, torch for the PyTorch task). Tasks whose runtime is missing are reported as skipped.
