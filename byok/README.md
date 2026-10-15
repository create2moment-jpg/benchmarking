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

## 1. Prepare a session (in the repo)

```bash
pip install -e ".[dev]"            # once; the scripts also work without installing
python byok/prepare_session.py --profile standard --out ~/elb-byok-sessions/qwen38-run1
# or: --profile smoke (4 tasks), --task python.fix-interval-merge,sql.sensor-gaps
```

This creates (outside the repository, so the agent cannot read hidden tests or reference
solutions):

```
qwen38-run1/
  starting_point.md            <- give this to the agent
  tasks/01_<task>/TASK.md      <- one folder per task: code + full task contract
  tasks/02_<task>/...
  .bench/session.json          <- manifest (a trusted copy is kept in artifacts/byok_sessions/)
```

See [starting_point.example.md](starting_point.example.md) for what the agent receives
(standard profile).

## 2. Let the agent work (in VS Code)

1. **File > Open Folder…** and open the *session folder* (not this repository).
2. Open Copilot Chat, switch to **Agent** mode, and select your BYOK model in the model picker
   (`Manage Models…` if it is not listed yet).
3. Attach `starting_point.md` (drag it into the chat or use `#file:starting_point.md`) and send:
   > Follow starting_point.md and complete all tasks.
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

Each new benchmark of a model (or of a different model) needs a **fresh session**: run
`prepare_session.py` again with a new `--out` folder.

## Comparing models

Run one session per model (same profile), collect each with its own `--model-label`, then use
the dashboard's **Compare** tab or:

```bash
bench runs
bench compare --run-ids RUN_A,RUN_B
bench report --format html --out reports/byok.html
```

## Notes and limitations

* Token usage and cost are not visible to this tool through Copilot; they are stored as
  unknown. Per-task time is unknown; `--session-minutes` records the whole session.
* Copilot's own system prompt, context management and tool behaviour are part of what you
  measure. Results are a **model + Copilot harness** combination; compare them with other
  models through the same harness.
* The collecting machine needs the evaluator runtimes (Python with pytest, g++, cmake, numpy,
  duckdb, torch for the PyTorch task). Tasks whose runtime is missing are reported as skipped.
