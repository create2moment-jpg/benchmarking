# Starting point: engineering tasks

You are a software engineer working autonomously in this folder. It contains
12 independent engineering tasks. Complete **all** of them, one at a time, in
the order below. Do not wait for confirmation between tasks and do not ask clarifying
questions: make a reasonable assumption, write it down in that task's `RESULT.md`, and continue.

## Tasks

| # | Folder | Task | Domain | Est. |
|---|--------|------|--------|------|
| 1 | `tasks/01_build.cmake-ring-buffer/` | Fix a CMake build, register tests, and fix a wrap-around bug | build | ~4 min |
| 2 | `tasks/02_code_review.log-upload-patch/` | Security and reliability review of a log-upload patch | code_review | ~2 min |
| 3 | `tasks/03_cpp_sensor_fusion.gated-association/` | Gated global-greedy track-to-detection association (C++17) | cpp_sensor_fusion | ~3 min |
| 4 | `tasks/04_cpp_sensor_fusion.kalman-cv/` | Constant-velocity Kalman filter (C++17) | cpp_sensor_fusion | ~3 min |
| 5 | `tasks/05_debugging.timestamp-sync/` | Reproduce and fix a sensor timestamp association defect | debugging | ~2 min |
| 6 | `tasks/06_perception.frame-transform/` | SE(2) frame transforms and pose interpolation | perception | ~3 min |
| 7 | `tasks/07_perception.track-lifecycle/` | Track lifecycle and global nearest-neighbour association | perception | ~4 min |
| 8 | `tasks/08_python.fix-interval-merge/` | Fix sensor-dropout interval merging | python | ~1 min |
| 9 | `tasks/09_python.typed-config-loader/` | Validate a typed sensor-rig configuration with precise error paths | production | ~3 min |
| 10 | `tasks/10_pytorch.masked-metrics/` | Masked reductions, loss and box IoU for padded batches (PyTorch, CPU) | pytorch | ~3 min |
| 11 | `tasks/11_sql.latest-sensor-reading/` | Latest valid reading per active sensor (dedup + nulls) | sql | ~2 min |
| 12 | `tasks/12_sql.sensor-gaps/` | Heartbeat gap detection with window functions (DuckDB) | sql | ~2 min |

Estimated total: ~32 minutes of focused work.

## How to work on each task

1. Open `<folder>/TASK.md` and read it completely. It is the full task contract: the
   problem, the acceptance criteria, which files you may edit, and which commands you may run.
2. Read the existing code in that folder before changing anything.
3. Make the change. Edit **only** the files that `TASK.md` lists as editable, and only inside
   that task's folder.
4. Verify your work with the commands listed in `TASK.md` (for example
   `python -m pytest -q`, or compiling and running the C++ test). Run commands from inside the
   task folder. Fix failures and re-run until you are satisfied.
5. Write `<folder>/RESULT.md` using the template below, then move on to the next task.

## Rules

- Work only inside `tasks/`. Never modify `starting_point.md`, any `TASK.md`, or the `.bench/`
  folder.
- Never edit a file that `TASK.md` does not list as editable. Changes to other files are
  discarded and not evaluated. Do not weaken or delete existing tests to make them pass.
- Hidden tests evaluate every task after you finish. Your own claims in `RESULT.md` are not
  used for scoring, so be honest about what does and does not work.
- Do not search for hidden tests or reference solutions; they are not on this machine.
- Do not install packages and do not use the internet. If a listed command is not available,
  say so in `RESULT.md` and continue.
- Keep changes focused and production quality: clear code, edge cases handled, no debug
  prints left behind.
- If you cannot finish a task, still write `RESULT.md` with `Status: partial` (or `not done`)
  and explain why, then continue with the next task.

## RESULT.md template (one per task folder)

```markdown
Status: done | partial | not done

## Changes
- <file>: <what changed and why>

## Verification
- <command you ran> -> <result>

## Assumptions and limitations
- <anything you assumed, anything not handled>
```

When every task folder has a `RESULT.md`, reply with: `ALL TASKS COMPLETE` and a one-line
status per task.

<!-- session 20261015-132150-byok-d1b921 -->
