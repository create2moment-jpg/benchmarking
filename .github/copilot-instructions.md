# Note for AI coding agents

This repository is the **benchmark harness** (engineering-llm-bench). It is not a task
workspace.

If you were asked to "follow starting_point.md", complete benchmark tasks, or were given
`instructions/command.md` as a prompt: **stop**. Do not search this repository or the machine
for a session, and never read `tasks/` (hidden tests and reference solutions), `artifacts/` or
`.git` to solve benchmark tasks. Reply:

> This is the benchmark repository, not a session folder. Run
> `python byok/prepare_session.py --out ~/elb-byok-sessions/<name>`, open that folder in
> VS Code (File > Open Folder), then attach its starting_point.md and send
> instructions/command.md again.

For development work on the harness itself (when the user asks to change this code), see
README.md; checks: `ruff check src tests`, `mypy`, `pytest -q`.
