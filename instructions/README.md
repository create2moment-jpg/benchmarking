# Instructions for the model under test

`command.md` is the prompt you send to the agent (GitHub Copilot agent mode with your BYOK
model). It is the same for every model, so results stay comparable, and it names no model or
serving stack.

## Run everything

```bash
./byok/start.sh          # in a terminal, inside this repository (smoke: ./byok/start.sh smoke)
```

A new VS Code window opens with the session folder and the prompt is on the clipboard (if no
clipboard tool is installed, it is printed: copy it from the terminal or from `command.md`).

1. In that window: **Copilot Chat → Agent mode → pick your BYOK model** (the only choice
   you make).
2. Paste (Ctrl+V) and press Enter. Nothing needs to be attached: the prompt points the agent
   at `starting_point.md` in the open folder.
3. If it stops before replying `ALL TASKS COMPLETE`, send:
   `Continue with the next unfinished task in starting_point.md, following the same rules.`

```bash
./byok/finish.sh         # detects the model, scores, opens the dashboard
```

If `start.sh` cannot open VS Code (no `code` command on the PATH: in VS Code run
**Shell Command: Install 'code' command in PATH**), open the printed folder with
**File > Open Folder…** in a **new window**.

## Troubleshooting

| What you see | Cause and fix |
|---|---|
| "No starting_point.md exists … this workspace is the benchmark harness" | VS Code has this repository open. Use the window `./byok/start.sh` opened (its Explorer shows `starting_point.md`, `AGENTS.md`, `tasks/`). The repository's `AGENTS.md` tells agents to stop here instead of searching. |
| Agent replies `NO SESSION FOLDER OPEN` | Same cause: the open workspace has no `starting_point.md` at its root. |
| A **Run bash command? Allow / Skip** box for every command | That is VS Code's tool approval, not the benchmark; the prompt cannot change it. You can click **Allow** each time; use the arrow next to **Allow** to allow that command for the session; or change the approval level in the picker under the chat input ("Default Approvals"). Decide this yourself: auto-approved commands run as your user, outside the benchmark's sandbox. Approve only the commands the tasks need (`python`, `pytest`, `ruff`, `g++`, `cmake`, `ctest`, `./<test binary>`), never `rm`, `curl`, `pip` or `git`. |
| "Continue to iterate?" after many steps | VS Code's per-request step limit; click **Continue**, or raise `chat.agent.maxRequests` in your user settings. |
| Agent stops after one task | Send the continue message from step 3. |
| `python -m pytest`: "No module named pytest" in the agent's terminal | The session points VS Code at `~/elb-byok-sessions/.runtime-venv`; open a new terminal, or the agent uses that interpreter's full path (listed in `AGENTS.md`). |
| `finish.sh` asks which model ran | The server no longer has exactly one model loaded; type the name you picked in Copilot, or pass `--model-label`. |
| `start.sh`: "has not been collected yet" | Run `./byok/finish.sh` for the previous session first (or delete that folder to discard it). |

`bench byok prepare` prints the same prompt; `src/engineering_llm_bench/byok/command.md` is the
packaged copy (a test keeps both identical).
