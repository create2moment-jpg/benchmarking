# Running a model through Copilot BYOK

```bash
./setup.sh          # in a terminal, inside this repository   (quick test: ./setup.sh smoke)
```

A new VS Code window opens on a fresh session folder. Then:

1. If VS Code asks whether you trust the folder's authors, answer **Yes** (needed for the
   auto-approved commands).
2. **Copilot Chat → Agent → pick your model.**
3. **Paste (Ctrl+V) → Enter.** The prompt ([command.md](command.md)) is on the clipboard.

When the agent replies `ALL TASKS COMPLETE`:

```bash
./finish.sh         # scores the run and opens the dashboard
```

For the next model, run `./setup.sh` again.

## Troubleshooting

| What you see | Fix |
|---|---|
| VS Code does not open | Install the `code` command (in VS Code: **Shell Command: Install 'code' command in PATH**), or open the printed folder with **File > Open Folder…** in a new window. |
| Prompt not on the clipboard | Copy it from the terminal output or from [command.md](command.md). |
| Agent says there is no `starting_point.md` / `NO SESSION FOLDER OPEN` | The wrong folder is open (e.g. this repository). Use the window `./setup.sh` opened: its Explorer shows `starting_point.md`, `AGENTS.md`, `tasks/`. |
| Still asked to **Allow** commands | The folder is not trusted (**Manage Workspace Trust** → Trust), or your VS Code version ignores workspace auto-approval: use the approval picker under the chat input. `rm`, `git`, `pip`, `curl`, `find` and similar always ask on purpose. |
| "Continue to iterate?" | Click **Continue**. |
| Agent stops before finishing | Send: `Continue with the next unfinished task in starting_point.md.` |
| "No module named pytest" in the agent's terminal | Open a new terminal in that window; the session selects `~/elb-byok-sessions/.runtime-venv`. |
| `finish.sh` asks which model ran | Type the name you picked in Copilot, or pass `--model-label NAME`. |
| `setup.sh`: "has not been collected yet" | Run `./finish.sh` first (or delete that session folder to discard it). |

Auto-approval lets the agent run the task commands (including `python`, which can run any code)
as your user without asking. Package managers, network tools, git and deletion still ask.
