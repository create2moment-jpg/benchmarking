"""What a model's commands must not see, and the command runner that enforces it."""

from __future__ import annotations

from pathlib import Path

from engineering_llm_bench.config.models import BenchmarkConfig
from engineering_llm_bench.tools.runner import CommandRunner
from engineering_llm_bench.workspace.manager import workspaces_root

#: Default parent folder of Copilot BYOK sessions (see byok/).
BYOK_SESSIONS_DIR = Path.home() / "elb-byok-sessions"


def hidden_paths_for(
    tasks_dir: Path, artifacts_dir: Path, outputs_dir: Path | None = None
) -> tuple[Path, ...]:
    """Directories hidden from every command run for a model:

    * per-model output folders (other models' code) and the results store (diffs, traces, DB);
    * the task catalogue (reference solutions, hidden evaluator files) and the repository's
      ``.git`` (which also contains them);
    * Copilot BYOK session folders.
    Other attempts' workspaces are hidden separately via the workspaces root.
    """
    paths = [p for p in (outputs_dir, artifacts_dir, tasks_dir) if p is not None]
    git = tasks_dir.resolve().parent / ".git"
    if git.is_dir():
        paths.append(git)
    paths.append(BYOK_SESSIONS_DIR)
    return tuple(paths)


def make_command_runner(bench: BenchmarkConfig, hidden: tuple[Path, ...]) -> CommandRunner:
    return CommandRunner(
        bench.limits,
        bench.isolation.network,
        hidden_paths=hidden,
        workspaces_root=workspaces_root(),
        filesystem=bench.isolation.filesystem,
    )
