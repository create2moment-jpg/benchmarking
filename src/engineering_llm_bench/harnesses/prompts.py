"""Prompt construction. Identical task contracts are presented to every model."""

from __future__ import annotations

from engineering_llm_bench.catalog.registry import Task
from engineering_llm_bench.workspace.manager import iter_files

AGENT_SYSTEM = """You are a software engineer working in an isolated repository workspace.
You act by replying with EXACTLY ONE JSON object per message (no other text), one of:
  {"action": "list_files", "path": "."}
  {"action": "read_file", "path": "relative/path"}
  {"action": "write_file", "path": "relative/path", "content": "<complete new file content>"}
  {"action": "run_command", "argv": ["pytest", "-q"]}
  {"action": "finish", "summary": "<what changed, how you verified it, limitations>"}
Rules:
- Paths are relative to the workspace root. write_file replaces the whole file.
- run_command takes an argv list (no shell). Only these commands are allowed: {commands}.
- You may only write files matching: {editable}.
- Work in small steps: inspect, plan, edit, run tests, fix failures, then finish.
- You have at most {turns} actions. Finish before running out.
- Hidden tests evaluate your work after you finish; your own claims are not trusted.
"""

DIRECT_SYSTEM = """You are a software engineer. You will receive a task and the repository files.
Reply with the COMPLETE content of every file you create or change, each in a block:
<file path="relative/path">
...complete file content...
</file>
Then end with <summary>what you changed, assumptions, limitations</summary>.
Only write files matching: {editable}. Do not include unchanged files.
"""

_MAX_INLINE_FILE = 60_000
_MAX_INLINE_TOTAL = 240_000


def task_contract(task: Task) -> str:
    spec = task.spec
    criteria = "\n".join(f"- {c}" for c in spec.acceptance_criteria)
    artifacts = (
        "\nExpected artifacts: " + ", ".join(spec.expected_artifacts)
        if spec.expected_artifacts
        else ""
    )
    return (
        f"# Task {spec.id} (v{spec.version}): {spec.title}\n\n{spec.prompt.strip()}\n\n"
        f"## Acceptance criteria\n{criteria}{artifacts}\n"
    )


def file_listing(task: Task) -> str:
    files = [p.relative_to(task.fixture_dir).as_posix() for p in iter_files(task.fixture_dir)]
    return "\n".join(files) if files else "(empty)"


def agent_system_prompt(task: Task, allowed_commands: set[str], max_turns: int) -> str:
    # str.replace, not str.format: the template contains literal JSON braces.
    return (
        AGENT_SYSTEM.replace("{commands}", ", ".join(sorted(allowed_commands)) or "(none)")
        .replace("{editable}", ", ".join(task.spec.editable_paths))
        .replace("{turns}", str(max_turns))
    )


def agent_user_prompt(task: Task) -> str:
    return f"{task_contract(task)}\n## Workspace files\n{file_listing(task)}\n"


def direct_system_prompt(task: Task) -> str:
    return DIRECT_SYSTEM.replace("{editable}", ", ".join(task.spec.editable_paths))


def direct_user_prompt(task: Task) -> str:
    parts = [task_contract(task), "## Repository files\n"]
    total = 0
    for path in iter_files(task.fixture_dir):
        rel = path.relative_to(task.fixture_dir).as_posix()
        data = path.read_bytes()
        if (
            b"\0" in data[:4096]
            or len(data) > _MAX_INLINE_FILE
            or total + len(data) > _MAX_INLINE_TOTAL
        ):
            parts.append(f'<file path="{rel}" omitted="too large or binary"/>\n')
            continue
        total += len(data)
        parts.append(f'<file path="{rel}">\n{data.decode("utf-8", "replace")}</file>\n')
    return "\n".join(parts)


def external_prompt(task: Task) -> str:
    """Prompt for external agent CLIs, which use their own tools inside the workspace."""
    return (
        f"{task_contract(task)}\nWork in the current directory. Only modify files matching: "
        f"{', '.join(task.spec.editable_paths)}. When done, print a short summary of what you "
        "changed, how you verified it, and any limitations.\n"
    )
