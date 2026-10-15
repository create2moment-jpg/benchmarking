"""Command runner policy: allowlist, argument checks, timeouts, output caps, env scrubbing,
network isolation; workspace path safety and diffs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from engineering_llm_bench.config.models import LimitsConfig
from engineering_llm_bench.control import CancelToken
from engineering_llm_bench.tools.runner import (
    LOCAL_BINARIES,
    CommandRefusedError,
    CommandRunner,
    mount_hiding_available,
    network_namespace_available,
)
from engineering_llm_bench.workspace.manager import (
    PathEscapeError,
    is_editable,
    resolve_inside,
    unified_diff,
)


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


def run(runner: CommandRunner, ws: Path, argv: list[str], **kw: object):  # type: ignore[no-untyped-def]
    return runner.run(
        argv, cwd=ws, root=ws, scratch=ws.parent / "scratch", allowed={"python"}, **kw
    )  # type: ignore[arg-type]


def test_not_allowlisted(ws: Path) -> None:
    with pytest.raises(CommandRefusedError, match="not allowed"):
        run(CommandRunner(LimitsConfig()), ws, ["bash", "-c", "echo hi"])


@pytest.mark.parametrize("arg", ["../secret.txt", "/etc/passwd", "--file=/etc/shadow", "-o/tmp/x"])
def test_unsafe_arguments_refused(ws: Path, arg: str) -> None:
    with pytest.raises(CommandRefusedError):
        run(CommandRunner(LimitsConfig()), ws, ["python", arg])


def test_absolute_path_inside_workspace_ok(ws: Path) -> None:
    (ws / "a.py").write_text("print('inside')")
    res = run(CommandRunner(LimitsConfig()), ws, ["python", str(ws / "a.py")])
    assert res.ok and "inside" in res.stdout


def test_timeout_kills_process(ws: Path) -> None:
    res = run(
        CommandRunner(LimitsConfig(command_timeout_s=1)),
        ws,
        ["python", "-c", "import time; time.sleep(30)"],
    )
    assert res.timed_out and res.exit_code is None and res.duration_s < 10


def test_cancellation(ws: Path) -> None:
    token = CancelToken()
    token.cancel("stop")
    res = run(
        CommandRunner(LimitsConfig()),
        ws,
        ["python", "-c", "import time; time.sleep(30)"],
        cancel=token,
    )
    assert res.cancelled and not res.ok


def test_output_truncated(ws: Path) -> None:
    res = run(
        CommandRunner(LimitsConfig(max_output_bytes=1000)),
        ws,
        ["python", "-c", "print('x' * 100000)"],
    )
    assert res.truncated and len(res.stdout) < 2000


def test_environment_scrubbed(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak-000000000000")
    res = run(
        CommandRunner(LimitsConfig()), ws, ["python", "-c", "import os; print(sorted(os.environ))"]
    )
    assert "OPENAI_API_KEY" not in res.stdout
    assert "HOME" in res.stdout


def test_memory_limit(ws: Path) -> None:
    res = run(
        CommandRunner(LimitsConfig(memory_mb=256)),
        ws,
        ["python", "-c", "x = bytearray(1024 * 1024 * 1024)"],
    )
    assert res.exit_code != 0


@pytest.mark.skipif(not network_namespace_available(), reason="no network namespaces here")
def test_network_denied_but_loopback_works(ws: Path) -> None:
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 443), timeout=3); print('EXTERNAL OPEN')\n"
        "except OSError:\n"
        "    print('external blocked')\n"
    )
    res = run(CommandRunner(LimitsConfig()), ws, ["python", "-c", code])
    assert "external blocked" in res.stdout


def test_local_binary_requires_opt_in(ws: Path) -> None:
    script = ws / "tool.sh"
    script.write_text("#!/bin/sh\necho local\n")
    script.chmod(0o755)
    runner = CommandRunner(LimitsConfig())
    with pytest.raises(CommandRefusedError):
        runner.run(["./tool.sh"], cwd=ws, root=ws, scratch=ws.parent / "s", allowed={"python"})
    res = runner.run(
        ["./tool.sh"], cwd=ws, root=ws, scratch=ws.parent / "s", allowed={LOCAL_BINARIES}
    )
    assert res.stdout.strip() == "local"
    with pytest.raises(CommandRefusedError):
        runner.run(
            ["./../escape"], cwd=ws, root=ws, scratch=ws.parent / "s", allowed={LOCAL_BINARIES}
        )


def test_resolve_inside_rejects_traversal_and_symlinks(ws: Path, tmp_path: Path) -> None:
    for bad in ["../x", "/etc/passwd", "a/../../x", ""]:
        with pytest.raises(PathEscapeError):
            resolve_inside(ws, bad)
    (tmp_path / "outside.txt").write_text("secret")
    os.symlink(tmp_path / "outside.txt", ws / "link")
    with pytest.raises(PathEscapeError):
        resolve_inside(ws, "link")
    assert resolve_inside(ws, "sub/file.py") == (ws / "sub/file.py").resolve()


def test_editable_patterns() -> None:
    assert is_editable("tests/test_a.py", ["tests/*.py"])
    assert not is_editable("src/a.py", ["tests/*.py"])


def test_unified_diff() -> None:
    diff = unified_diff({"a.py": b"x = 1\n"}, {"a.py": b"x = 2\n", "b.py": b"new\n"})
    assert "-x = 1" in diff and "+x = 2" in diff and "+++ b/b.py" in diff
    assert unified_diff({"a": b"\x00\x01"}, {"a": b"\x00\x02"}).startswith("Binary files")


def test_python_resolves_to_benchmark_interpreter(ws: Path) -> None:
    res = run(
        CommandRunner(LimitsConfig()), ws, ["python", "-c", "import sys; print(sys.executable)"]
    )
    assert Path(res.stdout.strip()).resolve() == Path(sys.executable).resolve()


LIST_DIRS = (
    "import os\n"
    "for d in {dirs!r}:\n"
    "    try:\n"
    "        print(d, sorted(os.listdir(d)))\n"
    "    except OSError as e:\n"
    "        print(d, 'ERR', e.errno)\n"
)


@pytest.mark.skipif(not mount_hiding_available(), reason="no mount namespaces here")
def test_hidden_paths_and_other_workspaces_are_invisible(tmp_path: Path) -> None:
    secret = tmp_path / "model_outputs"
    (secret / "other-model").mkdir(parents=True)
    (secret / "other-model" / "solution.py").write_text("x = 1\n")
    root = tmp_path / "workspaces"
    mine, other = root / "elb-a", root / "elb-b"
    (mine / "ws").mkdir(parents=True)
    (other / "ws").mkdir(parents=True)
    (other / "ws" / "code.py").write_text("y = 2\n")
    (mine / "ws" / "own.py").write_text("z = 3\n")
    runner = CommandRunner(LimitsConfig(), hidden_paths=(secret,), workspaces_root=root)
    res = runner.run(
        ["python", "-c", LIST_DIRS.format(dirs=[str(secret), str(other), str(mine / "ws")])],
        cwd=mine / "ws",
        root=mine / "ws",
        scratch=mine / "scratch",
        allowed={"python"},
    )
    assert res.ok, res.stderr
    lines = dict(line.split(" ", 1) for line in res.stdout.splitlines())
    assert lines[str(secret)] == "[]"
    assert lines[str(other)] == "[]"
    assert "own.py" in lines[str(mine / "ws")]
    # The host view is untouched.
    assert (secret / "other-model" / "solution.py").is_file()
    assert (other / "ws" / "code.py").is_file()


def test_filesystem_hide_is_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import engineering_llm_bench.tools.runner as runner_mod  # noqa: PLC0415

    monkeypatch.setattr(runner_mod, "mount_hiding_available", lambda: False)
    with pytest.raises(CommandRefusedError, match="filesystem=hide"):
        CommandRunner(LimitsConfig(), filesystem="hide")
    CommandRunner(LimitsConfig(), filesystem="hide_if_possible")  # degrades silently
