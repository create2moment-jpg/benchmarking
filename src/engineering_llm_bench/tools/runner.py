"""Sandboxed command runner used for both agent tool calls and evaluator commands.

Safety properties (defence in depth - see docs/security.md for the threat model):

* structured argv only (never ``shell=True``); executable must be on an allowlist and is
  resolved to an absolute path by the runner, not by the caller;
* arguments containing ``..`` path segments or absolute paths outside the workspace are refused;
* scrubbed environment: no host secrets, no proxy variables; ``HOME``/``TMPDIR`` point to a
  per-attempt scratch directory;
* wall-clock timeout (process group killed), CPU / address-space / file-size rlimits;
* stdout/stderr captured to files and truncated to ``max_output_bytes``;
* optional network denial via an unprivileged Linux user+network namespace (``unshare -rn``).

These controls bound accidents and casual misuse. They are not a substitute for a container or
VM when running untrusted models against sensitive hosts.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO

from engineering_llm_bench.config.models import LimitsConfig
from engineering_llm_bench.control import CancelToken

_POLL_S = 0.02


class CommandRefusedError(Exception):
    """The command violates policy (not allowlisted, unsafe argument, missing executable)."""


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    cancelled: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.cancelled

    def summary(self, max_chars: int = 4000) -> str:
        """Compact text used as an agent observation and in evidence logs."""
        status = (
            "TIMEOUT"
            if self.timed_out
            else "CANCELLED"
            if self.cancelled
            else f"exit={self.exit_code}"
        )
        body = (self.stdout + ("\n[stderr]\n" + self.stderr if self.stderr.strip() else "")).strip()
        if len(body) > max_chars:
            half = max_chars // 2
            body = body[:half] + "\n...[truncated]...\n" + body[-half:]
        return f"$ {' '.join(self.argv)}\n[{status}, {self.duration_s:.1f}s]\n{body}"


@functools.cache
def network_namespace_available() -> bool:
    """True when ``unshare -rn`` works for this user (Linux only)."""
    exe = shutil.which("unshare")
    if exe is None or sys.platform != "linux":
        return False
    try:
        proc = subprocess.run([exe, "-rn", "true"], capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _python_module_cmd(module: str) -> list[str] | None:
    if importlib.util.find_spec(module) is None:
        return None
    return [sys.executable, "-m", module]


def resolve_executable(name: str) -> list[str]:
    """Map an allowlisted logical command name to an absolute argv prefix.

    ``python``/``pytest``/``ruff`` resolve to the benchmark interpreter so evaluations use the
    pinned environment rather than whatever is first on PATH.
    """
    if name in {"python", "python3"}:
        return [sys.executable]
    if name in {"pytest", "ruff"}:
        cmd = _python_module_cmd(name)
        if cmd is not None:
            return cmd
    path = shutil.which(name)
    if path is None:
        raise CommandRefusedError(f"executable {name!r} is not installed")
    return [path]


def _check_argument(arg: str, root: Path, cwd: Path) -> None:
    candidates = [arg]
    if arg.startswith("-") and "=" in arg:
        candidates.append(arg.split("=", 1)[1])
    elif arg.startswith(("-I", "-L", "-o")) and len(arg) > 2:
        candidates.append(arg[2:])
    for value in candidates:
        if "\x00" in value:
            raise CommandRefusedError("NUL byte in argument")
        if ".." in PurePosixPath(value).parts:
            raise CommandRefusedError(f"argument {arg!r} contains '..'")
        if value.startswith("/"):
            resolved = Path(value).resolve()
            if not resolved.is_relative_to(root.resolve()):
                raise CommandRefusedError(f"absolute path {value!r} is outside the workspace")
    if not cwd.resolve().is_relative_to(root.resolve()):
        raise CommandRefusedError("working directory escapes the workspace")


@dataclass
class CommandRunner:
    """Runs allowlisted commands inside a workspace with limits applied."""

    limits: LimitsConfig
    network: str = "deny_if_possible"  # deny_if_possible | deny | allow
    extra_env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.network == "deny" and not network_namespace_available():
            raise CommandRefusedError(
                "isolation.network=deny but user network namespaces (unshare -rn) are unavailable;"
                " use deny_if_possible, or run inside a container without network"
            )

    @property
    def network_isolated(self) -> bool:
        return self.network != "allow" and network_namespace_available()

    def _env(self, scratch: Path) -> dict[str, str]:
        path_dirs = [str(Path(sys.executable).parent), "/usr/local/bin", "/usr/bin", "/bin"]
        env = {
            "PATH": os.pathsep.join(path_dirs),
            "HOME": str(scratch),
            "TMPDIR": str(scratch),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTEST_ADDOPTS": "-p no:cacheprovider",
            "SOURCE_DATE_EPOCH": "0",
        }
        env.update(self.extra_env)
        return env

    def _preexec(self, memory_mb: int | None) -> Callable[[], None]:
        limits = self.limits

        def apply() -> None:  # runs in the child between fork and exec
            cpu = limits.cpu_seconds
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
            fsize = limits.max_written_file_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
            if memory_mb is not None:
                mem = memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        return apply

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        root: Path,
        scratch: Path,
        allowed: set[str],
        timeout_s: float | None = None,
        cancel: CancelToken | None = None,
        memory_mb: int | None = -1,
        stdin_text: str | None = None,
        check_args: bool = True,
    ) -> CommandResult:
        """Run ``argv`` (``argv[0]`` is a logical, allowlisted name).

        ``memory_mb=-1`` uses the configured limit; ``None`` disables the address-space limit
        (needed by JVM-based runtimes). Raises ``CommandRefusedError`` on policy violations;
        execution failures are reported in the returned ``CommandResult``.
        """
        if not argv:
            raise CommandRefusedError("empty command")
        name = argv[0]
        if name not in allowed:
            raise CommandRefusedError(
                f"command {name!r} is not allowed here; allowed: {', '.join(sorted(allowed))}"
            )
        if check_args:
            for arg in argv[1:]:
                _check_argument(arg, root, cwd)
        full = [*resolve_executable(name), *argv[1:]]
        if self.network_isolated:
            unshare = shutil.which("unshare")
            if unshare:
                full = [unshare, "-rn", "--", *full]
        timeout = min(timeout_s or self.limits.command_timeout_s, self.limits.command_timeout_s)
        mem = self.limits.memory_mb if memory_mb == -1 else memory_mb
        scratch.mkdir(parents=True, exist_ok=True)
        return self._execute(list(argv), full, cwd, scratch, timeout, cancel, mem, stdin_text)

    def run_trusted(
        self,
        full_argv: Sequence[str],
        *,
        cwd: Path,
        scratch: Path,
        timeout_s: float,
        cancel: CancelToken | None = None,
        stdin_text: str | None = None,
        env_passthrough: Sequence[str] = (),
        allow_network: bool = False,
    ) -> CommandResult:
        """Run an operator-configured command (external agent CLI) without the allowlist.

        Only used for harness commands that come from the user's own config and were enabled
        explicitly. Environment scrubbing, timeout, rlimits (minus address space) and output caps
        still apply; ``env_passthrough`` names variables to forward (e.g. an API key).
        """
        exe = shutil.which(full_argv[0]) if full_argv else None
        if exe is None:
            raise CommandRefusedError(f"harness executable {full_argv[:1]} not found")
        full = [exe, *full_argv[1:]]
        if not allow_network and self.network_isolated:
            unshare = shutil.which("unshare")
            if unshare:
                full = [unshare, "-rn", "--", *full]
        extra = {k: os.environ[k] for k in env_passthrough if k in os.environ}
        scratch.mkdir(parents=True, exist_ok=True)
        runner = CommandRunner(self.limits, "allow", {**self.extra_env, **extra})
        return runner._execute(
            list(full_argv), full, cwd, scratch, timeout_s, cancel, None, stdin_text
        )

    def _execute(
        self,
        display_argv: list[str],
        full: list[str],
        cwd: Path,
        scratch: Path,
        timeout: float,
        cancel: CancelToken | None,
        memory_mb: int | None,
        stdin_text: str | None,
    ) -> CommandResult:
        start = time.monotonic()
        timed_out = cancelled = False
        with (
            tempfile.TemporaryFile(dir=scratch) as out_f,
            tempfile.TemporaryFile(dir=scratch) as err_f,
        ):
            try:
                proc = subprocess.Popen(
                    full,
                    cwd=cwd,
                    env=self._env(scratch),
                    stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                    stdout=out_f,
                    stderr=err_f,
                    start_new_session=True,
                    preexec_fn=self._preexec(memory_mb),  # noqa: PLW1509 - no threads touched
                )
            except OSError as exc:
                return CommandResult(display_argv, None, "", f"failed to start: {exc}", 0.0)
            if stdin_text is not None and proc.stdin is not None:
                try:
                    proc.stdin.write(stdin_text.encode())
                    proc.stdin.close()
                except OSError:
                    pass
            while proc.poll() is None:
                if time.monotonic() - start > timeout:
                    timed_out = True
                    break
                if cancel is not None and cancel.cancelled:
                    cancelled = True
                    break
                time.sleep(_POLL_S)
            if timed_out or cancelled:
                _kill_group(proc)
            exit_code = proc.wait()
            stdout, t1 = _read_capped(out_f, self.limits.max_output_bytes)
            stderr, t2 = _read_capped(err_f, self.limits.max_output_bytes)
        return CommandResult(
            argv=display_argv,
            exit_code=None if (timed_out or cancelled) else exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_s=time.monotonic() - start,
            timed_out=timed_out,
            cancelled=cancelled,
            truncated=t1 or t2,
        )


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def _read_capped(handle: IO[bytes], cap: int) -> tuple[str, bool]:
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(0)
    if size <= cap:
        return handle.read().decode("utf-8", errors="replace"), False
    head = handle.read(cap // 2)
    handle.seek(size - cap // 2)
    tail = handle.read()
    text = head.decode("utf-8", "replace") + "\n...[output truncated]...\n"
    return text + tail.decode("utf-8", "replace"), True
