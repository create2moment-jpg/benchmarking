"""Sandboxed command runner used for both agent tool calls and evaluator commands.

Safety properties (defence in depth - see docs/security.md for the threat model):

* structured argv only (never ``shell=True``); executable must be on an allowlist and is
  resolved to an absolute path by the runner, not by the caller;
* arguments containing ``..`` path segments or absolute paths outside the workspace are refused;
* scrubbed environment: no host secrets, no proxy variables; ``HOME``/``TMPDIR`` point to a
  per-attempt scratch directory;
* wall-clock timeout (process group killed), CPU / address-space / file-size rlimits;
* stdout/stderr captured to files and truncated to ``max_output_bytes``;
* optional network denial: commands run in a fresh Linux network namespace that has only a
  loopback interface (created in-process via ``os.unshare``; falls back to ``unshare -rn``).

These controls bound accidents and casual misuse. They are not a substitute for a container or
VM when running untrusted models against sensitive hosts.
"""

from __future__ import annotations

import ctypes
import fcntl
import functools
import importlib.util
import os
import resource
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO, Any

from engineering_llm_bench.config.models import LimitsConfig
from engineering_llm_bench.control import CancelToken

_POLL_S = 0.02
#: Pseudo-command: when allowlisted, ``./path`` programs built inside the workspace may run.
LOCAL_BINARIES = "local-binaries"


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
    #: Compute profile of the process tree (from wait4 rusage); None when unavailable.
    cpu_s: float | None = None
    max_rss_mb: float | None = None

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


_SIOCGIFFLAGS = 0x8913
_SIOCSIFFLAGS = 0x8914
_IFF_UP = 0x1
_CLONE_NEWNET = 0x40000000
_CLONE_NEWUSER = 0x10000000
_CLONE_NEWNS = 0x00020000
_MS_RDONLY = 1
_MS_NOSUID = 2
_MS_NODEV = 4
_MS_NOEXEC = 8
_MS_REC = 0x4000
_MS_PRIVATE = 1 << 18


def _write_proc(path: str, text: str) -> None:
    fd = os.open(path, os.O_WRONLY)
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)


@dataclass(frozen=True)
class SandboxSpec:
    """What the child enters before exec: a private network (loopback only) and/or a private
    mount view in which ``masks`` (and every attempt directory under ``workspaces_root`` other
    than ``keep``) are covered by empty, read-only tmpfs mounts."""

    net: bool
    masks: tuple[str, ...] = ()
    workspaces_root: str | None = None
    keep: str | None = None

    @property
    def mounts(self) -> bool:
        return bool(self.masks or self.workspaces_root)


_libc: Any = None


def _load_libc() -> Any:
    global _libc  # noqa: PLW0603 - loaded once in the parent, used in forked children
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
    return _libc


def _mount_empty(target: str) -> None:
    """Cover ``target`` with an empty read-only tmpfs (in the current mount namespace)."""
    flags = _MS_NOSUID | _MS_NODEV | _MS_NOEXEC | _MS_RDONLY
    if _libc.mount(b"elb-hidden", target.encode(), b"tmpfs", flags, b"size=4k,mode=0555") != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"cannot hide {target}: {os.strerror(err)}")


def _enter_sandbox(spec: SandboxSpec) -> None:
    """Child-side (after fork, before exec). Network: fresh namespace with only loopback (brought
    up so local IPC such as Spark's py4j still works). Filesystem: private mount namespace in
    which other models' outputs, results, hidden tests/reference solutions and other attempts'
    workspaces are replaced by empty directories."""
    unshare = getattr(os, "unshare")  # noqa: B009 - Python >= 3.12 only; guarded by caller
    flags = (_CLONE_NEWNET if spec.net else 0) | (_CLONE_NEWNS if spec.mounts else 0)
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        unshare(flags)
    else:
        unshare(_CLONE_NEWUSER | flags)
        _write_proc("/proc/self/setgroups", "deny")
        _write_proc("/proc/self/uid_map", f"{uid} {uid} 1")
        _write_proc("/proc/self/gid_map", f"{gid} {gid} 1")
    if spec.net:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            req = struct.pack("16sH", b"lo", 0)
            iff = struct.unpack("16sH", fcntl.ioctl(sock, _SIOCGIFFLAGS, req)[:18])[1]
            fcntl.ioctl(sock, _SIOCSIFFLAGS, struct.pack("16sH", b"lo", iff | _IFF_UP))
    if spec.mounts:
        # Stop our mounts from propagating back to the host.
        if _libc.mount(b"none", b"/", None, _MS_REC | _MS_PRIVATE, None) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"cannot make mounts private: {os.strerror(err)}")
        targets = list(spec.masks)
        if spec.workspaces_root and os.path.isdir(spec.workspaces_root):
            targets += [
                os.path.join(spec.workspaces_root, name)
                for name in os.listdir(spec.workspaces_root)
                if os.path.join(spec.workspaces_root, name) != spec.keep
            ]
        for target in targets:
            if os.path.isdir(target):
                _mount_empty(target)


def _enter_network_namespace() -> None:
    _enter_sandbox(SandboxSpec(net=True))


@functools.cache
def _inprocess_netns_available() -> bool:
    if sys.platform != "linux" or not hasattr(os, "unshare"):
        return False
    probe = (
        "import socket\n"
        "s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen()\n"
        "c = socket.create_connection(s.getsockname(), timeout=2)\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            timeout=10,
            check=False,
            preexec_fn=_enter_network_namespace,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


@functools.cache
def mount_hiding_available() -> bool:
    """True when commands can get a private filesystem view (Linux mount namespaces)."""
    if sys.platform != "linux" or not hasattr(os, "unshare"):
        return False
    _load_libc()
    with tempfile.TemporaryDirectory() as tmp:
        secret = Path(tmp) / "hidden"
        secret.mkdir()
        (secret / "x").write_text("visible")
        probe = f"import os,sys; sys.exit(0 if not os.listdir({str(secret)!r}) else 3)"
        spec = SandboxSpec(net=False, masks=(str(secret),))
        try:
            proc = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                timeout=10,
                check=False,
                preexec_fn=lambda: _enter_sandbox(spec),
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0


@functools.cache
def _unshare_binary_available() -> bool:
    exe = shutil.which("unshare")
    if exe is None or sys.platform != "linux":
        return False
    try:
        proc = subprocess.run([exe, "-rn", "true"], capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def network_namespace_available() -> bool:
    """True when commands can be run without external network on this host (Linux)."""
    return _inprocess_netns_available() or _unshare_binary_available()


def filesystem_isolation_method() -> str:
    return "mount namespace (other models' folders hidden)" if mount_hiding_available() else "none"


def isolation_method() -> str:
    """Human-readable description of the network isolation in effect."""
    if _inprocess_netns_available():
        return "network namespace (loopback only)"
    if _unshare_binary_available():
        return "unshare -rn (no network, loopback down)"
    return "none"


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
    #: Directories made invisible to every command (other models' outputs, results, tasks...).
    hidden_paths: tuple[Path, ...] = ()
    #: Root holding all attempt workspaces; sibling attempts are hidden from each command.
    workspaces_root: Path | None = None
    filesystem: str = "hide_if_possible"  # hide_if_possible | hide | off

    def __post_init__(self) -> None:
        if self.filesystem == "hide" and not mount_hiding_available():
            raise CommandRefusedError(
                "isolation.filesystem=hide but Linux mount namespaces are unavailable here "
                "(needs Linux and Python >= 3.12); use hide_if_possible or run in a container"
            )
        if self.network == "deny" and not network_namespace_available():
            raise CommandRefusedError(
                "isolation.network=deny but Linux network namespaces are unavailable here;"
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

    def _masks(self, cwd: Path) -> tuple[tuple[str, ...], str | None, str | None]:
        """Paths to hide for a command running in ``cwd`` (never the cwd, its parents or the
        interpreter), plus the workspaces root and the attempt directory to keep visible."""
        if self.filesystem == "off" or not mount_hiding_available():
            return (), None, None
        cwd_r = cwd.resolve()
        keep_paths = [cwd_r, Path(sys.executable).resolve()]
        masks = []
        for path in self.hidden_paths:
            p = path.resolve()
            if p.is_dir() and not any(k.is_relative_to(p) for k in keep_paths):
                masks.append(str(p))
        root = keep = None
        if self.workspaces_root is not None and cwd_r.is_relative_to(
            self.workspaces_root.resolve()
        ):
            root_r = self.workspaces_root.resolve()
            root = str(root_r)
            keep = str(root_r / cwd_r.relative_to(root_r).parts[0])
        return tuple(masks), root, keep

    def _isolate(
        self, full: list[str], cwd: Path, allow_network: bool = False
    ) -> tuple[list[str], SandboxSpec | None]:
        """Return (argv, sandbox to enter in the child) according to the isolation policy."""
        masks, root, keep = self._masks(cwd)
        want_net = not (allow_network or self.network == "allow")
        if want_net and _inprocess_netns_available():
            return full, SandboxSpec(True, masks, root, keep)
        spec = SandboxSpec(False, masks, root, keep) if (masks or root) else None
        if want_net:
            unshare = shutil.which("unshare")
            if unshare and _unshare_binary_available() and spec is None:
                return [unshare, "-rn", "--", *full], None
        return full, spec

    def _preexec(self, memory_mb: int | None, sandbox: SandboxSpec | None) -> Callable[[], None]:
        limits = self.limits

        def apply() -> None:  # runs in the child between fork and exec
            if sandbox is not None:
                _enter_sandbox(sandbox)
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
        if name.startswith("./") and LOCAL_BINARIES in allowed:
            return self._run_local_binary(
                list(argv), cwd, root, scratch, timeout_s, cancel, memory_mb, stdin_text
            )
        if name not in allowed:
            raise CommandRefusedError(
                f"command {name!r} is not allowed here; allowed: {', '.join(sorted(allowed))}"
            )
        if check_args:
            for arg in argv[1:]:
                _check_argument(arg, root, cwd)
        full = [*resolve_executable(name), *argv[1:]]
        full, netns = self._isolate(full, cwd)
        timeout = min(timeout_s or self.limits.command_timeout_s, self.limits.command_timeout_s)
        mem = self.limits.memory_mb if memory_mb == -1 else memory_mb
        scratch.mkdir(parents=True, exist_ok=True)
        return self._execute(
            list(argv), full, cwd, scratch, timeout, cancel, mem, stdin_text, netns
        )

    def _run_local_binary(
        self,
        argv: list[str],
        cwd: Path,
        root: Path,
        scratch: Path,
        timeout_s: float | None,
        cancel: CancelToken | None,
        memory_mb: int | None,
        stdin_text: str | None,
    ) -> CommandResult:
        """Run a program built inside the workspace (e.g. a compiled C++ test)."""
        target = (cwd / argv[0]).resolve()
        if not target.is_relative_to(root.resolve()) or ".." in PurePosixPath(argv[0]).parts:
            raise CommandRefusedError(f"{argv[0]!r} is outside the workspace")
        if not target.is_file() or not os.access(target, os.X_OK):
            raise CommandRefusedError(f"{argv[0]!r} is not an executable file")
        for arg in argv[1:]:
            _check_argument(arg, root, cwd)
        full = [str(target), *argv[1:]]
        full, netns = self._isolate(full, cwd)
        timeout = min(timeout_s or self.limits.command_timeout_s, self.limits.command_timeout_s)
        mem = self.limits.memory_mb if memory_mb == -1 else memory_mb
        scratch.mkdir(parents=True, exist_ok=True)
        return self._execute(argv, full, cwd, scratch, timeout, cancel, mem, stdin_text, netns)

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
        full, netns = self._isolate([exe, *full_argv[1:]], cwd, allow_network)
        extra = {k: os.environ[k] for k in env_passthrough if k in os.environ}
        scratch.mkdir(parents=True, exist_ok=True)
        runner = CommandRunner(self.limits, "allow", {**self.extra_env, **extra}, filesystem="off")
        return runner._execute(
            list(full_argv), full, cwd, scratch, timeout_s, cancel, None, stdin_text, netns
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
        netns: SandboxSpec | None = None,
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
                    preexec_fn=self._preexec(memory_mb, netns),  # noqa: PLW1509 - no threads touched
                )
            except OSError as exc:
                return CommandResult(display_argv, None, "", f"failed to start: {exc}", 0.0)
            if stdin_text is not None and proc.stdin is not None:
                try:
                    proc.stdin.write(stdin_text.encode())
                    proc.stdin.close()
                except OSError:
                    pass
            reaped: tuple[int, float | None, float | None] | None = None
            while reaped is None:
                reaped = _try_reap(proc)
                if reaped is not None:
                    break
                if time.monotonic() - start > timeout:
                    timed_out = True
                    break
                if cancel is not None and cancel.cancelled:
                    cancelled = True
                    break
                time.sleep(_POLL_S)
            if reaped is None:
                _kill_group(proc)
                reaped = _wait_with_rusage(proc)
            exit_code, cpu_s, max_rss_mb = reaped
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
            cpu_s=cpu_s,
            max_rss_mb=max_rss_mb,
        )


def _rusage_result(
    proc: subprocess.Popen[bytes], status: int, usage: Any
) -> tuple[int, float, float]:
    code = os.waitstatus_to_exitcode(status)
    proc.returncode = code
    rss_kb = usage.ru_maxrss / 1024 if sys.platform == "darwin" else usage.ru_maxrss
    return code, round(usage.ru_utime + usage.ru_stime, 3), round(rss_kb / 1024, 1)


def _try_reap(proc: subprocess.Popen[bytes]) -> tuple[int, float | None, float | None] | None:
    """Non-blocking reap. Uses ``wait4`` (captures CPU time and peak RSS of the process and the
    descendants it waited for) where available, ``Popen.poll`` elsewhere."""
    if not hasattr(os, "wait4"):
        code = proc.poll()
        return None if code is None else (code, None, None)
    try:
        pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
    except ChildProcessError:
        return proc.wait(), None, None
    return None if pid == 0 else _rusage_result(proc, status, usage)


def _wait_with_rusage(proc: subprocess.Popen[bytes]) -> tuple[int, float | None, float | None]:
    if not hasattr(os, "wait4"):
        return proc.wait(), None, None
    try:
        _, status, usage = os.wait4(proc.pid, 0)
    except ChildProcessError:
        return proc.wait(), None, None
    return _rusage_result(proc, status, usage)


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
