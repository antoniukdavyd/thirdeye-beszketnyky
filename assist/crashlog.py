"""Crash forensics: session log file, uncaught-exception hooks, stall watchdog.

Everything printed to the terminal ([nekit ...] lines, library prints,
tracebacks) is also written to logs/nekit-<timestamp>.log, line-buffered, so
the file survives a crash. On top of that:

  * uncaught exceptions in any thread (and "unraisable" ones) are logged with
    the thread name and full traceback;
  * a native crash (SIGSEGV/SIGABRT/...) dumps every thread's Python stack
    into the same file via faulthandler;
  * SIGTERM/SIGHUP are logged before shutdown, and `kill -USR1 <pid>` dumps
    all stacks on demand without stopping the app;
  * a watchdog logs when the render loop or the detector stops making
    progress, with every thread's stack at that moment, plus a periodic
    health line (RSS, CPU, threads, thermal state).
"""

from __future__ import annotations

import atexit
import faulthandler
import io
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, TextIO

from .debuglog import log

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
KEEP_LOGS = 30
STALL_SEC = 1.5
HEALTH_SEC = 10.0
THERMAL_SEC = 30.0
STACK_DEPTH = 14

_log_file: Optional[TextIO] = None
_started_at = time.monotonic()


class _Tee(io.TextIOBase):
    """Write to the original stream and the session log file."""

    def __init__(self, stream: TextIO, file: TextIO, lock: threading.Lock) -> None:
        self._stream = stream
        self._file = file
        self._lock = lock

    def write(self, s: str) -> int:
        with self._lock:
            try:
                self._stream.write(s)
            except Exception:
                pass
            try:
                self._file.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:
        for target in (self._stream, self._file):
            try:
                target.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self):  # type: ignore[override]
        return getattr(self._stream, "encoding", "utf-8")


def log_file() -> Optional[TextIO]:
    return _log_file


def log_path() -> Optional[str]:
    return getattr(_log_file, "name", None)


def _prune_old_logs() -> None:
    logs = sorted(LOG_DIR.glob("nekit-*.log"))
    for old in logs[:-KEEP_LOGS]:
        try:
            old.unlink()
        except OSError:
            pass


def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=LOG_DIR.parent,
            capture_output=True,
            text=True,
            timeout=2,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=LOG_DIR.parent,
            capture_output=True,
            text=True,
            timeout=2,
        )
        rev = out.stdout.strip() or "?"
        return rev + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "?"


def _log_exception(where: str, exc_type, exc, tb) -> None:
    text = "".join(traceback.format_exception(exc_type, exc, tb)).rstrip()
    log("crash", f"UNCAUGHT {exc_type.__name__} in {where}: {exc}")
    print(text, flush=True)


def _sys_excepthook(exc_type, exc, tb) -> None:
    if not issubclass(exc_type, KeyboardInterrupt):
        _log_exception("main thread", exc_type, exc, tb)
    sys.__excepthook__(exc_type, exc, tb)


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread is not None else "?"
    _log_exception(f"thread {name!r}", args.exc_type, args.exc_value, args.exc_traceback)


def _unraisable_hook(unraisable) -> None:
    where = f"unraisable ({unraisable.err_msg or 'error'}: {unraisable.object!r})"
    _log_exception(where[:200], unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback)


def _on_signal(signum, _frame) -> None:
    name = signal.Signals(signum).name
    log("crash", f"received {name} — shutting down")
    # Unwind through the main loop's finally so devices are released.
    raise KeyboardInterrupt(name)


def _at_exit() -> None:
    log("crash", "process exit", uptime_s=round(time.monotonic() - _started_at, 1))


def format_thread_stacks(depth: int = STACK_DEPTH) -> str:
    """Every live thread's current Python stack, labelled with thread names."""
    names = {t.ident: t.name for t in threading.enumerate()}
    parts = []
    for ident, frame in sys._current_frames().items():
        stack = traceback.format_stack(frame)[-depth:]
        parts.append(f"--- thread {names.get(ident, ident)!r} ---\n" + "".join(stack).rstrip())
    return "\n".join(parts)


def install(args: Optional[list] = None) -> Optional[str]:
    """Start the session log and crash hooks. Safe to call once at startup."""
    global _log_file
    if _log_file is not None:
        return log_path()
    if (os.getenv("NEKIT_LOG_FILE") or "1").strip().lower() in ("0", "false", "no"):
        return None
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _prune_old_logs()
        path = LOG_DIR / f"nekit-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        _log_file = open(path, "a", buffering=1, encoding="utf-8")
    except OSError as exc:
        print(f"[nekit] could not open log file: {exc}", flush=True)
        return None

    lock = threading.Lock()
    sys.stdout = _Tee(sys.__stdout__, _log_file, lock)
    sys.stderr = _Tee(sys.__stderr__, _log_file, lock)

    # Native crashes: faulthandler writes straight to the fd, no Python needed.
    faulthandler.enable(file=_log_file, all_threads=True)
    try:
        faulthandler.register(signal.SIGUSR1, file=_log_file, all_threads=True)
    except (AttributeError, ValueError):
        pass

    sys.excepthook = _sys_excepthook
    threading.excepthook = _thread_excepthook
    sys.unraisablehook = _unraisable_hook
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass
    atexit.register(_at_exit)

    log(
        "crash",
        "session start",
        pid=os.getpid(),
        python=platform.python_version(),
        mac=platform.mac_ver()[0] or platform.platform(),
        rev=_git_rev(),
        args=" ".join(args or sys.argv[1:]) or "-",
        log=str(path),
    )
    return str(path)


class Watchdog:
    """Detects stalled loops and logs periodic process health.

    Loops call beat(name) every iteration. A name that has beaten at least
    once and then goes quiet for longer than its limit is logged as stalled
    (once per stall, with all thread stacks), and again when it recovers.
    """

    def __init__(
        self,
        stall_sec: float = STALL_SEC,
        health_sec: float = HEALTH_SEC,
        thermal_sec: float = THERMAL_SEC,
    ) -> None:
        self.stall_sec = stall_sec
        self.health_sec = health_sec
        self.thermal_sec = thermal_sec
        self._beats: Dict[str, float] = {}
        self._limits: Dict[str, float] = {}
        self._stalled: Dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = None
        self._thermal = ""
        self.stall_count = 0

    def beat(self, name: str, limit_sec: Optional[float] = None) -> None:
        if limit_sec is not None:
            self._limits[name] = limit_sec
        self._beats[name] = time.monotonic()

    def forget(self, name: str) -> None:
        """Stop watching `name` (its loop ended on purpose)."""
        self._beats.pop(name, None)
        self._stalled.pop(name, None)

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            import psutil

            self._proc = psutil.Process()
            self._proc.cpu_percent(None)
        except Exception:
            self._proc = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def check(self, now: Optional[float] = None) -> None:
        """One watchdog pass; separate from the thread so it is testable."""
        now = time.monotonic() if now is None else now
        for name, last in list(self._beats.items()):
            age = now - last
            limit = self._limits.get(name, self.stall_sec)
            if name in self._stalled:
                if age < limit:
                    stalled_for = now - self._stalled.pop(name)
                    log("watchdog", f"{name} resumed", stalled_s=round(stalled_for, 2))
            elif age >= limit:
                self._stalled[name] = last
                self.stall_count += 1
                log("watchdog", f"{name} STALLED — no progress", age_s=round(age, 2))
                print(format_thread_stacks(), flush=True)

    def health(self) -> dict:
        out: dict = {"uptime_s": round(time.monotonic() - _started_at)}
        if self._proc is not None:
            try:
                out["rss_mb"] = round(self._proc.memory_info().rss / 1e6)
                out["cpu_pct"] = round(self._proc.cpu_percent(None))
                out["threads"] = self._proc.num_threads()
            except Exception:
                pass
        if self._thermal:
            out["thermal"] = self._thermal
        if self._stalled:
            out["stalled"] = ",".join(sorted(self._stalled))
        return out

    def _read_thermal(self) -> None:
        try:
            res = subprocess.run(
                ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=2
            )
        except Exception:
            return
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        # Throttling shows as e.g. "CPU_Speed_Limit = 70"; otherwise "No ... recorded".
        limits = [ln.replace(" ", "") for ln in lines if "=" in ln]
        state = ";".join(limits) if limits else "ok"
        if state != self._thermal:
            if self._thermal:
                log("watchdog", "thermal state changed", old=self._thermal, new=state)
            self._thermal = state

    def _run(self) -> None:
        next_health = time.monotonic() + self.health_sec
        next_thermal = time.monotonic()
        while not self._stop.wait(0.25):
            try:
                now = time.monotonic()
                self.check(now)
                if now >= next_thermal:
                    self._read_thermal()
                    next_thermal = now + self.thermal_sec
                if now >= next_health:
                    log("health", "process", **self.health())
                    next_health = now + self.health_sec
            except Exception as exc:  # the watchdog must outlive what it watches
                log("watchdog", f"watchdog error: {type(exc).__name__}: {exc}")
