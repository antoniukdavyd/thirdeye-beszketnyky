#!/usr/bin/env python3
"""Entrypoint: python run_assist.py"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
from pathlib import Path

from assist import crashlog
from assist.debuglog import log

LOCK_PATH = Path(__file__).resolve().parent / ".nekit.lock"


def single_instance_lock():
    """Refuse to start twice: Record3D streams to one client only, so a second
    copy silently steals the camera and the first one freezes."""
    fh = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.seek(0)
        other = fh.read().strip() or "?"
        log("app", "another instance is already running — exiting", other_pid=other)
        print(f"nekit is already running (pid {other}). Close it first (q in its window).")
        sys.exit(1)
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    return fh  # keep open: the lock lives as long as this file handle


def main():
    # Before anything heavy is imported, so even an import-time crash lands in
    # logs/nekit-*.log.
    crashlog.install(sys.argv[1:])
    _lock = single_instance_lock()  # noqa: F841
    from assist.app import AssistApp

    parser = argparse.ArgumentParser(description="nekit blind-assist MVP (Mac + Record3D)")
    parser.add_argument("--device", type=int, default=0, help="Record3D device index")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO-World")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice session")
    args = parser.parse_args()

    app = AssistApp(enable_yolo=not args.no_yolo, enable_voice=not args.no_voice)
    app.connect(dev_idx=args.device)
    app.run()


if __name__ == "__main__":
    main()
