"""Uniform terminal logging — prefix every line with [nekit ...] for easy copy/paste."""

from __future__ import annotations

import sys
import time
from datetime import datetime


def log(area: str, message: str, **fields) -> None:
    """
    Print one structured line to stdout (always flushed).

    Example:
      [nekit 23:15:01.234] voice | state=listening busy=0 | session ON
    """
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    extra = ""
    if fields:
        parts = []
        for k, v in fields.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.3f}")
            elif isinstance(v, bool):
                parts.append(f"{k}={int(v)}")
            elif v is None:
                parts.append(f"{k}=None")
            else:
                s = str(v)
                # Always quote strings with spaces or non-ascii for easy copy
                if " " in s or not s.isascii() or "'" in s:
                    parts.append(f"{k}={s!r}")
                else:
                    parts.append(f"{k}={s}")
        extra = " ".join(parts) + " | "
    line = f"[nekit {ts}] {area} | {extra}{message}"
    print(line, flush=True)
    sys.stdout.flush()


def log_exc(area: str, message: str, exc: BaseException) -> None:
    log(area, f"{message}: {type(exc).__name__}: {exc}")
