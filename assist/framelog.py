"""Debug logging for the capture → render loop.

Per-frame lines would flood the terminal at 30–60 fps, so this keeps running
counters and prints one summary every FRAME_LOG_SEC seconds. Discrete events
(first frame, resolution/device change, stalls, grab failures) are logged as
they happen regardless of the interval.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .debuglog import log

DEFAULT_INTERVAL_SEC = 2.0
# No new frame for this long counts as a stall (logged once per stall).
STALL_SEC = 1.0
STAGES = ("wait", "grab", "zones", "publish", "draw", "show")


def frame_log_interval() -> float:
    """FRAME_LOG_SEC from env; 0 or negative disables the periodic summary."""
    raw = (os.getenv("FRAME_LOG_SEC") or "").strip()
    if not raw:
        return DEFAULT_INTERVAL_SEC
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_INTERVAL_SEC


def depth_stats(depth: np.ndarray, conf: np.ndarray) -> Dict[str, object]:
    """Valid-pixel share and depth spread; only run on summary frames."""
    valid = np.isfinite(depth) & (depth > 0)
    n = int(depth.size) or 1
    out: Dict[str, object] = {"valid_pct": round(100.0 * int(valid.sum()) / n, 1)}
    if np.any(valid):
        d = depth[valid]
        out["d_min"] = float(d.min())
        out["d_med"] = float(np.median(d))
        out["d_max"] = float(d.max())
    else:
        out["d_min"] = out["d_med"] = out["d_max"] = None
    # Record3D confidence: 0 low, 1 medium, 2 high.
    out["conf_hi_pct"] = round(100.0 * int((conf >= 2).sum()) / n, 1)
    return out


class FrameLogger:
    """Collects loop timings and frame metadata; emits periodic summaries."""

    def __init__(
        self,
        interval_sec: Optional[float] = None,
        stream_stats: Optional[Callable[[], dict]] = None,
    ) -> None:
        self.interval = frame_log_interval() if interval_sec is None else interval_sec
        # Receive-side counters from the capture (frames the phone sent, frames
        # dropped as stale). phone_fps > cap_fps is fine; frames are skipped,
        # not queued.
        self._stream_stats = stream_stats
        self._window_start = time.monotonic()
        self._last_frame_at: Optional[float] = None
        self._stalled = False
        self._shape: Optional[Tuple[int, ...]] = None
        self._device: Optional[str] = None
        self._frames = 0
        self._renders = 0
        self._wait_timeouts = 0
        self._grab_none = 0
        self._grab_none_total = 0
        # YOLO passes finished in this window (written by the detector thread).
        self._detects = 0
        self._detect_ms = 0.0
        self._stage_ms: Dict[str, float] = {s: 0.0 for s in STAGES}
        self._stage_max_ms: Dict[str, float] = {s: 0.0 for s in STAGES}
        self.total_frames = 0

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def stage(self, name: str, ms: float) -> None:
        self._stage_ms[name] = self._stage_ms.get(name, 0.0) + ms
        if ms > self._stage_max_ms.get(name, 0.0):
            self._stage_max_ms[name] = ms

    def wait_timeout(self) -> None:
        self._wait_timeouts += 1
        now = time.monotonic()
        since = now - (self._last_frame_at or self._window_start)
        if not self._stalled and since >= STALL_SEC:
            self._stalled = True
            log(
                "frame",
                "no camera frames — stream stalled?",
                since_s=round(since, 2),
                total=self.total_frames,
            )

    def grab_failed(self) -> None:
        self._grab_none += 1
        self._grab_none_total += 1
        # First few, then every 100th: a persistent failure at 60 fps would
        # otherwise drown every other log line.
        n = self._grab_none_total
        if n <= 3 or n % 100 == 0:
            log("frame", "grab returned no frame", count=n, total=self.total_frames)

    def frame(self, frame) -> None:
        """Record a grabbed FrameBundle; logs first frame and format changes."""
        now = time.monotonic()
        if self._stalled:
            gap = now - (self._last_frame_at or self._window_start)
            log("frame", "frames resumed", gap_s=round(gap, 2))
            self._stalled = False
        self._last_frame_at = now
        self._frames += 1
        self.total_frames += 1

        shape = (
            tuple(frame.rgb_bgr.shape)
            + tuple(frame.depth.shape)
            + tuple(frame.conf.shape)
        )
        device = frame.device_name
        if shape != self._shape or device != self._device:
            h, w = frame.rgb_bgr.shape[:2]
            msg = "first frame" if self._shape is None else "frame format changed"
            log(
                "frame",
                msg,
                device=device,
                rgb=f"{w}x{h}",
                depth=f"{frame.depth.shape[1]}x{frame.depth.shape[0]}",
                depth_dtype=str(frame.depth.dtype),
                conf_dtype=str(frame.conf.dtype),
                K=_fmt_intrinsics(frame.K),
            )
            self._shape = shape
            self._device = device

    def rendered(self) -> None:
        self._renders += 1

    def detected(self, ms: float) -> None:
        """One finished YOLO pass and how long it took."""
        self._detects += 1
        self._detect_ms += ms

    def maybe_summary(self, frame=None, zones=None, objects=None) -> None:
        """Emit one summary line if the interval elapsed, then reset counters."""
        if not self.enabled:
            return
        now = time.monotonic()
        elapsed = now - self._window_start
        if elapsed < self.interval:
            return

        fields: Dict[str, object] = {}
        if self._stream_stats is not None:
            rx = self._stream_stats() or {}
            if "rx_frames" in rx:
                fields["phone_fps"] = round(rx["rx_frames"] / elapsed, 1)
                fields["rx_mbs"] = round(rx.get("rx_bytes", 0) / elapsed / 1e6, 1)
                fields["dropped"] = rx.get("dropped", 0)
        fields.update({
            "cap_fps": round(self._frames / elapsed, 1),
            "draw_fps": round(self._renders / elapsed, 1),
            "det_fps": round(self._detects / elapsed, 1),
            "frames": self.total_frames,
        })
        if self._detects:
            fields["det_ms"] = round(self._detect_ms / self._detects, 1)
        if self._wait_timeouts:
            fields["timeouts"] = self._wait_timeouts
        if self._grab_none:
            fields["grab_none"] = self._grab_none
        n = max(1, self._frames)
        for s in STAGES:
            fields[f"{s}_ms"] = round(self._stage_ms[s] / n, 1)
        slowest = max(STAGES, key=lambda s: self._stage_max_ms[s])
        fields["max_stage"] = f"{slowest}:{self._stage_max_ms[slowest]:.0f}ms"
        if frame is not None:
            fields.update(depth_stats(frame.depth, frame.conf))
        if zones is not None:
            fields["zones"] = "/".join(
                "-" if v is None else f"{v:.2f}"
                for v in (zones.left, zones.center, zones.right)
            )
        if objects is not None:
            fields["objs"] = len(objects)
        log("frame", "loop stats", **fields)

        self._window_start = now
        self._frames = 0
        self._renders = 0
        self._detects = 0
        self._detect_ms = 0.0
        self._wait_timeouts = 0
        self._grab_none = 0
        self._stage_ms = {s: 0.0 for s in STAGES}
        self._stage_max_ms = {s: 0.0 for s in STAGES}


def _fmt_intrinsics(K) -> str:
    """fx,fy,cx,cy from Record3D intrinsics (struct or 3x3), best effort."""
    try:
        if all(hasattr(K, a) for a in ("fx", "fy", "tx", "ty")):
            return f"{K.fx:.0f},{K.fy:.0f},{K.tx:.0f},{K.ty:.0f}"
        m = np.asarray(K, dtype=float).reshape(3, 3)
        return f"{m[0, 0]:.0f},{m[1, 1]:.0f},{m[0, 2]:.0f},{m[1, 2]:.0f}"
    except Exception:
        return "?"
