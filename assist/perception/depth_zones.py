"""Left / center / right clearance — waist-up ROI (above cane height)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..capture import MIN_CONF

# Waist-up / torso band — white cane covers the floor; we cover above belt
V_TOP = 0.12
V_BOT = 0.55
ALERT_CENTER_M = 0.8
ALERT_SIDE_M = 0.6
EXIT_CENTER_M = ALERT_CENTER_M + 0.25
EXIT_SIDE_M = ALERT_SIDE_M + 0.2
CAR_SIDE_ALERT_M = 1.2
MIN_DEPTH = 0.08
MAX_DEPTH = 8.0


@dataclass
class ZoneClearance:
    left: Optional[float]
    center: Optional[float]
    right: Optional[float]

    def as_dict(self) -> dict:
        def fmt(v: Optional[float]):
            return round(v, 2) if v is not None else None

        return {
            "left": fmt(self.left),
            "center": fmt(self.center),
            "right": fmt(self.right),
        }

    def hint(self) -> str:
        parts = []
        c = self.center
        if c is not None and c < ALERT_CENTER_M:
            parts.append("center blocked")
        if self.left is not None and self.left < ALERT_SIDE_M:
            parts.append("left close")
        if self.right is not None and self.right < ALERT_SIDE_M:
            parts.append("right close")
        if not parts:
            if c is not None:
                parts.append("path clearer" if c >= 1.5 else "path ok")
            else:
                parts.append("no depth")
        return "; ".join(parts)


class ZoneFilter:
    """EMA smoother for L/C/R clearances to reduce beep flicker."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._prev: Optional[ZoneClearance] = None

    def _ema(self, old: Optional[float], new: Optional[float]) -> Optional[float]:
        if new is None:
            return old
        if old is None:
            return new
        return old * (1.0 - self.alpha) + new * self.alpha

    def update(self, zones: ZoneClearance) -> ZoneClearance:
        if self._prev is None:
            self._prev = zones
            return zones
        out = ZoneClearance(
            left=self._ema(self._prev.left, zones.left),
            center=self._ema(self._prev.center, zones.center),
            right=self._ema(self._prev.right, zones.right),
        )
        self._prev = out
        return out


def _zone_min_depth(
    depth: np.ndarray,
    conf: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    min_conf: int = MIN_CONF,
) -> Optional[float]:
    patch_d = depth[y0:y1, x0:x1]
    patch_c = conf[y0:y1, x0:x1]
    mask = (patch_d > MIN_DEPTH) & (patch_d < MAX_DEPTH) & (patch_c >= min_conf)
    if not np.any(mask):
        return None
    return float(np.percentile(patch_d[mask], 10))


def compute_zones(
    depth: np.ndarray, conf: np.ndarray, min_conf: int = MIN_CONF
) -> ZoneClearance:
    """Clearance in waist-up band (objects the cane often misses)."""
    h, w = depth.shape
    y0 = int(h * V_TOP)
    y1 = int(h * V_BOT)
    w3 = w // 3
    left = _zone_min_depth(depth, conf, 0, w3, y0, y1, min_conf)
    center = _zone_min_depth(depth, conf, w3, 2 * w3, y0, y1, min_conf)
    right = _zone_min_depth(depth, conf, 2 * w3, w, y0, y1, min_conf)
    return ZoneClearance(left=left, center=center, right=right)


def _hysteresis(
    value: Optional[float], active: bool, enter_m: float, exit_m: float
) -> bool:
    if value is None:
        return False
    if active:
        return value < exit_m
    return value < enter_m


class BeepAlert:
    """Passive proximity + optional car side alerts (with hysteresis)."""

    def __init__(self, enable_audio: bool = True):
        self._last_beep = 0.0
        self._enabled = True
        self._use_sounddevice = False
        self._sd = None
        self._alert_center = False
        self._alert_left = False
        self._alert_right = False
        if enable_audio:
            try:
                import sounddevice as sd

                self._sd = sd
                self._use_sounddevice = True
            except ImportError:
                pass

    def _tone(self, freq: float, duration: float = 0.06, volume: float = 0.25) -> None:
        if self._use_sounddevice and self._sd is not None:
            sr = 22050
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            wave = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            try:
                self._sd.play(wave, sr, blocking=False)
            except Exception:
                print("\a", end="", flush=True)
        else:
            print("\a", end="", flush=True)

    def update(
        self,
        zones: ZoneClearance,
        objects: Optional[List] = None,
    ) -> None:
        if not self._enabled:
            return

        now = time.monotonic()
        candidates = []

        self._alert_center = _hysteresis(
            zones.center, self._alert_center, ALERT_CENTER_M, EXIT_CENTER_M
        )
        self._alert_left = _hysteresis(
            zones.left, self._alert_left, ALERT_SIDE_M, EXIT_SIDE_M
        )
        self._alert_right = _hysteresis(
            zones.right, self._alert_right, ALERT_SIDE_M, EXIT_SIDE_M
        )

        if self._alert_center and zones.center is not None:
            interval = 0.12 + 0.5 * min(1.0, zones.center / ALERT_CENTER_M)
            candidates.append((interval, 880.0, "center"))
        if self._alert_left and zones.left is not None:
            interval = 0.18 + 0.55 * min(1.0, zones.left / ALERT_SIDE_M)
            candidates.append((interval, 520.0, "left"))
        if self._alert_right and zones.right is not None:
            interval = 0.18 + 0.55 * min(1.0, zones.right / ALERT_SIDE_M)
            candidates.append((interval, 660.0, "right"))

        if objects:
            for o in objects:
                label = getattr(o, "label", "") or ""
                lab = label.lower()
                if not any(k in lab for k in ("car", "bus", "truck")):
                    continue
                dist = float(getattr(o, "dist_m", 99))
                bearing = getattr(o, "bearing", "") or ""
                if dist < CAR_SIDE_ALERT_M and bearing in (
                    "left",
                    "right",
                    "center-left",
                    "center-right",
                ):
                    interval = 0.22 + 0.4 * (dist / CAR_SIDE_ALERT_M)
                    freq = 420.0 if "left" in bearing else 480.0
                    candidates.append((interval, freq, "car"))

        if not candidates:
            return

        candidates.sort(key=lambda x: x[0])
        interval, freq, kind = candidates[0]
        if now - self._last_beep >= interval:
            self._last_beep = now
            dur = 0.1 if kind == "car" else 0.06
            self._tone(freq, duration=dur)
