"""Left / center / right clearance — waist-up ROI (above cane height)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ..capture import MIN_CONF
from ..debuglog import log

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

# Beep mixer: fallback rate when the device rate cannot be queried, and the
# cap on queued tones so a burst of alerts cannot pile up into a drone.
TONE_FALLBACK_SR = 22050
TONE_MAX_QUEUED = 3
TONE_FADE_SEC = 0.006


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


class ToneMixer:
    """One long-lived output stream that beeps are pushed into.

    ``sounddevice.play()`` closes the previous CoreAudio stream and opens a
    brand new one on *every* call (see ``_CallbackContext.start_stream``). That
    costs ~115 ms on macOS, and proximity beeps fire up to ~8x/second straight
    from the render loop — which stalled the video loop to a few FPS, starved
    the Deepgram mic/playback callbacks, and in turn stalled the asyncio loop
    that has to answer the agent websocket's keepalive pings.

    So: open the device once, pre-render each tone once, and let the audio
    callback drain a queue. Pushing a beep is then a deque append — no device
    calls, no allocation, no blocking, from any thread.
    """

    def __init__(self, sd_module) -> None:
        self._sd = sd_module
        self._stream = None
        self._failed = False
        self.samplerate = float(TONE_FALLBACK_SR)
        # deque append/popleft are atomic under the GIL, so the realtime audio
        # callback needs no lock — and therefore can never be blocked by (or
        # block) the render loop.
        self._pending: Deque[np.ndarray] = deque()
        self._head: Optional[np.ndarray] = None
        self._cache: Dict[Tuple[float, float, float], np.ndarray] = {}

    def _device_samplerate(self) -> float:
        try:
            info = self._sd.query_devices(kind="output")
            rate = float(info["default_samplerate"])
            if rate > 0:
                return rate
        except Exception:
            pass
        return float(TONE_FALLBACK_SR)

    def _render(self, freq: float, duration: float, volume: float) -> np.ndarray:
        key = (freq, duration, volume)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        sr = self.samplerate
        n = max(1, int(sr * duration))
        t = np.arange(n, dtype=np.float32) / np.float32(sr)
        wave = (volume * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        # Raised-cosine edges: a hard start/stop on a sine clicks, and clicks
        # read as "something broke" to a user navigating by sound alone.
        fade = min(int(sr * TONE_FADE_SEC), n // 2)
        if fade > 0:
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade, dtype=np.float32)))
            wave[:fade] *= ramp
            wave[-fade:] *= ramp[::-1]
        self._cache[key] = wave
        return wave

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        out = outdata.reshape(-1)
        filled = 0
        while filled < frames:
            head = self._head
            if head is None or head.size == 0:
                try:
                    self._head = self._pending.popleft()
                except IndexError:
                    self._head = None
                    break
                continue
            n = min(frames - filled, head.size)
            out[filled : filled + n] = head[:n]
            self._head = head[n:]
            filled += n
        if filled < frames:
            out[filled:] = 0.0

    def _ensure_stream(self) -> bool:
        if self._stream is not None:
            return True
        if self._failed:
            return False
        try:
            self.samplerate = self._device_samplerate()
            stream = self._sd.OutputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            stream.start()
            self._stream = stream
            log("beep", "tone stream open", sr=self.samplerate)
            return True
        except Exception as exc:
            self._failed = True
            log("beep", f"tone stream unavailable: {type(exc).__name__}: {exc}")
            return False

    def play(self, freq: float, duration: float, volume: float) -> bool:
        if not self._ensure_stream():
            return False
        # Drop the beep rather than queue a drone: alerts are a *rate* signal,
        # and a backlog would smear the rate the user is listening to.
        if len(self._pending) >= TONE_MAX_QUEUED:
            return True
        try:
            self._pending.append(self._render(freq, duration, volume))
        except Exception as exc:
            log("beep", f"tone render failed: {type(exc).__name__}: {exc}")
            return False
        return True

    def close(self) -> None:
        stream, self._stream = self._stream, None
        self._pending.clear()
        self._head = None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


class BeepAlert:
    """Passive proximity + optional car side alerts (with hysteresis)."""

    def __init__(self, enable_audio: bool = True):
        self._last_beep = 0.0
        self._enabled = True
        self._mixer: Optional[ToneMixer] = None
        self._alert_center = False
        self._alert_left = False
        self._alert_right = False
        if enable_audio:
            try:
                import sounddevice as sd

                self._mixer = ToneMixer(sd)
            except ImportError:
                pass

    @property
    def _use_sounddevice(self) -> bool:
        return self._mixer is not None

    def close(self) -> None:
        if self._mixer is not None:
            self._mixer.close()

    def _tone(self, freq: float, duration: float = 0.06, volume: float = 0.25) -> None:
        if self._mixer is not None and self._mixer.play(freq, duration, volume):
            return
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
