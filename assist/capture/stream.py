"""Record3D USB stream wrapper: RGB + depth + confidence + intrinsics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Event
from typing import Optional

import cv2
import numpy as np

from ..debuglog import log

DEVICE_TYPE__TRUEDEPTH = 0
DEVICE_TYPE__LIDAR = 1
MIN_CONF = 1


@dataclass
class FrameBundle:
    rgb_bgr: np.ndarray
    depth: np.ndarray
    conf: np.ndarray
    K: object
    device_type: int

    @property
    def device_name(self) -> str:
        return "truedepth" if self.device_type == DEVICE_TYPE__TRUEDEPTH else "lidar"


def align_depth_to_rgb(
    depth: np.ndarray, conf: Optional[np.ndarray], rgb_shape: tuple
) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb_shape[:2]
    depth_a = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    if conf is not None and conf.size > 0:
        conf_a = cv2.resize(conf, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        conf_a = np.full((h, w), 2, dtype=np.uint8)
    return depth_a, conf_a


def capture_backend() -> str:
    """R3D_BACKEND: "native" (default) or "record3d" (vendor library)."""
    return (os.getenv("R3D_BACKEND") or "native").strip().lower()


def _to_bundle(rgb_bgr, depth, conf, K, device_type: int) -> FrameBundle:
    if device_type == DEVICE_TYPE__TRUEDEPTH:
        depth = cv2.flip(depth, 1)
        rgb_bgr = cv2.flip(rgb_bgr, 1)
        if conf is not None and conf.size > 0:
            conf = cv2.flip(conf, 1)
    depth_a, conf_a = align_depth_to_rgb(depth, conf, rgb_bgr.shape)
    return FrameBundle(
        rgb_bgr=rgb_bgr,
        depth=depth_a,
        conf=conf_a,
        K=K,
        device_type=device_type,
    )


class Record3DCapture:
    """Connects to Record3D and yields the newest aligned frame.

    The default "native" backend (see native.py) receives frames on its own
    thread and decodes only the newest one, so a slow render loop drops frames
    instead of building a backlog. The vendor library ("record3d") decodes
    every frame on one thread and could not keep up with the phone's 60 fps at
    1440x1920 — the picture fell further behind every second.
    """

    def __init__(self, backend: Optional[str] = None):
        self.event = Event()
        self.backend = backend or capture_backend()
        # record3d is the iPhone USB SDK; only needed for the "record3d"
        # backend, and it does not build on every Python. Import lazily.
        self.session = None
        self._reader = None
        self._last_seq = 0
        self._last_bundle: Optional[FrameBundle] = None

    def on_new_frame(self):
        self.event.set()

    def on_stream_stopped(self):
        print("Stream stopped")
        log("frame", "Record3D stream stopped")

    def connect(self, dev_idx: int = 0) -> None:
        if self.backend == "record3d":
            self._connect_record3d(dev_idx)
            return
        from .native import LatestFrameReader, wait_for_device

        dev = wait_for_device(dev_idx)
        self._reader = LatestFrameReader(
            dev.device_id,
            on_new_frame=self.on_new_frame,
            on_stream_stopped=self.on_stream_stopped,
        )
        self._reader.start()
        print("Record3D reader started.")
        log(
            "frame",
            "Record3D reader started",
            backend="native",
            dev_idx=dev_idx,
            product_id=dev.product_id,
        )

    def _connect_record3d(self, dev_idx: int) -> None:
        from record3d import Record3DStream

        print("Searching for devices...")
        while True:
            devs = Record3DStream.get_connected_devices()
            print(f"{len(devs)} device(s) found")
            for i, dev in enumerate(devs):
                print(f"  [{i}] ID={dev.product_id}  UDID={dev.udid}")
            if len(devs) > dev_idx:
                break
            print(
                "No device yet. Enable USB Streaming in Record3D, press Record, wait..."
            )
            self.event.wait(2.0)
            self.event.clear()

        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devs[dev_idx])
        print("Connected to Record3D stream.")
        log(
            "frame",
            "Record3D connected",
            backend="record3d",
            dev_idx=dev_idx,
            product_id=devs[dev_idx].product_id,
        )

    def stop(self) -> None:
        """Best-effort teardown used by AssistApp.finally."""
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        self.session = None
        self.event.set()

    def take_stats(self) -> dict:
        """Receive counters since the last call (native backend only)."""
        return self._reader.take_stats() if self._reader is not None else {}

    def wait_frame(self, timeout: Optional[float] = None) -> bool:
        ok = self.event.wait(timeout)
        return bool(ok)

    def clear_event(self) -> None:
        self.event.clear()

    def grab(self) -> Optional[FrameBundle]:
        if self._reader is not None:
            return self._grab_native()
        if self.session is None:
            return None

        depth = self.session.get_depth_frame()
        rgb = self.session.get_rgb_frame()
        conf = self.session.get_confidence_frame()
        K = self.session.get_intrinsic_mat()
        device_type = int(self.session.get_device_type())
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return _to_bundle(rgb_bgr, depth, conf, K, device_type)

    def _grab_native(self) -> Optional[FrameBundle]:
        from .native import decode_frame

        raw = self._reader.latest()
        if raw is None:
            return None
        # A frame landing between clear_event() and latest() leaves the event
        # set for a frame we already hold; don't decode it twice.
        if raw.seq == self._last_seq and self._last_bundle is not None:
            return self._last_bundle
        decoded = decode_frame(raw.body)
        if decoded is None:
            log("frame", "Record3D frame decode failed", seq=raw.seq)
            return None
        bundle = _to_bundle(
            decoded.rgb_bgr,
            decoded.depth,
            decoded.conf,
            decoded.K,
            decoded.device_type,
        )
        self._last_seq = raw.seq
        self._last_bundle = bundle
        return bundle


def median_depth_in_box(
    depth: np.ndarray,
    conf: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    min_conf: int = MIN_CONF,
) -> tuple[Optional[float], int]:
    """Near-biased depth inside bbox; returns (meters, median conf).

    Uses 25th percentile so foreground object wins over far background
    that often fills the box edges / gaps.
    """
    h, w = depth.shape
    xa, xb = max(0, int(x1)), min(w, int(x2))
    ya, yb = max(0, int(y1)), min(h, int(y2))
    if xa >= xb or ya >= yb:
        return None, 0
    # Focus on central 50% of box to avoid background edges
    dx, dy = xb - xa, yb - ya
    mx0 = xa + dx // 4
    mx1 = xb - dx // 4
    my0 = ya + dy // 4
    my1 = yb - dy // 4
    if mx0 >= mx1 or my0 >= my1:
        mx0, mx1, my0, my1 = xa, xb, ya, yb
    patch_d = depth[my0:my1, mx0:mx1]
    patch_c = conf[my0:my1, mx0:mx1]
    mask = (patch_d > 0.05) & (patch_d < 20.0) & (patch_c >= min_conf)
    if not np.any(mask):
        return None, 0
    return float(np.percentile(patch_d[mask], 25)), int(np.median(patch_c[mask]))
