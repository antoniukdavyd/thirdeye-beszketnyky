"""Record3D USB stream wrapper: RGB + depth + confidence + intrinsics."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Optional

import cv2
import numpy as np
from record3d import Record3DStream

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


class Record3DCapture:
    """Connects to Record3D and yields aligned frames."""

    def __init__(self):
        self.event = Event()
        self.session: Optional[Record3DStream] = None

    def on_new_frame(self):
        self.event.set()

    def on_stream_stopped(self):
        print("Stream stopped")

    def connect(self, dev_idx: int = 0) -> None:
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

    def wait_frame(self, timeout: Optional[float] = None) -> bool:
        ok = self.event.wait(timeout)
        return bool(ok)

    def clear_event(self) -> None:
        self.event.clear()

    def grab(self) -> Optional[FrameBundle]:
        if self.session is None:
            return None

        depth = self.session.get_depth_frame()
        rgb = self.session.get_rgb_frame()
        conf = self.session.get_confidence_frame()
        K = self.session.get_intrinsic_mat()
        device_type = int(self.session.get_device_type())

        if self.session.get_device_type() == DEVICE_TYPE__TRUEDEPTH:
            depth = cv2.flip(depth, 1)
            rgb = cv2.flip(rgb, 1)
            if conf is not None and conf.size > 0:
                conf = cv2.flip(conf, 1)

        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_a, conf_a = align_depth_to_rgb(depth, conf, rgb_bgr.shape)
        return FrameBundle(
            rgb_bgr=rgb_bgr,
            depth=depth_a,
            conf=conf_a,
            K=K,
            device_type=device_type,
        )


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
