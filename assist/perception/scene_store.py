"""Thread-safe latest RGB + SENSOR_JSON for agent tools."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .detect import DetectedObject
from .scene import compact_scene_for_llm


@dataclass
class SceneSnapshot:
    rgb_bgr: Optional[np.ndarray]
    scene: dict
    objects: List[DetectedObject]
    updated_at: float  # time.monotonic()

    @property
    def age_ms(self) -> float:
        return max(0.0, (time.monotonic() - self.updated_at) * 1000.0)

    @property
    def has_frame(self) -> bool:
        return self.rgb_bgr is not None and bool(self.scene)


class SceneStore:
    """Published by the perception loop; read by agent tools."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rgb: Optional[np.ndarray] = None
        self._scene: dict = {}
        self._objects: List[DetectedObject] = []
        self._updated_at: float = 0.0

    def publish(
        self,
        rgb_bgr: np.ndarray,
        scene: dict,
        objects: Optional[List[DetectedObject]] = None,
    ) -> None:
        with self._lock:
            self._rgb = rgb_bgr
            self._scene = dict(scene) if scene else {}
            self._objects = list(objects or [])
            self._updated_at = time.monotonic()

    def snapshot(self, copy_rgb: bool = True) -> SceneSnapshot:
        with self._lock:
            rgb = None
            if self._rgb is not None:
                rgb = self._rgb.copy() if copy_rgb else self._rgb
            return SceneSnapshot(
                rgb_bgr=rgb,
                scene=dict(self._scene),
                objects=list(self._objects),
                updated_at=self._updated_at or time.monotonic(),
            )

    def compact(self) -> dict:
        snap = self.snapshot(copy_rgb=False)
        out = compact_scene_for_llm(snap.scene)
        objs = out.get("objects") or []
        out["objects"] = objs[:8]
        out["age_ms"] = round(snap.age_ms, 1)
        return out
