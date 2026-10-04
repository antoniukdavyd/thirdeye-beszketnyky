"""On-demand focused detection for find_object.

The live detector runs a small model over 17 classes; doors and people it
scores low are simply missing from its list. The finder runs only when asked,
with a bigger model, higher resolution, a low score floor and synonyms per
target (YOLO-World scores each prompt, so a doorway or glass door can match
where "door" alone would not).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..capture import median_depth_in_box
from ..debuglog import log, log_exc
from .detect import YOLO_LOCK, bearing_from_cx, iter_boxes, yolo_device

DEFAULT_FIND_MODEL = "yolov8l-worldv2.pt"
DEFAULT_FIND_IMGSZ = 1280
FIND_MIN_SCORE = 0.10
# Hits at or above this are spoken as certain; below as "possibly".
STRONG_SCORE = 0.35

FIND_SYNONYMS = {
    "door": ("door", "doorway", "glass door", "entrance"),
    "person": ("person", "pedestrian", "child"),
    "stairs": ("stairs", "staircase", "steps"),
}

# One fixed vocabulary, set once: any set_classes call makes the next MPS
# predict ~1 s slower (measured on an M2: 260-380 ms per find with a fixed
# vocabulary, 1.2-3.2 s when switching per target). Hits are filtered to the
# target's synonyms afterwards. No "wall": it steals door boxes.
FIND_VOCAB = tuple(p for group in FIND_SYNONYMS.values() for p in group) + (
    "car",
    "bus",
    "truck",
    "traffic light",
    "chair",
    "table",
    "bench",
    "pole",
    "bicycle",
)


class FinderUnavailable(RuntimeError):
    """The finder model could not be loaded; callers fall back to stored labels."""


@dataclass
class FindHit:
    label: str
    prompt: str
    score: float
    dist_m: Optional[float]
    bearing: str
    box: tuple

    @property
    def certain(self) -> bool:
        return self.score >= STRONG_SCORE


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def prompts_for(target: str) -> List[str]:
    t = " ".join((target or "").lower().split())
    if not t:
        return []
    return list(FIND_SYNONYMS.get(t, (t,)))


def select_hit(hits: List[FindHit]) -> Optional[FindHit]:
    """Nearest strong hit with a distance, else the best strong, else the best weak."""
    strong = [h for h in hits if h.certain]
    if strong:
        measured = [h for h in strong if h.dist_m is not None]
        if measured:
            return min(measured, key=lambda h: h.dist_m)
        return max(strong, key=lambda h: h.score)
    return max(hits, key=lambda h: h.score, default=None)


class ObjectFinder:
    """Own YOLO-World instance; safe to construct without weights or GPU."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        imgsz: Optional[int] = None,
        conf: float = FIND_MIN_SCORE,
    ):
        self.model_name = model_name or os.getenv("FIND_MODEL") or DEFAULT_FIND_MODEL
        self.imgsz = imgsz or _env_int("FIND_IMGSZ", DEFAULT_FIND_IMGSZ)
        self.conf = conf
        self.device = yolo_device()
        self._model = None
        self._failed = False
        self._vocab: Optional[List[str]] = None
        # Loading (a ~90 MB download on first run) happens outside YOLO_LOCK so
        # it never stalls live labeling; this only stops a double load.
        self._load_lock = threading.Lock()

    def _ensure_model(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            if self._failed:
                raise FinderUnavailable(self.model_name)
            try:
                from ultralytics import YOLO

                self._model = YOLO(self.model_name)
                log(
                    "find",
                    "finder model loaded",
                    model=self.model_name,
                    device=self.device,
                    imgsz=self.imgsz,
                )
            except Exception as exc:
                # One attempt only: retrying a failed download on every find
                # would stall each answer.
                self._failed = True
                log_exc("find", "finder model unavailable", exc)
                raise FinderUnavailable(self.model_name) from exc

    def _ensure_vocab(self, prompts: List[str]) -> None:
        """Set FIND_VOCAB once; a new word is appended (one slow call) and kept."""
        vocab = self._vocab if self._vocab is not None else list(FIND_VOCAB)
        missing = [p for p in prompts if p not in vocab]
        if self._vocab is None or missing:
            self._vocab = vocab + missing
            self._model.set_classes(list(self._vocab))

    def _predict(self, rgb_bgr: np.ndarray):
        try:
            return self._model.predict(
                rgb_bgr, conf=self.conf, verbose=False, imgsz=self.imgsz, device=self.device
            )
        except Exception as exc:
            if self.device == "cpu":
                raise
            log_exc("find", f"finder on {self.device} failed, falling back to cpu", exc)
            self.device = "cpu"
            return self._model.predict(
                rgb_bgr, conf=self.conf, verbose=False, imgsz=self.imgsz, device=self.device
            )

    def find(
        self,
        rgb_bgr: np.ndarray,
        depth: np.ndarray,
        conf_map: np.ndarray,
        target: str,
    ) -> List[FindHit]:
        prompts = prompts_for(target)
        if not prompts:
            return []
        label = " ".join(target.lower().split())
        self._ensure_model()
        # Held for the whole pass: live labeling skips while we run.
        with YOLO_LOCK:
            self._ensure_vocab(prompts)
            boxes = [b for b in iter_boxes(self._predict(rgb_bgr)) if b[4] in prompts]

        w = rgb_bgr.shape[1]
        hits = []
        for x1, y1, x2, y2, prompt, score in boxes:
            z, _ = median_depth_in_box(depth, conf_map, x1, y1, x2, y2)
            hits.append(
                FindHit(
                    label=label,
                    prompt=prompt,
                    score=score,
                    dist_m=z,
                    bearing=bearing_from_cx(0.5 * (x1 + x2), w),
                    box=(int(x1), int(y1), int(x2), int(y2)),
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def warmup(self, shape: tuple = (1920, 1440, 3)) -> None:
        """Load + one pass so the first real find skips load and MPS compile."""
        dummy = np.zeros(shape, dtype=np.uint8)
        depth = np.ones(shape[:2], dtype=np.float32)
        conf = np.full(shape[:2], 2, dtype=np.uint8)
        self.find(dummy, depth, conf, "door")
