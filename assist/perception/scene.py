"""Assemble SENSOR_JSON for LLM / UI — feed-all, no class-type priority."""

from __future__ import annotations

from typing import List

from .depth_zones import ZoneClearance
from .detect import DetectedObject

# Technical caps only (speed / noise), not importance-by-type
MAX_OBJECTS = 15
MIN_DEPTH_M = 0.05
MAX_DEPTH_M = 12.0


def _valid_object(o: DetectedObject) -> bool:
    if o.dist_m < MIN_DEPTH_M or o.dist_m > MAX_DEPTH_M:
        return False
    if o.conf_depth <= 0:
        return False
    return True


def build_scene(
    zones: ZoneClearance,
    objects: List[DetectedObject],
    device: str = "lidar",
) -> dict:
    """
    Full scene for UI + LLM.
    Objects are NOT filtered/sorted by class. Cap by count only.
    Order: nearest-first (sensor order), LLM decides importance.
    """
    kept = [o for o in objects if _valid_object(o)]
    kept = kept[:MAX_OBJECTS]
    return {
        "zones_m": zones.as_dict(),
        "objects": [o.as_dict() for o in kept],
        "device": device,
        "hint": zones.hint(),
    }


def compact_scene_for_llm(scene: dict) -> dict:
    """Strip heavy fields (score, conf_depth) before API call."""
    objs = []
    for o in scene.get("objects") or []:
        objs.append(
            {
                "label": o.get("label"),
                "dist_m": o.get("dist_m"),
                "bearing": o.get("bearing"),
            }
        )
    return {
        "zones_m": scene.get("zones_m") or {},
        "objects": objs,
        "hint": scene.get("hint", ""),
    }
