"""Assemble SENSOR_JSON for LLM / UI — feed-all, no class-type priority."""

from __future__ import annotations

from typing import List

from .depth_zones import ALERT_CENTER_M, ALERT_SIDE_M, ZoneClearance
from .detect import DetectedObject

# Technical caps only (speed / noise), not importance-by-type
MAX_OBJECTS = 15
MIN_DEPTH_M = 0.05
MAX_DEPTH_M = 12.0
# Anything closer than this in the walking line is reported as "ahead". The
# beep threshold (ALERT_CENTER_M) is far too short for this: the vision model
# turned "path ok" with a chair 0.92 m ahead into "The path is clear."
PATH_AHEAD_M = 2.0
# Box-center bearings that fall inside the walking line (~±0.25 m at 1 m).
AHEAD_BEARINGS = ("center", "center-left", "center-right")


def _valid_object(o: DetectedObject) -> bool:
    if o.dist_m < MIN_DEPTH_M or o.dist_m > MAX_DEPTH_M:
        return False
    if o.conf_depth <= 0:
        return False
    return True


def path_hint(zones: ZoneClearance, objects: List[DetectedObject]) -> str:
    """Short factual path note for the LLM. Never claims the path is clear.

    Zones only see the waist-up band, and YOLO only sees its classes, so the
    nearest of both in the walking line is reported; with neither, it says
    "nothing detected", not "clear".
    """
    ahead = [
        (o.dist_m, o.label)
        for o in objects
        if o.bearing in AHEAD_BEARINGS and o.dist_m < PATH_AHEAD_M
    ]
    if zones.center is not None and zones.center < PATH_AHEAD_M:
        ahead.append((zones.center, "obstacle"))

    parts = []
    if ahead:
        dist, label = min(ahead)
        prefix = "blocked ahead" if dist < ALERT_CENTER_M else "ahead"
        parts.append(f"{prefix}: {label} {dist:.1f} m")
    elif zones.center is None:
        parts.append("no depth ahead")
    else:
        parts.append(f"nothing detected ahead within {PATH_AHEAD_M:.0f} m")
    if zones.left is not None and zones.left < ALERT_SIDE_M:
        parts.append("left close")
    if zones.right is not None and zones.right < ALERT_SIDE_M:
        parts.append("right close")
    return "; ".join(parts)


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
        "hint": path_hint(zones, kept),
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
