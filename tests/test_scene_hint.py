"""SENSOR_JSON hint must never call the path clear with something in the way.

The vision model repeats the hint: "path ok" with a chair 0.92 m ahead came
back as "The path is clear." (session 2026-10-04 07:31:53).
"""

from __future__ import annotations

from assist.perception.depth_zones import ZoneClearance
from assist.perception.detect import DetectedObject
from assist.perception.scene import build_scene


def _obj(label: str, dist_m: float, bearing: str) -> DetectedObject:
    return DetectedObject(
        label=label, dist_m=dist_m, bearing=bearing, conf_depth=2, box=(0, 0, 1, 1), score=0.8
    )


def _hint(zones, objects=()) -> str:
    return build_scene(ZoneClearance(*zones), list(objects))["hint"]


def _claims_clear(hint: str) -> bool:
    h = hint.lower()
    return "clear" in h or "path ok" in h


def test_obstacle_ahead_within_walking_range_is_not_ok():
    # Logged zones at 07:31:53, chair dead ahead.
    hint = _hint((2.92, 0.92, 0.84))
    assert not _claims_clear(hint)
    assert "ahead" in hint and "0.9" in hint


def test_detected_object_in_path_is_named():
    # Logged zones at 07:31:15: center zone far, chair center-right ~1 m.
    hint = _hint((2.55, 2.55, 1.02), [_obj("chair", 1.0, "center-right")])
    assert not _claims_clear(hint)
    assert "chair" in hint and "ahead" in hint


def test_nearest_thing_ahead_wins():
    hint = _hint((3.0, 1.4, 3.0), [_obj("chair", 0.9, "center")])
    assert "chair" in hint and "0.9" in hint


def test_very_close_center_is_blocked():
    hint = _hint((3.0, 0.5, 3.0))
    assert "blocked" in hint


def test_side_objects_and_far_objects_do_not_block_path():
    hint = _hint((3.0, 3.0, 3.0), [_obj("chair", 0.7, "left"), _obj("table", 3.5, "center")])
    assert "chair" not in hint and "table" not in hint
    assert not _claims_clear(hint)
    assert "nothing detected ahead" in hint


def test_side_zone_close_still_reported():
    hint = _hint((0.5, 3.0, 3.0))
    assert "left close" in hint


def test_no_depth_ahead():
    hint = _hint((None, None, None))
    assert "no depth" in hint
    assert not _claims_clear(hint)
