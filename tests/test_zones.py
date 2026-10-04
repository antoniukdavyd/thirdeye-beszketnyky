"""Zone smoothing + beep hysteresis."""

from __future__ import annotations

import numpy as np

from assist.perception.depth_zones import (
    ALERT_CENTER_M,
    BeepAlert,
    ZoneClearance,
    ZoneFilter,
    compute_zones,
)


def _depth_wall(center_m: float, h: int = 120, w: int = 180) -> tuple:
    depth = np.full((h, w), 3.0, dtype=np.float32)
    conf = np.full((h, w), 2, dtype=np.uint8)
    # waist band center third
    y0, y1 = int(h * 0.12), int(h * 0.55)
    w3 = w // 3
    depth[y0:y1, w3 : 2 * w3] = center_m
    return depth, conf


def test_compute_zones_center_close():
    depth, conf = _depth_wall(0.4)
    z = compute_zones(depth, conf)
    assert z.center is not None
    assert z.center < 0.6


def test_zone_filter_ema_smooths_spike():
    zf = ZoneFilter(alpha=0.4)
    a = zf.update(ZoneClearance(2.0, 2.0, 2.0))
    assert a.center == 2.0
    b = zf.update(ZoneClearance(2.0, 0.3, 2.0))  # spike
    assert b.center is not None
    assert b.center > 0.3  # not fully jumped to spike
    assert b.center < 2.0


def test_beep_hysteresis_no_chatter():
    tones = []
    beep = BeepAlert(enable_audio=False)
    beep._tone = lambda freq, duration=0.06, volume=0.25: tones.append(freq)

    # Enter alert
    beep.update(ZoneClearance(2.0, ALERT_CENTER_M - 0.1, 2.0))
    assert len(tones) >= 1
    n = len(tones)
    # Oscillate just above enter threshold but below exit → still alerting, may beep by interval
    beep._last_beep = 0.0
    beep.update(ZoneClearance(2.0, ALERT_CENTER_M + 0.05, 2.0))
    # Still in hysteresis band — alert stays on
    assert beep._alert_center is True
    # Clear well above exit
    for _ in range(5):
        beep.update(ZoneClearance(2.0, ALERT_CENTER_M + 0.4, 2.0))
    assert beep._alert_center is False
