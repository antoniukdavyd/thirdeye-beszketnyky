"""HUD scaling helpers."""

from assist.app import hud_metrics, hud_scale


def test_hud_scale_720p():
    assert hud_scale(720) >= 1.0
    m = hud_metrics(720)
    assert m["zone_font"] >= 0.9
    assert m["bar_h"] >= 72


def test_hud_scale_1080p():
    m = hud_metrics(1080)
    assert m["zone_font"] >= 0.9
    assert m["bar_h"] >= 72
    assert m["scale"] == 1080 / 480
