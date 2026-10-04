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


def test_label_color_distinct_per_class():
    from assist.app import label_color
    from assist.perception.detect import DEFAULT_CLASSES

    colors = [label_color(c) for c in DEFAULT_CLASSES]
    assert len(set(colors)) == len(DEFAULT_CLASSES)


def test_label_color_stable_and_case_insensitive():
    from assist.app import label_color

    assert label_color("Person") == label_color("person")
    assert label_color("person") == label_color("person")


def test_label_color_unknown_label_is_stable_bgr():
    from assist.app import label_color

    c = label_color("scooter")
    assert c == label_color("scooter")
    assert len(c) == 3 and all(0 <= v <= 255 for v in c)
