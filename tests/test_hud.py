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


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_place_labels_never_overlap_each_other():
    from assist.app import place_labels

    boxes = [(100, 200, 300, 400)] * 3  # three boxes on the same spot
    rects = place_labels(boxes, [(120, 30)] * 3, 720, 960, reserved=[])
    placed = [r for r in rects if r is not None]
    assert len(placed) == 3
    for i, a in enumerate(placed):
        for b in placed[i + 1 :]:
            assert not _overlap(a, b)


def test_place_labels_avoid_reserved_header_and_bar():
    from assist.app import place_labels

    header, bar = (0, 0, 720, 90), (0, 780, 720, 960)
    rects = place_labels([(50, 10, 400, 700)], [(150, 30)], 720, 960, reserved=[header, bar])
    r = rects[0]
    assert r is not None
    assert not _overlap(r, header) and not _overlap(r, bar)


def test_place_labels_stay_inside_frame():
    from assist.app import place_labels

    r = place_labels([(680, 300, 715, 500)], [(150, 30)], 720, 960, reserved=[])[0]
    assert r is not None
    assert 0 <= r[0] and r[2] <= 720 and 0 <= r[1] and r[3] <= 960


def test_place_labels_none_when_no_room():
    from assist.app import place_labels

    rects = place_labels([(10, 10, 50, 50)], [(40, 20)], 60, 60, reserved=[(0, 0, 60, 60)])
    assert rects == [None]


def test_box_thickness_by_size_and_distance():
    from assist.app import box_thickness

    assert box_thickness((0, 0, 700, 900), 1.0, 720, 960) == 1  # wall-like: thin
    assert box_thickness((100, 100, 200, 300), 1.0, 720, 960) == 3  # near
    assert box_thickness((100, 100, 200, 300), 3.0, 720, 960) == 2


def test_hud_objects_keeps_nearest_eight():
    from assist.app import hud_objects
    from assist.perception.detect import DetectedObject

    objs = [DetectedObject("cup", float(d), "center", 2, (0, 0, 1, 1), 0.5) for d in range(12, 0, -1)]
    shown = hud_objects(objs)
    assert [o.dist_m for o in shown] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def test_place_labels_tall_box_gets_label_inside():
    # Door from under the readout down into the bottom bar: above, top and
    # below are all blocked, so the label must go further down inside the box.
    from assist.app import place_labels

    header, bar = (0, 0, 720, 90), (0, 780, 720, 960)
    r = place_labels([(300, 50, 550, 900)], [(120, 26)], 720, 960, reserved=[header, bar])[0]
    assert r is not None
    assert 50 <= r[1] and r[3] <= 900
    assert not _overlap(r, header) and not _overlap(r, bar)
