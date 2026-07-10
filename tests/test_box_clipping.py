"""YOLO box clipping keeps edges inside [0,1] (M2 fix)."""

from src.models.export import clip_box_xywhn

EPS = 1e-9


def _in_bounds(xc, yc, w, h):
    return (
        -EPS <= xc - w / 2
        and xc + w / 2 <= 1 + EPS
        and -EPS <= yc - h / 2
        and yc + h / 2 <= 1 + EPS
    )


def test_edge_box_shrunk_into_bounds():
    # Center near the right edge with a wide box -> would spill past 1 if only
    # the center were clipped.
    assert _in_bounds(*clip_box_xywhn(0.95, 0.5, 0.4, 0.2))


def test_corner_box_in_bounds():
    assert _in_bounds(*clip_box_xywhn(0.02, 0.98, 0.5, 0.5))


def test_interior_box_unchanged():
    assert clip_box_xywhn(0.5, 0.5, 0.2, 0.2) == (0.5, 0.5, 0.2, 0.2)


def test_widths_non_negative():
    _, _, w, h = clip_box_xywhn(1.0, 0.0, 0.3, 0.3)
    assert w >= 0 and h >= 0
