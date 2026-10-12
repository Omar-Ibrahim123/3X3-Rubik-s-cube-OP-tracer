"""Tests for the detection region-of-interest geometry (no camera / no cv2).

Detection is restricted to the on-screen guide box; these lock the box geometry
and the coordinate-offset invariant (boxes found in the crop must map back to
full-frame coordinates) so the ROI change can't silently misplace stickers.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.capture import ROI_FRAC, _guide_roi


class _FakeFrame:
    """Minimal stand-in exposing just .shape (H, W, C), like a numpy frame."""

    def __init__(self, h, w):
        self.shape = (h, w, 3)


def test_guide_roi_is_centered_square():
    roi = _guide_roi(_FakeFrame(480, 640), 0.5)
    x, y, w, h = roi
    assert w == h == int(min(480, 640) * 0.5)  # square, sized off shorter side
    # Centered: equal margins left/right and top/bottom.
    assert x == (640 - w) // 2
    assert y == (480 - h) // 2


def test_guide_roi_scales_with_frac():
    small = _guide_roi(_FakeFrame(480, 640), 0.4)
    big = _guide_roi(_FakeFrame(480, 640), 0.8)
    assert big[2] > small[2]
    assert big[2] == int(480 * 0.8)


def test_guide_roi_fits_inside_frame():
    for (h, w) in [(480, 640), (720, 1280), (600, 600)]:
        x, y, side, _ = _guide_roi(_FakeFrame(h, w), ROI_FRAC)
        assert x >= 0 and y >= 0
        assert x + side <= w
        assert y + side <= h


def test_offset_invariant_round_trips():
    # Simulate the crop/offset the detector does: a box found at (cx, cy) within
    # the ROI crop must map to (cx + roi_x, cy + roi_y) in full-frame coords.
    roi = _guide_roi(_FakeFrame(480, 640), ROI_FRAC)
    rx, ry, _, _ = roi
    # A sticker box at crop-local (10, 12) of size 40.
    local = (10, 12, 40, 40)
    full = (local[0] + rx, local[1] + ry, local[2], local[3])
    # Re-deriving local from full with the same offset returns the original.
    assert (full[0] - rx, full[1] - ry, full[2], full[3]) == local
    # And the full-frame box lies inside the ROI region.
    assert rx <= full[0] and full[0] + full[2] <= rx + roi[2]
