"""Tests for the pure cube-face detection geometry (no camera / no OpenCV)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import pytest

import math

from vision.detect import (
    Square,
    detect_face_grid,
    filter_similar_squares,
    is_sticker_blob,
    pick_white_reference,
    subdivide_face,
    trimmed_median_color,
)


# --------------------------- pick_white_reference ---------------------------

def test_pick_white_from_scrambled_face():
    # Center (index 4) is the only white; the other 8 are random vivid colors.
    samples = [
        (200.0, 30.0, 40.0),   # red
        (0.0, 150.0, 70.0),    # green
        (255.0, 140.0, 0.0),   # orange
        (0.0, 90.0, 200.0),    # blue
        (238.0, 240.0, 236.0), # WHITE center
        (255.0, 210.0, 0.0),   # yellow
        (0.0, 150.0, 70.0),    # green
        (200.0, 30.0, 40.0),   # red
        (0.0, 90.0, 200.0),    # blue
    ]
    white = pick_white_reference(samples)
    assert white == (238.0, 240.0, 236.0)


def test_pick_white_prefers_neutral_over_bright_colored():
    # A very bright yellow (high brightness) must NOT beat a slightly dimmer
    # neutral white, because yellow has large channel spread.
    samples = [
        (250.0, 250.0, 40.0),   # bright yellow (big spread)
        (225.0, 228.0, 223.0),  # white (neutral)
    ]
    assert pick_white_reference(samples) == (225.0, 228.0, 223.0)


def test_pick_white_empty():
    assert pick_white_reference([]) == (255.0, 255.0, 255.0)


# --------------------------- subdivide inset ---------------------------

def test_subdivide_inset_pulls_outer_cells_inward():
    full = subdivide_face(300, 300, 300, inset=1.0)
    inset = subdivide_face(300, 300, 300, inset=0.8)
    # Center cell unchanged; a corner cell is closer to center with inset.
    assert inset[4] == pytest.approx(full[4])
    d_full = math.hypot(full[0][0] - 300, full[0][1] - 300)
    d_inset = math.hypot(inset[0][0] - 300, inset[0][1] - 300)
    assert d_inset < d_full


# --------------------------- subdivide_face ---------------------------

def test_subdivide_upright_face_reading_order():
    # Face centered at (300,300), side 150, inset=1.0 -> step 50, cells at +/-50.
    centers = subdivide_face(300, 300, 150, angle=0.0, inset=1.0)
    assert len(centers) == 9
    # Top-left cell, bottom-right cell.
    assert centers[0] == pytest.approx((250, 250))
    assert centers[4] == pytest.approx((300, 300))  # face center
    assert centers[8] == pytest.approx((350, 350))
    # Reading order: rows top->bottom, cols left->right.
    rows = [centers[0:3], centers[3:6], centers[6:9]]
    for row in rows:
        xs = [p[0] for p in row]
        assert xs == sorted(xs)
    assert rows[0][0][1] < rows[2][0][1]


def test_subdivide_rotated_face_keeps_9_centers_near_face():
    centers = subdivide_face(300, 300, 150, angle=math.radians(10))
    assert len(centers) == 9
    # All cells stay within ~size of the face center.
    for (x, y) in centers:
        assert math.hypot(x - 300, y - 300) <= 150


def test_subdivide_center_cell_is_face_center_any_angle():
    for deg in (-30, -10, 0, 15, 40):
        centers = subdivide_face(100, 200, 90, angle=math.radians(deg))
        assert centers[4] == pytest.approx((100, 200))


# --------------------------- trimmed_median_color ---------------------------

def test_trimmed_median_plain_white():
    pix = [(240.0, 242.0, 238.0)] * 50
    r, g, b = trimmed_median_color(pix)
    assert abs(r - 240) < 3 and abs(g - 242) < 3 and abs(b - 238) < 3


def test_trimmed_median_ignores_center_logo():
    # 70% white sticker + 30% dark-grey logo pixels. The logo (dark) is dropped,
    # so the result should read white, not grey.
    white = [(240.0, 240.0, 240.0)] * 70
    logo = [(70.0, 70.0, 72.0)] * 30
    r, g, b = trimmed_median_color(white + logo)
    assert r > 200 and g > 200 and b > 200, (r, g, b)


def test_trimmed_median_ignores_glare():
    # Mostly red with a few blown-out glare pixels; glare dropped from the top.
    red = [(180.0, 30.0, 30.0)] * 80
    glare = [(255.0, 255.0, 255.0)] * 10
    r, g, b = trimmed_median_color(red + glare)
    assert r > 150 and g < 70 and b < 70, (r, g, b)


def test_trimmed_median_empty():
    assert trimmed_median_color([]) == (0.0, 0.0, 0.0)


# A 640x480 frame for area-fraction tests.
FRAME_AREA = 640 * 480


def test_sticker_blob_accepts_typical_sticker():
    # A ~60x60 square sticker on a 640x480 frame.
    assert is_sticker_blob(area=3400, bbox_w=60, bbox_h=60, frame_area=FRAME_AREA)


def test_sticker_blob_rejects_too_small_and_too_big():
    assert not is_sticker_blob(10, 3, 3, FRAME_AREA)          # tiny speck
    assert not is_sticker_blob(FRAME_AREA * 0.5, 400, 400, FRAME_AREA)  # huge


def test_sticker_blob_rejects_non_square_aspect():
    # Long thin streak (background), area in range but aspect way off.
    assert not is_sticker_blob(area=3000, bbox_w=150, bbox_h=20,
                               frame_area=FRAME_AREA)


def test_sticker_blob_rejects_low_solidity():
    # Area small relative to its bounding box -> hollow/streaky, not a sticker.
    assert not is_sticker_blob(area=1300, bbox_w=60, bbox_h=60,
                               frame_area=FRAME_AREA)


def test_sticker_blob_tolerates_rounded_corners():
    # Rounded sticker: ~pi/4 of bbox filled (~0.79 solidity) still accepted.
    assert is_sticker_blob(area=60 * 60 * 0.80, bbox_w=60, bbox_h=60,
                           frame_area=FRAME_AREA)


def test_sticker_blob_degenerate_inputs():
    assert not is_sticker_blob(0, 0, 0, FRAME_AREA)
    assert not is_sticker_blob(3400, 60, 60, 0)


def make_grid(cx=300, cy=300, spacing=60, size=50, angle=0.0):
    """9 squares arranged in a 3x3 grid, optionally rotated by `angle` rad."""
    squares = []
    for row in range(3):
        for col in range(3):
            dx = (col - 1) * spacing
            dy = (row - 1) * spacing
            # rotate (dx, dy) by angle
            rx = dx * math.cos(angle) - dy * math.sin(angle)
            ry = dx * math.sin(angle) + dy * math.cos(angle)
            squares.append(Square(cx=cx + rx, cy=cy + ry, size=size))
    return squares


def test_detects_clean_grid_and_orders_reading_order():
    squares = make_grid()
    centers = detect_face_grid(squares)
    assert centers is not None
    assert len(centers) == 9
    # Reading order: row 0 (smallest y) first, left-to-right within row.
    rows = [centers[0:3], centers[3:6], centers[6:9]]
    for row in rows:
        xs = [p[0] for p in row]
        assert xs == sorted(xs), "row not left-to-right"
    ys_top = [p[1] for p in rows[0]]
    ys_bot = [p[1] for p in rows[2]]
    assert max(ys_top) < min(ys_bot), "rows not top-to-bottom"


def test_detects_slightly_tilted_grid():
    squares = make_grid(angle=math.radians(8))
    centers = detect_face_grid(squares)
    assert centers is not None
    assert len(centers) == 9


def test_rejects_too_few_squares():
    squares = make_grid()[:7]
    assert detect_face_grid(squares) is None


def test_filters_out_wrong_sized_squares():
    squares = make_grid()
    # Add a huge background rectangle and a tiny speck.
    squares.append(Square(cx=50, cy=50, size=400))
    squares.append(Square(cx=600, cy=600, size=5))
    kept = filter_similar_squares(squares)
    assert len(kept) == 9  # the 9 real stickers only


def test_detects_face_despite_extra_noise_squares():
    squares = make_grid()
    # Scatter a few similar-sized noise squares far from the face centroid.
    squares += [
        Square(cx=20, cy=20, size=50),
        Square(cx=580, cy=40, size=50),
    ]
    centers = detect_face_grid(squares)
    assert centers is not None
    assert len(centers) == 9


def test_rejects_non_grid_arrangement():
    # 9 similar squares but in a line, not a 3x3 grid.
    squares = [Square(cx=100 + i * 40, cy=300, size=50) for i in range(9)]
    assert detect_face_grid(squares) is None


def test_empty_input():
    assert detect_face_grid([]) is None
    assert filter_similar_squares([]) == []
