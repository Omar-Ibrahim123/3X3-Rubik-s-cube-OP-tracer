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
    dominant_color,
    has_logo_signature,
    filter_similar_squares,
    find_cube_by_neighbors,
    is_sticker_blob,
    is_sticker_square,
    pick_white_reference,
    score_face_candidate,
    subdivide_face,
    trimmed_median_color,
)


# --------------------------- Qbr-style detection ---------------------------

def _grid_boxes(x0=200, y0=150, step=50, w=40):
    """9 sticker boxes (x,y,w,h) in a clean 3x3 grid; gap = step - w."""
    boxes = []
    for row in range(3):
        for col in range(3):
            boxes.append((x0 + col * step, y0 + row * step, w, w))
    return boxes


def test_is_sticker_square_accepts_typical():
    assert is_sticker_square(w=45, h=45, area=45 * 45 * 0.9)


def test_is_sticker_square_rejects_non_square_and_sparse():
    assert not is_sticker_square(w=45, h=20, area=600)       # not square
    assert not is_sticker_square(w=45, h=45, area=45 * 45 * 0.3)  # too empty
    assert not is_sticker_square(w=5, h=5, area=20)          # too small


def test_find_cube_by_neighbors_clean_grid():
    boxes = _grid_boxes()
    face = find_cube_by_neighbors(boxes)
    assert face is not None and len(face) == 9
    # Reading order: first box is top-left, last is bottom-right.
    assert face[0][0] < face[2][0]       # left < right in top row
    assert face[0][1] < face[6][1]       # top < bottom


def test_find_cube_by_neighbors_rejects_too_few():
    # Fewer than min_matches (default 7) squares: no lattice to anchor to.
    assert find_cube_by_neighbors(_grid_boxes()[:6]) is None


def test_find_cube_by_neighbors_reconstructs_missing_center():
    # The real-hardware case: 8 OUTER stickers detected, the logo'd CENTER
    # sticker missing. The lattice must be completed by inferring the center.
    boxes = _grid_boxes()
    # Drop the center (index 4 in a row-major 3x3).
    without_center = [b for i, b in enumerate(boxes) if i != 4]
    assert len(without_center) == 8
    face = find_cube_by_neighbors(without_center)
    assert face is not None and len(face) == 9
    # The reconstructed center must sit at the geometric middle of the face.
    cx = (face[0][0] + face[8][0] + face[8][2]) / 2.0  # rough middle x
    center_box = face[4]
    center_cx = center_box[0] + center_box[2] / 2.0
    center_cy = center_box[1] + center_box[3] / 2.0
    expected = boxes[4]
    exp_cx = expected[0] + expected[2] / 2.0
    exp_cy = expected[1] + expected[3] / 2.0
    assert abs(center_cx - exp_cx) < 6 and abs(center_cy - exp_cy) < 6


def test_find_cube_by_neighbors_reconstructs_missing_corner():
    # A missing CORNER sticker (glare) must also be reconstructed in place.
    boxes = _grid_boxes()
    without_corner = [b for i, b in enumerate(boxes) if i != 0]  # drop top-left
    face = find_cube_by_neighbors(without_corner)
    assert face is not None and len(face) == 9
    # Top-left (reading-order index 0) is the reconstructed one; it should be
    # up-and-left of the center.
    assert face[0][0] < face[4][0] and face[0][1] < face[4][1]


def test_find_cube_by_neighbors_two_missing_still_recovers():
    # Up to two missing stickers (min_matches=7) must still complete the grid.
    boxes = _grid_boxes()
    sparse = [b for i, b in enumerate(boxes) if i not in (4, 0)]
    assert len(sparse) == 7
    face = find_cube_by_neighbors(sparse)
    assert face is not None and len(face) == 9


def test_find_cube_by_neighbors_strict_mode_still_requires_all_nine():
    # Opting back into strict behaviour (min_matches=9) rejects an 8/9 face.
    boxes = _grid_boxes()
    without_center = [b for i, b in enumerate(boxes) if i != 4]
    assert find_cube_by_neighbors(without_center, min_matches=9) is None
    # ...but accepts a complete grid.
    assert find_cube_by_neighbors(boxes, min_matches=9) is not None


def test_find_cube_by_neighbors_ignores_scattered_noise():
    # A clean 3x3 grid plus stray squares far away that don't form a grid.
    boxes = _grid_boxes() + [(20, 20, 40, 40), (600, 440, 40, 40)]
    face = find_cube_by_neighbors(boxes)
    assert face is not None and len(face) == 9


def test_find_cube_by_neighbors_no_grid():
    # 9 squares in a horizontal line -> no 3x3 center with 9 neighbors.
    line = [(50 + i * 60, 240, 40, 40) for i in range(9)]
    assert find_cube_by_neighbors(line) is None


# --------------------------- score_face_candidate ---------------------------

FW, FH = 640.0, 480.0


def test_score_central_square_beats_edge_square():
    # Same-sized square: one centered, one hugging the left edge.
    center = score_face_candidate(320, 240, 180, 180, 0.9, FW, FH)
    edge = score_face_candidate(60, 240, 180, 180, 0.9, FW, FH)
    assert center > edge > 0


def test_score_rejects_non_square():
    assert score_face_candidate(320, 240, 300, 120, 0.9, FW, FH) == 0.0


def test_score_rejects_poor_fill():
    # A blobby region that doesn't fill its bounding rect (skin/wall splotch).
    assert score_face_candidate(320, 240, 180, 180, 0.4, FW, FH) == 0.0


def test_score_accepts_close_and_far_cube():
    # Close: cube ~45% of width. Far: cube ~18% of width. Both should score > 0.
    close = score_face_candidate(320, 240, 290, 290, 0.9, FW, FH)
    far = score_face_candidate(320, 240, 115, 115, 0.9, FW, FH)
    assert close > 0 and far > 0


def test_score_rejects_whole_frame_grab():
    # A region covering almost the entire frame is not a held cube.
    assert score_face_candidate(320, 240, 630, 470, 0.9, FW, FH) == 0.0


def test_score_prefers_central_cube_over_corner_poster():
    # A vivid square poster in the corner must lose to a central cube, even if
    # the poster is a touch bigger.
    cube = score_face_candidate(320, 240, 200, 200, 0.85, FW, FH)
    poster = score_face_candidate(590, 60, 210, 210, 0.9, FW, FH)
    assert cube > poster


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
    # A very bright yellow (high saturation) must NOT beat a slightly dimmer
    # neutral white.
    samples = [
        (250.0, 250.0, 40.0),   # bright yellow (vivid)
        (225.0, 228.0, 223.0),  # white (neutral)
    ]
    assert pick_white_reference(samples) == (225.0, 228.0, 223.0)


def test_pick_white_picks_bright_white_despite_mild_cast():
    # Real captured data: a bright but slightly blue-cast white center must be
    # chosen over dim flat-grey cells (regression for the muddy-white bug).
    samples = [
        (172.0, 163.0, 163.0), (154.0, 141.0, 154.0), (167.0, 156.0, 156.0),
        (145.0, 125.0, 149.0), (206.0, 231.0, 255.0), (167.0, 156.0, 153.0),
        (51.0, 41.0, 44.0), (26.0, 54.0, 120.0), (27.0, 55.0, 120.0),
    ]
    assert pick_white_reference(samples) == (206.0, 231.0, 255.0)


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


# --------------------------- dominant_color ---------------------------

def test_dominant_color_ignores_center_logo():
    # 65% white sticker + 35% blue logo -> dominant should be white.
    white = [(158.0, 184.0, 233.0)] * 65
    blue_logo = [(20.0, 110.0, 220.0)] * 35
    r, g, b = dominant_color(white + blue_logo)
    assert r > 120 and g > 150 and b > 200  # white-ish, not the blue logo
    # And clearly not the blue logo color.
    assert not (r < 60 and b > 180)


def test_dominant_color_pure_patch():
    r, g, b = dominant_color([(200.0, 50.0, 40.0)] * 40)
    assert abs(r - 200) < 48 and abs(g - 50) < 48 and abs(b - 40) < 48


def test_dominant_color_empty():
    assert dominant_color([]) == (0.0, 0.0, 0.0)


# --------------------------- has_logo_signature ---------------------------

def test_logo_signature_detects_white_plus_logo():
    # Real blue-cast white sticker + a blue logo blob -> white face.
    patch = [(158.0, 184.0, 233.0)] * 72 + [(20.0, 110.0, 220.0)] * 28
    assert has_logo_signature(patch)


def test_logo_signature_not_triggered_by_plain_blue():
    # A plain blue STICKER must not look like a white face with a blue logo
    # (brightness is the discriminator: a blue sticker isn't bright).
    assert not has_logo_signature([(0.0, 103.0, 223.0)] * 100)


def test_logo_signature_not_triggered_by_plain_colors():
    for rgb in [(158.0, 184.0, 233.0),   # plain white, no logo
                (108.0, 170.0, 148.0),   # green
                (159.0, 70.0, 103.0),    # red
                (195.0, 216.0, 150.0)]:  # yellow
        assert not has_logo_signature([rgb] * 100), rgb


def test_logo_signature_empty():
    assert not has_logo_signature([])


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
