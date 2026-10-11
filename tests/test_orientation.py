"""Tests for per-face orientation recovery (pure, no camera / no OpenCV)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from cube.orientation import (
    _CW90,
    _CW180,
    _CW270,
    _IDENTITY,
    FACE_ORDER,
    assemble,
    is_solvable,
    is_structurally_valid,
    is_valid_cube,
    rotate_face,
    solve_orientation,
)
from tests.cube_moves import SOLVED, apply_sequence


def _split(facelets):
    """54-char string -> {face: 9-char string} in Kociemba order."""
    return {FACE_ORDER[i]: facelets[i * 9:(i + 1) * 9] for i in range(6)}


# --------------------------- rotate_face ---------------------------

def test_rotate_face_identity_and_full_turn():
    f = "ABCDEFGHI"
    assert rotate_face(f, 0) == f
    assert rotate_face(f, 4) == f  # four quarter turns == identity


def test_rotate_face_cw_quarter():
    # 0 1 2 / 3 4 5 / 6 7 8  rotated CW ->  6 3 0 / 7 4 1 / 8 5 2
    assert rotate_face("012345678", 1) == "630741852"


def test_rotate_face_keeps_center_fixed():
    for q in range(4):
        assert rotate_face("ABCDXFGHI", q)[4] == "X"


def test_rotate_face_rejects_bad_length():
    with pytest.raises(ValueError):
        rotate_face("ABC", 1)


# --------------------------- is_valid_cube ---------------------------

def test_solved_cube_is_valid():
    assert is_valid_cube(SOLVED)


def test_scramble_is_valid():
    assert is_valid_cube(apply_sequence("R U R' U' F2 L D B'"))


def test_wrong_length_is_invalid():
    assert not is_valid_cube("UUU")


def test_wrong_color_counts_invalid():
    bad = "U" * 10 + "R" * 8 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9
    assert not is_valid_cube(bad)


# --------------------------- is_solvable (parity) ---------------------------

def test_legal_scrambles_are_solvable():
    from tests.cube_moves import apply_sequence as seq
    for s in [
        "R U R' U'",
        "F2 B2 L2 R2 U2 D2",
        "R U2 F' L D B R2 U' L' F",
        "D L F2 U' R B2 L' D2 F R' U B",
    ]:
        assert is_solvable(seq(s)), s


def test_single_edge_flip_is_unsolvable():
    # Flip one edge in place: structurally valid but violates edge-orientation.
    scr = list(apply_sequence("R U R' F2 L"))
    scr[1], scr[46] = scr[46], scr[1]  # swap an edge's two stickers
    flipped = "".join(scr)
    assert is_structurally_valid(flipped)
    assert not is_solvable(flipped)


def test_single_corner_twist_is_unsolvable():
    scr = list(apply_sequence("R U R' F2 L"))
    scr[0], scr[36], scr[47] = scr[36], scr[47], scr[0]  # cycle a corner's 3 stickers
    twisted = "".join(scr)
    assert is_structurally_valid(twisted)
    assert not is_solvable(twisted)


def test_two_piece_swap_is_unsolvable():
    scr = list(apply_sequence("R U R' F2 L"))
    for i, j in [(0, 2), (36, 45), (47, 11)]:  # swap two whole corners
        scr[i], scr[j] = scr[j], scr[i]
    swapped = "".join(scr)
    assert is_structurally_valid(swapped)
    assert not is_solvable(swapped)


# --------------------------- solve_orientation ---------------------------

def test_solved_cube_needs_no_rotation():
    result = solve_orientation(_split(SOLVED))
    assert result.ok
    assert result.facelets == SOLVED
    assert all(q == 0 for q in result.rotations.values())


def test_recovers_a_single_rotated_face():
    # Take a real scramble, rotate ONE face 90 deg (simulating a tilted
    # capture), and confirm the solver puts it back and recovers the cube.
    scramble = apply_sequence("D L F2 U' R B2 L' D2 F R' U B")
    faces = _split(scramble)
    faces["F"] = rotate_face(faces["F"], 1)
    result = solve_orientation(faces)
    assert result.ok, result.message
    assert result.facelets == scramble
    assert result.rotations["F"] == 3  # 3 more quarter-turns undoes the +1


def test_recovers_several_rotated_faces():
    # Rotate three faces of a scramble that has a UNIQUE valid orientation.
    scramble = apply_sequence("U R2 F L' D2 B R")
    faces = _split(scramble)
    faces["U"] = rotate_face(faces["U"], 2)
    faces["D"] = rotate_face(faces["D"], 1)
    faces["L"] = rotate_face(faces["L"], 3)
    result = solve_orientation(faces)
    assert result.ok, result.message
    assert result.facelets == scramble


def test_ambiguous_orientation_is_declined_not_guessed():
    # Some states genuinely admit more than one valid orientation; the solver
    # must DECLINE (ok=False) rather than silently pick one.
    scramble = apply_sequence("R U2 F' L D B R2 U' L' F")
    faces = _split(scramble)
    faces["U"] = rotate_face(faces["U"], 2)
    faces["D"] = rotate_face(faces["D"], 2)
    faces["L"] = rotate_face(faces["L"], 1)
    result = solve_orientation(faces)
    assert not result.ok
    assert result.candidate_count >= 2
    assert "disambiguate" in result.message


def test_reports_missing_faces():
    faces = _split(SOLVED)
    del faces["B"]
    result = solve_orientation(faces)
    assert not result.ok
    assert "missing" in result.message


def test_scanner_classify_scan_recovers_rotated_capture():
    """End-to-end: the scanner un-rotates faces captured at odd angles.

    Build colour grids for a real scramble, rotate three faces (as a free-hand
    capture would), feed them through StaticScanner.classify_scan, and confirm
    it returns the correct, solvable scramble rather than raising on an
    impossible cube.
    """
    from vision.colors import ColorClassifier
    from vision.scanner import StaticScanner
    from tests.test_vision import CAL, _faces_from_facelets

    scramble = apply_sequence("U R2 F L' D2 B R")
    grids = _faces_from_facelets(scramble)  # {face: [9 RGB]} correctly oriented

    # Rotate three faces' 9-sample grids to simulate tilted captures.
    def rot_grid(grid, q):
        remap = {0: _IDENTITY, 1: _CW90, 2: _CW180, 3: _CW270}[q]
        return [grid[remap[i]] for i in range(9)]

    grids["U"] = rot_grid(grids["U"], 2)
    grids["D"] = rot_grid(grids["D"], 1)
    grids["L"] = rot_grid(grids["L"], 3)

    scanner = StaticScanner()
    scanner.set_classifier(ColorClassifier.from_centers(CAL))
    assert scanner.classify_scan(grids) == scramble


def test_real_scan_recovers_valid_cube():
    """Regression: the user's real captured scan.

    This 54-char string had correct colour counts but described an impossible
    cube (solver error 'invalid piece at H: [U, D, L]'). Brute force showed a
    unique valid orientation exists (U and D faces each rotated 180 deg). The
    solver must find exactly that and produce a valid cube.
    """
    real = "RFUDUUDULFRRURLBDLBUUBFRLFRBBFLDRDDBFLRBLLUFUFBDFBRDDL"
    assert not is_valid_cube(real)  # as captured: invalid
    faces = _split(real)
    result = solve_orientation(faces)
    assert result.ok, result.message
    assert is_valid_cube(result.facelets)
    # The known-unique solution rotates U and D by 180 degrees.
    assert result.rotations["U"] == 2
    assert result.rotations["D"] == 2
    assert result.rotations["R"] == 0
