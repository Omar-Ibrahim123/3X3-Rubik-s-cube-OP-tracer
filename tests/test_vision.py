"""Tests for the pure vision logic (no camera / no OpenCV required).

Covers color classification (including the classic orange/red and white/yellow
confusions), facelet assembly, state validation, and the scanner's camera-free
classify_scan() path end to end against a real scramble.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from solver import trace
from cube.state import CubeState
from vision.colors import (
    ColorClassifier,
    assemble_facelets,
    color_distance,
    validate_facelets,
)
from vision.scanner import StaticScanner
from tests.cube_moves import SOLVED, apply_sequence


# A realistic calibration: typical sticker colors (R, G, B) for a standard cube.
# These stand in for the center colors the user shows during calibration.
CAL = {
    "U": (245, 245, 245),   # white
    "D": (255, 215, 0),     # yellow
    "F": (0, 155, 72),      # green
    "B": (0, 70, 173),      # blue
    "R": (183, 18, 52),     # red
    "L": (255, 88, 0),      # orange
}

# Map a face letter to a slightly noisy sample color, to simulate the camera
# reading a sticker of that color under mild lighting variation.
NOISY = {
    "U": (235, 240, 230),
    "D": (250, 205, 20),
    "F": (10, 150, 80),
    "B": (5, 65, 165),
    "R": (175, 25, 55),
    "L": (248, 95, 10),
}


def _classifier():
    return ColorClassifier.from_centers(CAL)


# --------------------------- color distance ---------------------------

def test_distance_identity_is_zero():
    assert color_distance((100, 50, 25), (100, 50, 25)) == 0.0


def test_distance_is_symmetric():
    a, b = (200, 10, 10), (10, 200, 10)
    assert color_distance(a, b) == pytest.approx(color_distance(b, a))


def test_distance_tolerates_brightness_change():
    # Same hue, different brightness (shadow) should be closer than a hue change
    bright_red = (200, 20, 20)
    dark_red = (100, 10, 10)
    green = (20, 200, 20)
    assert color_distance(bright_red, dark_red) < color_distance(bright_red, green)


# --------------------------- classifier ---------------------------

def test_classifies_each_calibrated_color():
    clf = _classifier()
    for face, rgb in CAL.items():
        assert clf.classify(rgb) == face


def test_classifies_noisy_samples():
    clf = _classifier()
    for face, rgb in NOISY.items():
        assert clf.classify(rgb) == face, f"{face} misclassified"


def test_orange_vs_red_disambiguated_by_calibration():
    # The hardest pair. With calibrated references, orange and red samples must
    # not cross over.
    clf = _classifier()
    assert clf.classify((255, 90, 5)) == "L"    # clearly orange
    assert clf.classify((180, 20, 50)) == "R"   # clearly red


def test_confidence_margin_small_for_ambiguous():
    clf = _classifier()
    # A color halfway between red and orange should have a small margin.
    _, clear_margin = clf.classify_with_confidence((255, 88, 0))  # exact orange
    _, amb_margin = clf.classify_with_confidence((220, 55, 25))   # between
    assert amb_margin < clear_margin


def test_uncalibrated_classifier_raises():
    with pytest.raises(RuntimeError):
        ColorClassifier().classify((1, 2, 3))


def test_missing_calibration_face_raises():
    partial = dict(CAL)
    del partial["B"]
    with pytest.raises(ValueError):
        ColorClassifier.from_centers(partial)


# --------------------------- assembly + validation ---------------------------

def _faces_from_facelets(facelets, cal=CAL):
    """Build the 6 face color-grids that correspond to a facelet string,
    by mapping each face letter to its calibrated color."""
    faces = {}
    order = ("U", "R", "F", "D", "L", "B")
    for i, face in enumerate(order):
        chunk = facelets[i * 9 : (i + 1) * 9]
        faces[face] = [cal[ch] for ch in chunk]
    return faces


def test_assemble_round_trips_solved():
    clf = _classifier()
    faces = _faces_from_facelets(SOLVED)
    assert assemble_facelets(faces, clf) == SOLVED


def test_assemble_round_trips_scramble():
    clf = _classifier()
    scrambled = apply_sequence("D L F2 U' R B2 L' D2 F R' U B")
    faces = _faces_from_facelets(scrambled)
    assert assemble_facelets(faces, clf) == scrambled


def test_validate_accepts_solved():
    assert validate_facelets(SOLVED).ok


def test_validate_rejects_wrong_length():
    r = validate_facelets("UUU")
    assert not r.ok and any("length" in e for e in r.errors)


def test_validate_rejects_bad_color_counts():
    bad = "U" * 10 + "R" * 8 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9
    r = validate_facelets(bad)
    assert not r.ok
    assert any("appears 10" in e for e in r.errors)
    assert any("appears 8" in e for e in r.errors)


def test_validate_rejects_bad_centers():
    # Swap two centers so the center set is wrong (duplicate a center color).
    chars = list(SOLVED)
    chars[4] = "R"  # U center now R -> U appears 8, R appears 10 and centers off
    r = validate_facelets("".join(chars))
    assert not r.ok


# --------------------------- scanner (camera-free path) ---------------------------

def test_scanner_classify_scan_produces_traceable_state():
    scrambled = apply_sequence("R U R' U R U2 R'")
    faces = _faces_from_facelets(scrambled)
    scanner = StaticScanner()
    scanner.set_classifier(_classifier())
    facelets = scanner.classify_scan(faces)
    assert facelets == scrambled
    # And the whole pipeline runs: scan -> state -> trace.
    result = trace(CubeState.from_facelets(facelets))
    assert result.corners or result.edges


def test_scanner_classify_scan_rejects_invalid():
    # Build faces that classify to an impossible state (all white) -> invalid.
    scanner = StaticScanner()
    scanner.set_classifier(_classifier())
    faces = {f: [CAL["U"]] * 9 for f in ("U", "R", "F", "D", "L", "B")}
    with pytest.raises(ValueError):
        scanner.classify_scan(faces)


def test_scanner_requires_classifier():
    scanner = StaticScanner()
    with pytest.raises(RuntimeError):
        scanner.classify_scan({})
