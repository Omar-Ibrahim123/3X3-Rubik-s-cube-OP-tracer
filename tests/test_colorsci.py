"""Tests for LAB + CIEDE2000 perceptual color and palette classification."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from vision.colors import ColorClassifier
from vision.colorsci import ciede2000, rgb_distance, rgb_to_lab


def test_lab_known_values():
    L, a, b = rgb_to_lab((255, 255, 255))
    assert abs(L - 100) < 0.5 and abs(a) < 1 and abs(b) < 1
    L0, _, _ = rgb_to_lab((0, 0, 0))
    assert abs(L0) < 0.5


def test_distance_identity_zero_and_symmetric():
    assert rgb_distance((120, 50, 30), (120, 50, 30)) == pytest.approx(0.0, abs=1e-6)
    d1 = rgb_distance((150, 26, 41), (199, 114, 81))
    d2 = rgb_distance((199, 114, 81), (150, 26, 41))
    assert d1 == pytest.approx(d2)


def test_red_orange_well_separated():
    # The pair that kept confusing: perceptual distance must be clearly larger
    # between red and orange than within red.
    red_orange = rgb_distance((150, 26, 41), (199, 114, 81))
    red_red = rgb_distance((150, 26, 41), (148, 30, 45))
    assert red_orange > 10 * red_red


# --- Palette classification on the user's REAL captured colors ---

REAL_PALETTE = {
    "U": (182, 200, 140),
    "F": (87, 149, 134),
    "D": (188, 206, 139),
    "B": (0, 83, 199),
    "R": (148, 60, 99),
    "L": (199, 114, 81),
}


def test_palette_classifies_real_oranges_as_orange():
    clf = ColorClassifier.from_palette(REAL_PALETTE)
    # These oranges were being misread as RED before the LAB/palette switch.
    for orange in [(194, 109, 65), (200, 122, 108), (195, 110, 67)]:
        assert clf.classify(orange) == "L", f"{orange} -> {clf.classify(orange)}"


def test_palette_classifies_all_real_samples():
    clf = ColorClassifier.from_palette(REAL_PALETTE)
    samples = {
        "R": [(150, 26, 41), (146, 54, 86)],
        "F": [(96, 154, 138), (87, 149, 134)],
        "B": [(5, 94, 204), (0, 83, 199)],
        "D": [(190, 209, 141), (188, 206, 139)],
        "L": [(194, 109, 65), (199, 114, 81)],
    }
    for face, cs in samples.items():
        for c in cs:
            assert clf.classify(c) == face, f"{c} -> {clf.classify(c)} (want {face})"


def test_palette_requires_all_six():
    with pytest.raises(ValueError):
        ColorClassifier.from_palette({"U": (1, 2, 3)})
