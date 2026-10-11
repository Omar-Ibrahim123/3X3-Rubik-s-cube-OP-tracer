"""Regression test using REAL captured camera data.

These RGB values came from an actual webcam scan under blue-cast lighting (the
debug output that drove the white-balance + HSV classifier design). Centers of
all six faces must classify correctly after white-anchor calibration. This
locks in the fix so future changes can't silently reintroduce the white/yellow
or yellow/orange confusion seen on real hardware.

Note: non-center stickers from that session are intentionally NOT asserted --
in that capture the grid was mis-sized and many non-center cells sampled the
black body between stickers, which is a sampling issue, not a color-logic one.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.colors import ColorClassifier


# Real white sample measured from the user's webcam (blue-cast lighting).
REAL_WHITE = (202, 222, 240)

# Real center colors of each face from that same scan.
REAL_CENTERS = {
    "U": (201, 222, 239),   # white
    "R": (194, 25, 31),     # red
    "B": (11, 102, 231),    # blue
    "D": (232, 173, 110),   # yellow (very warm on this cube/lighting)
    "L": (216, 109, 9),     # orange
}


def test_real_centers_classify_correctly_after_white_anchor():
    clf = ColorClassifier.from_white_anchor(REAL_WHITE)
    for face, rgb in REAL_CENTERS.items():
        assert clf.classify(rgb) == face, (
            f"{face} center {rgb} misclassified as {clf.classify(rgb)}"
        )


def test_real_yellow_not_confused_with_white_or_orange():
    # The specific failures reported on hardware: yellow->white and yellow->orange.
    clf = ColorClassifier.from_white_anchor(REAL_WHITE)
    assert clf.classify(REAL_CENTERS["D"]) == "D"
    assert clf.classify(REAL_CENTERS["D"]) not in ("U", "L")


# A second real session (stronger blue-cast lighting) where GREEN was being
# misread as WHITE. All six centers, including green, must classify correctly.
REAL_WHITE_2 = (160, 187, 237)
REAL_CENTERS_2 = {
    "U": (160, 187, 237),   # white
    "F": (108, 170, 148),   # green  <- was collapsing onto white
    "D": (195, 216, 150),   # yellow
    "B": (0, 103, 223),     # blue
    "R": (159, 70, 103),    # red
    "L": (188, 104, 47),    # orange
}


def test_real_green_not_misread_as_white():
    clf = ColorClassifier.from_white_anchor(REAL_WHITE_2)
    assert clf.classify(REAL_CENTERS_2["F"]) == "F"
    assert clf.classify(REAL_CENTERS_2["F"]) != "U"


def test_real_second_session_all_centers():
    clf = ColorClassifier.from_white_anchor(REAL_WHITE_2)
    for face, rgb in REAL_CENTERS_2.items():
        assert clf.classify(rgb) == face, (
            f"{face} center {rgb} misclassified as {clf.classify(rgb)}"
        )
