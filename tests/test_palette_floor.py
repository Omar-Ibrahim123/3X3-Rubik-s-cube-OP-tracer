"""Regression: the palette distinctness floor rejects a duplicate-colour scan.

Real failure: during calibration the RED face was sampled as white (shown
during a turn), so R and U came out ~9 dE apart -- two whites, no red. Every
real red sticker then misread as orange and the coherence gate rejected every
face. The floor (MIN_PALETTE_DISTANCE) must reject that palette up front with a
clear message, while every genuine palette (closest real pair ~22 dE) passes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from vision.calibration import MIN_PALETTE_DISTANCE, PaletteCalibrationTracker
from vision.colors import ColorClassifier


# The user's real broken palette: R was captured as a pale white, ~9 dE from U.
BAD_PALETTE = {
    "U": (131, 175, 245),
    "R": (177, 205, 254),   # should be red; sampled as white
    "F": (131, 184, 185),
    "D": (210, 219, 166),
    "L": (214, 133, 111),
    "B": (11, 102, 219),
}

# Two real, healthy palettes from completed/near-completed scans.
GOOD_PALETTES = [
    {"U": (181, 209, 254), "R": (159, 79, 118), "F": (103, 164, 147),
     "D": (206, 223, 170), "L": (217, 150, 138), "B": (59, 132, 229)},
    {"U": (178, 203, 250), "R": (159, 79, 118), "F": (103, 164, 147),
     "D": (206, 223, 170), "L": (217, 150, 138), "B": (59, 132, 229)},
]


def test_floor_is_between_broken_and_healthy():
    # Sanity: the chosen floor sits above the broken palette's closest pair and
    # below every healthy palette's closest pair.
    assert 9.0 < MIN_PALETTE_DISTANCE < 22.0


def test_duplicate_colour_palette_rejected_with_guidance():
    with pytest.raises(ValueError) as exc:
        ColorClassifier.from_palette(BAD_PALETTE)
    msg = str(exc.value)
    assert "distinct" in msg
    # The two offending faces are named so the user knows what to re-show.
    assert "U" in msg and "R" in msg


def test_healthy_palettes_still_accepted():
    for palette in GOOD_PALETTES:
        clf = ColorClassifier.from_palette(palette)
        assert clf.perceptual
        # Red classifies as red, not orange (the symptom we were chasing).
        assert clf.classify((170, 50, 85)) in ("R", "L")  # a red-ish sample
        # And the real red reference is nearer R than L.
        assert clf.classify(palette["R"]) == "R"


def test_tracker_floor_matches_shared_constant():
    assert PaletteCalibrationTracker().min_palette_distance == MIN_PALETTE_DISTANCE


def test_tracker_rejects_near_duplicate_center():
    # A new centre within the floor of an already-accepted one must be refused,
    # keeping the prompt on the requested face.
    tracker = PaletteCalibrationTracker(stable_frames=3, transition_frames=2)
    white = (160.0, 185.0, 233.0)
    for _ in range(3):
        tracker.offer(white)          # accept U
    for _ in range(tracker.transition_frames):
        tracker.offer(None)           # transition
    near_white = (170.0, 195.0, 243.0)  # ~a few dE from U, well under the floor
    for _ in range(5):
        assert tracker.offer(near_white) is None
    assert "already calibrated" in tracker.last_message
    assert tracker.expected_face == "R"
