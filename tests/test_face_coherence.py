"""Regression tests for the face-coherence sanity gate (pure, no camera).

These lock in a real failure captured from the user's webcam: the "auto" flow
calibrated six centers (two of which, R and F, came out nearly identical --
distance 7.84) and then "captured" faces whose nine stickers actually came from
FOUR OR FIVE different real faces. The old collector accepted them because it
judged a face only by its center sticker. The coherence gate must reject these
mixed-region readings, while still passing every legal face (even a scrambled
one, which can legitimately show up to six colors).

Data below is verbatim from the user's --debug output.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.colors import ColorClassifier
from vision.spin import (
    FaceCandidate,
    SpinCollector,
    face_coherence,
)


# --- Real calibrated palette from the user's session (--debug) ---------------
# Note R and F are nearly identical here (distance 7.84) -- a calibration that
# barely cleared the distinctness threshold. We build the classifier directly
# (bypassing from_palette's >7.0 guard) to reproduce the exact runtime state.
REAL_PALETTE = {
    "U": (160, 185, 233),
    "R": (159, 57, 83),
    "F": (161, 80, 112),
    "D": (199, 212, 147),
    "L": (0, 71, 179),
    "B": (120, 162, 223),
}


# --- Real "captured" face grids from the same session ------------------------
# Each is nine (r,g,b) samples in reading order. By eye these clearly contain
# stickers from several different faces -- exactly the corruption we must stop.
REAL_MIXED_GRIDS = {
    "B": [
        (176, 198, 241), (182, 205, 246), (203, 120, 91),
        (80, 138, 117), (120, 161, 222), (181, 206, 247),
        (182, 192, 124), (120, 160, 222), (198, 215, 172),
    ],
    "U": [
        (185, 207, 250), (188, 211, 253), (207, 122, 96),
        (135, 173, 229), (174, 201, 243), (186, 211, 252),
        (141, 24, 36), (187, 196, 129), (192, 203, 137),
    ],
    "R": [
        (84, 137, 122), (135, 19, 31), (141, 45, 61),
        (158, 174, 214), (125, 17, 21), (178, 78, 26),
        (162, 163, 87), (164, 167, 91), (171, 74, 17),
    ],
    "D": [
        (29, 99, 198), (12, 87, 191), (90, 145, 122),
        (189, 84, 28), (176, 182, 115), (142, 17, 23),
        (176, 178, 105), (176, 181, 109), (0, 67, 177),
    ],
    "L": [
        (99, 155, 143), (22, 102, 205), (202, 214, 150),
        (91, 148, 129), (16, 103, 200), (153, 46, 65),
        (179, 201, 239), (99, 157, 143), (184, 209, 248),
    ],
}


def _real_classifier():
    """The exact runtime classifier (perceptual palette), guard bypassed."""
    return ColorClassifier(references=dict(REAL_PALETTE), perceptual=True)


def test_real_mixed_grids_are_flagged_incoherent():
    """Every real mis-detected grid must fail the coherence check."""
    clf = _real_classifier()
    for label, grid in REAL_MIXED_GRIDS.items():
        result = face_coherence(grid, clf)
        assert not result.ok, (
            f"grid labelled {label} should be rejected as a mixed region, "
            f"but passed: {result}"
        )
        assert result.reason  # a human-readable explanation is always given


def test_collector_refuses_to_collect_the_mixed_grids():
    """Fed only the real mixed grids, the collector must collect NOTHING.

    Previously these were accepted (judged by center only), corrupting the
    scan. With the gate on, offering any of them many times collects no faces
    and records a rejection reason.
    """
    clf = _real_classifier()
    collector = SpinCollector(classifier=clf, steady_frames=6)
    for grid in REAL_MIXED_GRIDS.values():
        cand = FaceCandidate(samples=grid)
        for _ in range(10):
            assert collector.offer(cand) is None
    assert collector.collected_faces() == []
    assert collector.last_rejection != ""


def test_gate_still_passes_a_clean_scrambled_face():
    """A legitimate face (even showing several colors) must NOT be rejected.

    Build a face whose nine stickers are the six real calibrated colors (a
    plausible scrambled face) with a clean, confidently-classified center.
    """
    clf = _real_classifier()
    # Center is a clean red; the other eight are the real palette colors, so up
    # to six distinct colors appear -- all legal for one scrambled face.
    center_red = (135, 19, 31)
    grid = [
        REAL_PALETTE["U"], REAL_PALETTE["L"], REAL_PALETTE["B"],
        REAL_PALETTE["D"], center_red, REAL_PALETTE["U"],
        REAL_PALETTE["B"], REAL_PALETTE["L"], REAL_PALETTE["D"],
    ]
    result = face_coherence(grid, clf)
    assert result.ok, f"a legal scrambled face was rejected: {result}"


def test_wrong_sample_count_is_incoherent():
    clf = _real_classifier()
    assert not face_coherence([(0, 0, 0)] * 8, clf).ok
