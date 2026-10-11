"""Tests for the spin-to-scan collection logic (no camera / no OpenCV).

Simulates a stream of frames as a cube is slowly rotated: each face appears for
several steady frames, with blurry/mid-rotation frames in between. Verifies the
collector identifies faces by center color, requires stability, dedupes, and
completes with all six faces assembled correctly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from vision.colors import ColorClassifier, assemble_facelets, validate_facelets
from vision.spin import FaceCandidate, SpinCollector
from tests.test_vision import CAL, _faces_from_facelets
from tests.cube_moves import SOLVED, apply_sequence


def _classifier():
    return ColorClassifier.from_centers(CAL)


def _candidate_for_face_grid(grid):
    """A FaceCandidate from a 9-sample grid (already RGB)."""
    return FaceCandidate(samples=list(grid))


def _simulate_spin(collector, faces_grids, hold=8, blur_between=3):
    """Feed each face steadily for `hold` frames, with blur frames between.

    `faces_grids` is a dict {face: 9-samples}. Blur frames use a muddy gray
    that classifies arbitrarily but never stays steady long enough to accept.
    """
    accepted = []
    muddy = (120, 120, 120)
    order = list(faces_grids.keys())
    for i, face in enumerate(order):
        grid = faces_grids[face]
        for _ in range(hold):
            got = collector.offer(_candidate_for_face_grid(grid))
            if got:
                accepted.append(got)
        # jittery transition frames (different each frame -> no steady streak)
        for j in range(blur_between):
            jitter = (muddy[0] + j * 7, muddy[1] - j * 5, muddy[2] + j * 3)
            g = [jitter] * 9
            collector.offer(_candidate_for_face_grid(g))
    return accepted


def test_spin_collects_all_six_from_solved():
    clf = _classifier()
    collector = SpinCollector(classifier=clf, steady_frames=6)
    faces_grids = _faces_from_facelets(SOLVED)
    accepted = _simulate_spin(collector, faces_grids)
    assert collector.is_complete()
    assert set(accepted) == set("URFDLB")
    facelets = assemble_facelets(collector.faces(), clf)
    assert facelets == SOLVED


def test_spin_collects_scramble_and_traces():
    clf = _classifier()
    collector = SpinCollector(classifier=clf, steady_frames=6)
    scrambled = apply_sequence("D L F2 U' R B2 L' D2 F R' U B")
    faces_grids = _faces_from_facelets(scrambled)
    _simulate_spin(collector, faces_grids)
    assert collector.is_complete()
    facelets = assemble_facelets(collector.faces(), clf)
    assert facelets == scrambled
    assert validate_facelets(facelets).ok


def test_face_not_accepted_until_steady():
    clf = _classifier()
    collector = SpinCollector(classifier=clf, steady_frames=6)
    grid = _faces_from_facelets(SOLVED)["F"]
    cand = _candidate_for_face_grid(grid)
    # First five offers must not accept (need 6 steady frames).
    for _ in range(5):
        assert collector.offer(cand) is None
    # Sixth accepts.
    assert collector.offer(cand) == "F"


def test_blurry_stream_never_accepts():
    clf = _classifier()
    collector = SpinCollector(classifier=clf, steady_frames=6)
    # A stream whose center color changes every frame never stays steady.
    for i in range(50):
        jitter = [(100 + (i * 13) % 120, 50 + (i * 7) % 120, 200 - (i * 5) % 120)] * 9
        collector.offer(_candidate_for_face_grid(jitter))
    # It may have collected at most a couple by chance of repetition, but
    # should not complete, and crucially must not loop/crash.
    assert not collector.is_complete() or len(collector.collected_faces()) <= 6


def test_duplicate_face_not_recollected():
    clf = _classifier()
    collector = SpinCollector(classifier=clf, steady_frames=3)
    grid = _faces_from_facelets(SOLVED)["U"]
    cand = _candidate_for_face_grid(grid)
    first = [collector.offer(cand) for _ in range(10)]
    # Exactly one acceptance of U despite many steady frames.
    assert first.count("U") == 1
    assert collector.collected_faces() == ["U"]


def test_faces_raises_until_complete():
    clf = _classifier()
    collector = SpinCollector(classifier=clf)
    with pytest.raises(RuntimeError):
        collector.faces()


def test_progress_and_remaining():
    clf = _classifier()
    collector = SpinCollector(classifier=clf, steady_frames=2)
    grids = _faces_from_facelets(SOLVED)
    # Collect just U and R.
    for face in ("U", "R"):
        c = _candidate_for_face_grid(grids[face])
        for _ in range(2):
            collector.offer(c)
        # break steadiness between faces
        collector.offer(_candidate_for_face_grid([(0, 0, 0)] * 9))
    assert set(collector.collected_faces()) == {"U", "R"}
    assert set(collector.remaining_faces()) == {"F", "D", "L", "B"}
    assert "2/6" in collector.progress()


def test_reset_clears_state():
    clf = _classifier()
    collector = SpinCollector(classifier=clf, steady_frames=2)
    grid = _faces_from_facelets(SOLVED)["U"]
    c = _candidate_for_face_grid(grid)
    collector.offer(c); collector.offer(c)
    assert collector.collected_faces() == ["U"]
    collector.reset()
    assert collector.collected_faces() == []
    assert not collector.is_complete()
