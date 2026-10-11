"""Tests for the one-window six-center calibration state machine."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.calibration import PaletteCalibrationTracker, median_rgb


U = (190.0, 205.0, 235.0)
R = (160.0, 45.0, 60.0)
F = (80.0, 155.0, 125.0)
D = (205.0, 215.0, 145.0)
L = (205.0, 115.0, 65.0)
B = (5.0, 100.0, 215.0)


def offer_many(tracker, color, count):
    accepted = []
    for _ in range(count):
        result = tracker.offer(color)
        if result:
            accepted.append(result)
    return accepted


def transition_by_loss(tracker):
    for _ in range(tracker.transition_frames):
        tracker.offer(None)


def test_median_rgb_rejects_single_frame_noise():
    assert median_rgb([(10, 20, 30), (12, 22, 32), (200, 200, 200)]) == (
        12,
        22,
        32,
    )


def test_same_held_face_cannot_fill_next_palette_label():
    tracker = PaletteCalibrationTracker(stable_frames=3, transition_frames=2)
    assert offer_many(tracker, U, 3) == ["U"]
    assert tracker.waiting_for_transition

    # Leaving the exact same U face in view for many frames must never calibrate R.
    assert offer_many(tracker, U, 20) == []
    assert set(tracker.palette) == {"U"}
    assert tracker.expected_face == "R"


def test_duplicate_color_rejected_after_transition():
    tracker = PaletteCalibrationTracker(stable_frames=3, transition_frames=2)
    offer_many(tracker, U, 3)
    transition_by_loss(tracker)

    # Even after a valid transition, U cannot be stored under R.
    assert offer_many(tracker, U, 10) == []
    assert tracker.expected_face == "R"
    assert "already calibrated" in tracker.last_message


def test_six_distinct_faces_calibrate_in_order():
    tracker = PaletteCalibrationTracker(stable_frames=3, transition_frames=2)
    colors = [U, R, F, D, L, B]
    accepted = []
    for index, color in enumerate(colors):
        accepted.extend(offer_many(tracker, color, 3))
        if index < len(colors) - 1:
            transition_by_loss(tracker)

    assert accepted == ["U", "R", "F", "D", "L", "B"]
    assert tracker.complete
    assert tracker.palette["U"] == U
    assert tracker.palette["L"] == L


def test_color_change_can_satisfy_transition_without_losing_detection():
    tracker = PaletteCalibrationTracker(stable_frames=3, transition_frames=2)
    offer_many(tracker, U, 3)

    # Rotating directly to red while the detector stays locked transitions state.
    offer_many(tracker, R, 2)
    assert not tracker.waiting_for_transition
    assert tracker.expected_face == "R"
    assert offer_many(tracker, R, 3) == ["R"]


def test_unstable_samples_do_not_calibrate():
    tracker = PaletteCalibrationTracker(stable_frames=4)
    for sample in (U, R, U, F, U, B):
        tracker.offer(sample)
    assert tracker.palette == {}


# --------------------------- SPACE-to-capture (commit) ---------------------------

def test_commit_records_each_face_in_order_on_one_call():
    # SPACE capture: a single commit() per face records it immediately, with no
    # steady-run or transition wait.
    tracker = PaletteCalibrationTracker()
    accepted = []
    for color in (U, R, F, D, L, B):
        accepted.append(tracker.commit(color))
    assert accepted == ["U", "R", "F", "D", "L", "B"]
    assert tracker.complete
    assert tracker.palette["R"] == R


def test_commit_refuses_duplicate_color():
    # A second face too close to an already-calibrated one is refused (this is
    # the red-captured-as-white / blue-captured-wrong guard).
    tracker = PaletteCalibrationTracker()
    assert tracker.commit(U) == "U"
    near_u = (U[0] + 3, U[1] - 2, U[2] + 1)  # ~a couple dE from U
    assert tracker.commit(near_u) is None
    assert tracker.expected_face == "R"
    assert "already-calibrated" in tracker.last_message


def test_commit_ignores_no_face():
    tracker = PaletteCalibrationTracker()
    assert tracker.commit(None) is None
    assert tracker.expected_face == "U"
    assert "no face" in tracker.last_message


def test_commit_stops_when_complete():
    tracker = PaletteCalibrationTracker()
    for color in (U, R, F, D, L, B):
        tracker.commit(color)
    assert tracker.complete
    # Any further commit is a no-op.
    assert tracker.commit((123, 45, 67)) is None
    assert len(tracker.palette) == 6
