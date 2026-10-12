"""Tests for the scan-verification helpers (pure, no camera / no OpenCV)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from vision.colors import ColorClassifier
from vision.confirm import (
    build_report,
    low_confidence_stickers,
    net_lines,
)
from tests.cube_moves import SOLVED, apply_sequence
from tests.test_vision import CAL, NOISY, _faces_from_facelets


# --------------------------- net rendering ---------------------------

def test_net_plain_has_cross_layout():
    lines = net_lines(SOLVED, color=False)
    assert len(lines) == 9  # 3 (U) + 3 (L F R B) + 3 (D)
    # U band: three white letters, indented.
    assert lines[0].split() == ["W", "W", "W"]
    # Middle band row contains all four side faces (12 cells).
    mid = lines[3]
    assert mid.count("O") == 3  # L = orange
    assert mid.count("G") == 3  # F = green
    assert mid.count("R") == 3  # R = red
    assert mid.count("B") == 3  # B = blue
    # D band: yellow.
    assert lines[6].split() == ["Y", "Y", "Y"]


def test_net_color_wraps_with_ansi():
    colored = net_lines(SOLVED, color=True)
    assert any("\x1b[" in line for line in colored)


def test_net_rejects_bad_length():
    with pytest.raises(ValueError):
        net_lines("UUU")


# --------------------------- low-confidence detection ---------------------------

def test_clean_samples_have_no_low_confidence():
    clf = ColorClassifier.from_centers(CAL)
    faces = _faces_from_facelets(SOLVED, cal=NOISY)
    assert low_confidence_stickers(faces, clf) == []


def test_ambiguous_sticker_is_flagged():
    clf = ColorClassifier.from_centers(CAL)
    faces = _faces_from_facelets(SOLVED, cal=NOISY)
    # Replace one sticker with a colour halfway between red (R) and orange (L)
    # -- the classic confusable pair -- so its nearest/second-nearest gap is
    # tiny and it must be flagged as low confidence.
    midpoint = tuple((CAL["R"][i] + CAL["L"][i]) / 2 for i in range(3))
    faces["F"] = list(faces["F"])
    faces["F"][0] = midpoint
    flagged = low_confidence_stickers(faces, clf, min_margin=4.0)
    assert any(s.face == "F" and s.index == 0 for s in flagged)


# --------------------------- report ---------------------------

def test_report_on_solvable_scramble():
    clf = ColorClassifier.from_centers(CAL)
    scramble = apply_sequence("R U R' U' F2 L D B'")
    faces = _faces_from_facelets(scramble, cal=NOISY)
    report = build_report(scramble, faces, clf)
    assert report.solvable
    assert report.facelets == scramble
    text = "\n".join(report.summary_lines())
    assert "solvable" in text


def test_report_flags_unsolvable_string():
    clf = ColorClassifier.from_centers(CAL)
    # A structurally-valid-but-unsolvable cube: flip one edge of a scramble.
    scr = list(apply_sequence("R U R' F2 L"))
    scr[1], scr[46] = scr[46], scr[1]
    bad = "".join(scr)
    faces = _faces_from_facelets(bad, cal=NOISY)
    report = build_report(bad, faces, clf)
    assert not report.solvable
    assert "NOT a solvable cube" in "\n".join(report.summary_lines())
