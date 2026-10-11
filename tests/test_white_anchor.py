"""Tests for single-white-sample calibration (from_white_anchor).

Verifies the low-friction flow: showing only the white face and relying on the
built-in BOY/Western scheme still classifies all six colors correctly, both
under neutral light and under tinted/dim light (the reason we anchor to the
measured white rather than hard-coding references).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from vision.colors import STANDARD_SCHEME, ColorClassifier, color_distance


def _apply_light(scheme, gains):
    """Simulate a camera seeing the scheme under a per-channel lighting gain."""
    return {
        face: tuple(min(255.0, c * g) for c, g in zip(rgb, gains))
        for face, rgb in scheme.items()
    }


def test_canonical_white_reproduces_scheme():
    clf = ColorClassifier.from_white_anchor(STANDARD_SCHEME["U"])
    # Each scheme color classifies to its own face.
    for face, rgb in STANDARD_SCHEME.items():
        assert clf.classify(rgb) == face


def test_classifies_all_six_under_neutral_light():
    clf = ColorClassifier.from_white_anchor(STANDARD_SCHEME["U"])
    # Slightly noisy versions of each scheme color.
    noisy = {
        "U": (240, 248, 238), "D": (250, 208, 15), "F": (10, 150, 80),
        "B": (5, 66, 165), "R": (190, 35, 50), "L": (248, 95, 12),
    }
    for face, rgb in noisy.items():
        assert clf.classify(rgb) == face, f"{face} misclassified under neutral"


def test_warm_tinted_light_still_classifies():
    # Warm/incandescent light: boosts red, cuts blue. The white sample is
    # tinted the same way, so anchoring should compensate.
    gains = (1.15, 1.0, 0.8)
    lit = _apply_light(STANDARD_SCHEME, gains)
    clf = ColorClassifier.from_white_anchor(lit["U"])
    for face, rgb in lit.items():
        assert clf.classify(rgb) == face, f"{face} misclassified under warm light"


def test_dim_light_still_classifies():
    # Everything darker (under-exposed); white comes in gray-ish.
    gains = (0.55, 0.55, 0.55)
    lit = _apply_light(STANDARD_SCHEME, gains)
    clf = ColorClassifier.from_white_anchor(lit["U"])
    for face, rgb in lit.items():
        assert clf.classify(rgb) == face, f"{face} misclassified under dim light"


def test_orange_vs_red_under_white_anchor():
    # The hard pair must still separate with only a white sample.
    clf = ColorClassifier.from_white_anchor(STANDARD_SCHEME["U"])
    assert clf.classify((255, 90, 5)) == "L"    # orange
    assert clf.classify((195, 28, 42)) == "R"   # red


def test_gain_clamped_against_bad_white():
    # A near-black "white" (lens covered) must not blow up references; gains
    # are clamped, so the classifier is still built and usable.
    clf = ColorClassifier.from_white_anchor((5.0, 5.0, 5.0))
    assert set(clf.references) == set(STANDARD_SCHEME)
    for rgb in clf.references.values():
        assert all(0.0 <= c <= 255.0 for c in rgb)
