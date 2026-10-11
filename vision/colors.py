"""Pure color-classification logic for cube scanning (no camera, no OpenCV).

This module contains everything that can be unit-tested without hardware:
  * a nearest-color classifier that learns YOUR cube's actual colors from a
    short calibration step (robust to lighting and to orange-vs-red /
    white-vs-yellow confusion), and
  * assembly of per-face color grids into a 54-char facelet string, plus
    validation of the resulting state.

The OpenCV capture layer (vision/capture.py) feeds this module raw BGR/RGB
samples; this module decides what color each sticker is. Keeping the decision
logic here means the fiddly part is testable and the camera glue stays thin.

Color space: we classify in a simple, lighting-tolerant way using calibrated
reference colors and nearest-neighbor in a normalized color space. Samples and
references are plain (R, G, B) tuples, 0..255, so this module has NO dependency
on numpy or OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

# The six cube face names, in the Kociemba facelet order used project-wide.
FACE_NAMES = ("U", "R", "F", "D", "L", "B")

RGB = Tuple[float, float, float]

# Built-in reference colors for the standard BOY/Western color scheme, with the
# cube held white-up / green-front (so: U=white, D=yellow, F=green, B=blue,
# R=red, L=orange). Used by from_white_anchor() so the user only has to show
# the white face: these references are rescaled to the user's lighting using
# their measured white. Values are nominal sRGB-ish sticker colors.
STANDARD_SCHEME: Dict[str, RGB] = {
    "U": (240, 240, 240),   # white  (neutral, low saturation)
    "D": (235, 205, 60),    # yellow (warm/pale: separates from orange by sat)
    "F": (0, 150, 70),      # green
    "B": (0, 90, 200),      # blue
    "R": (190, 30, 40),     # red
    "L": (235, 95, 5),      # orange (vivid)
}


# ---------------------------------------------------------------------------
# Color comparison in HSV, after white-balancing.
#
# Real webcam/lighting findings (from debug captures): lighting adds a strong
# color cast (e.g. blue), which shifts every sticker and makes white/yellow and
# red/orange collide in raw RGB. Two fixes, in order:
#
#   1. White-balance: using the measured WHITE sample, scale each channel so
#      white becomes neutral. This removes the cast from EVERY sample.
#   2. Compare in HSV: hue+saturation separate the six cube colors far better
#      than RGB. White = very low saturation; yellow/orange/red differ mainly
#      in hue; this cleanly splits the pairs that confuse RGB distance.
# ---------------------------------------------------------------------------

def rgb_to_hsv(c: RGB) -> Tuple[float, float, float]:
    """Convert RGB (0..255) to HSV with H in [0,360), S,V in [0,1]."""
    r, g, b = (max(0.0, min(255.0, x)) / 255.0 for x in c)
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn
    if diff == 0:
        h = 0.0
    elif mx == r:
        h = (60 * ((g - b) / diff)) % 360
    elif mx == g:
        h = 60 * ((b - r) / diff) + 120
    else:
        h = 60 * ((r - g) / diff) + 240
    s = 0.0 if mx == 0 else diff / mx
    v = mx
    return (h, s, v)


def _hue_dist(h1: float, h2: float) -> float:
    """Circular hue distance in degrees, 0..180."""
    d = abs(h1 - h2) % 360
    return min(d, 360 - d)


def color_distance(a: RGB, b: RGB) -> float:
    """Perceptual-ish distance between two RGB colors via HSV.

    White (low saturation) is handled specially: if either color is nearly
    unsaturated, saturation dominates (so white never matches a vivid color on
    hue alone). Otherwise hue dominates, with a mild saturation/value term.
    """
    h1, s1, v1 = rgb_to_hsv(a)
    h2, s2, v2 = rgb_to_hsv(b)
    sat_d = abs(s1 - s2)
    val_d = abs(v1 - v2)
    # Hue only matters when both colors are saturated enough to have a hue.
    hue_weight = min(s1, s2)
    hue_d = _hue_dist(h1, h2) / 180.0
    return hue_weight * hue_d * 2.0 + sat_d * 1.5 + val_d * 0.3


@dataclass
class ColorClassifier:
    """Nearest-color classifier calibrated to one physical cube.

    `references` maps a face letter (U R F D L B) to the measured RGB color of
    that face's center sticker. Classification returns the face letter whose
    reference color is closest to the sample.
    """

    references: Dict[str, RGB] = field(default_factory=dict)
    # Per-channel white-balance gains applied to every sample before comparing.
    # Default (1,1,1) = no correction. from_white_anchor() sets these from the
    # measured white so the user's lighting cast is removed.
    wb_gains: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    @classmethod
    def from_centers(cls, centers: Dict[str, RGB]) -> "ColorClassifier":
        """Build a classifier from the six calibrated center colors.

        White-balance is derived from the measured white (U) center so the
        same cast correction applies as in the single-white flow.
        """
        missing = set(FACE_NAMES) - set(centers)
        if missing:
            raise ValueError(f"calibration missing faces: {sorted(missing)}")
        gains = _wb_gains_from_white(centers["U"])
        return cls(references=dict(centers), wb_gains=gains)

    @classmethod
    def from_white_anchor(cls, white_sample: RGB) -> "ColorClassifier":
        """Build a classifier from a single measured WHITE sample.

        Low-friction calibration: the user shows only the white (U) face. We
        compute white-balance gains that neutralize the measured white (removing
        the lighting cast), then classify every sample in that corrected space
        against the canonical BOY/Western references. Correcting the cast on the
        INPUT (rather than tinting the references) is what makes white/yellow and
        red/orange separate reliably under colored lighting.

        Assumes the standard (BOY/Western) color scheme. If a cube uses a
        different scheme, use from_centers() (the manual multi-center flow).
        """
        gains = _wb_gains_from_white(white_sample)
        return cls(references=dict(STANDARD_SCHEME), wb_gains=gains)

    def _balance(self, sample: RGB) -> RGB:
        g = self.wb_gains
        return tuple(max(0.0, min(255.0, sample[i] * g[i])) for i in range(3))

    def classify(self, sample: RGB) -> str:
        """Return the face letter whose reference is nearest the WB'd sample."""
        if not self.references:
            raise RuntimeError("classifier is not calibrated")
        s = self._balance(sample)
        best_face = None
        best_dist = float("inf")
        for face, ref in self.references.items():
            d = color_distance(s, ref)
            if d < best_dist:
                best_dist = d
                best_face = face
        return best_face

    def classify_with_confidence(self, sample: RGB) -> Tuple[str, float]:
        """Return (face, margin) where margin is the gap to the 2nd-nearest.

        A small margin means the sample is ambiguous (e.g. a borderline
        orange/red), which the capture layer can surface for a re-scan.
        """
        s = self._balance(sample)
        dists = sorted(
            (color_distance(s, ref), face)
            for face, ref in self.references.items()
        )
        best_d, best_face = dists[0]
        second_d = dists[1][0] if len(dists) > 1 else float("inf")
        return best_face, (second_d - best_d)


def _wb_gains_from_white(white: RGB) -> Tuple[float, float, float]:
    """Per-channel gains that make a measured white sample neutral gray.

    Scales each channel toward the sample's own average brightness, so the
    corrected white has equal R=G=B. Gains are clamped so a bad white reading
    (e.g. lens covered) can't blow up subsequent samples.
    """
    avg = sum(white) / 3.0
    gains = []
    for i in range(3):
        c = white[i]
        g = (avg / c) if c else 1.0
        gains.append(max(0.3, min(g, 3.0)))
    return tuple(gains)


# ---------------------------------------------------------------------------
# Facelet-string assembly.
# ---------------------------------------------------------------------------

def face_grid_to_letters(
    grid: Sequence[RGB], classifier: ColorClassifier
) -> List[str]:
    """Classify a face's 9 sticker samples (reading order) into face letters."""
    if len(grid) != 9:
        raise ValueError(f"a face grid must have 9 samples, got {len(grid)}")
    return [classifier.classify(s) for s in grid]


def assemble_facelets(
    faces: Dict[str, Sequence[RGB]], classifier: ColorClassifier
) -> str:
    """Assemble the 54-char facelet string from six classified face grids.

    `faces` maps each face letter (U R F D L B) to its 9 sticker samples in
    reading order. Returns the facelet string in Kociemba face order.
    """
    missing = set(FACE_NAMES) - set(faces)
    if missing:
        raise ValueError(f"missing scanned faces: {sorted(missing)}")
    out: List[str] = []
    for face in FACE_NAMES:
        out.extend(face_grid_to_letters(faces[face], classifier))
    return "".join(out)


# ---------------------------------------------------------------------------
# State validation (catch misreads before they reach the solver).
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)


def validate_facelets(facelets: str) -> ValidationResult:
    """Check structural validity of a scanned facelet string.

    Catches the common scan failures: wrong length, a color appearing more or
    fewer than 9 times, and centers not being the six distinct faces. This is
    a cheap sanity gate; it does not prove the cube is physically solvable
    (that's a deeper permutation/parity check the solver layer can add later),
    but it catches the vast majority of camera misreads.
    """
    errors: List[str] = []
    if len(facelets) != 54:
        return ValidationResult(False, [f"length {len(facelets)} != 54"])

    # Each of the six colors must appear exactly 9 times.
    counts: Dict[str, int] = {}
    for ch in facelets:
        counts[ch] = counts.get(ch, 0) + 1
    for face in FACE_NAMES:
        n = counts.get(face, 0)
        if n != 9:
            errors.append(f"color {face!r} appears {n} times (expected 9)")
    extra = set(counts) - set(FACE_NAMES)
    if extra:
        errors.append(f"unexpected colors: {sorted(extra)}")

    # Centers (index 4 of each face) must be the six distinct faces in order.
    centers = [facelets[i * 9 + 4] for i in range(6)]
    if sorted(centers) != sorted(FACE_NAMES):
        errors.append(f"centers are not the six faces: {centers}")

    return ValidationResult(not errors, errors)
