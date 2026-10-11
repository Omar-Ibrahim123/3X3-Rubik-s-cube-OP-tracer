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


# ---------------------------------------------------------------------------
# Color distance in a lighting-tolerant space.
#
# Pure Euclidean RGB distance is sensitive to brightness. We reduce that by
# comparing colors in a space that separates chroma (hue-ish direction) from
# overall brightness: normalize each color to unit length (direction) and also
# keep a mild brightness term. This handles shadows/highlights far better than
# raw RGB while needing no external libraries.
# ---------------------------------------------------------------------------

def _normalize(c: RGB) -> Tuple[float, float, float, float]:
    r, g, b = c
    mag = (r * r + g * g + b * b) ** 0.5
    if mag == 0:
        return (0.0, 0.0, 0.0, 0.0)
    # unit direction (chroma/hue) + normalized brightness (0..1)
    return (r / mag, g / mag, b / mag, mag / (255.0 * (3 ** 0.5)))


def color_distance(a: RGB, b: RGB) -> float:
    """Lighting-tolerant distance between two RGB colors.

    Direction (hue) is weighted more heavily than brightness, so a sticker in
    shadow still matches its calibrated reference.
    """
    na = _normalize(a)
    nb = _normalize(b)
    # Weight hue/direction strongly, brightness weakly.
    dir_d = sum((na[i] - nb[i]) ** 2 for i in range(3)) ** 0.5
    bright_d = abs(na[3] - nb[3])
    return dir_d + 0.25 * bright_d


@dataclass
class ColorClassifier:
    """Nearest-color classifier calibrated to one physical cube.

    `references` maps a face letter (U R F D L B) to the measured RGB color of
    that face's center sticker. Classification returns the face letter whose
    reference color is closest to the sample.
    """

    references: Dict[str, RGB] = field(default_factory=dict)

    @classmethod
    def from_centers(cls, centers: Dict[str, RGB]) -> "ColorClassifier":
        """Build a classifier from the six calibrated center colors."""
        missing = set(FACE_NAMES) - set(centers)
        if missing:
            raise ValueError(f"calibration missing faces: {sorted(missing)}")
        return cls(references=dict(centers))

    def classify(self, sample: RGB) -> str:
        """Return the face letter whose calibrated color is nearest `sample`."""
        if not self.references:
            raise RuntimeError("classifier is not calibrated")
        best_face = None
        best_dist = float("inf")
        for face, ref in self.references.items():
            d = color_distance(sample, ref)
            if d < best_dist:
                best_dist = d
                best_face = face
        return best_face

    def classify_with_confidence(self, sample: RGB) -> Tuple[str, float]:
        """Return (face, margin) where margin is the gap to the 2nd-nearest.

        A small margin means the sample is ambiguous (e.g. a borderline
        orange/red), which the capture layer can surface for a re-scan.
        """
        dists = sorted(
            (color_distance(sample, ref), face)
            for face, ref in self.references.items()
        )
        best_d, best_face = dists[0]
        second_d = dists[1][0] if len(dists) > 1 else float("inf")
        return best_face, (second_d - best_d)


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
