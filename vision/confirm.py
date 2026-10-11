"""Human verification of a scanned cube (pure logic, no camera / no OpenCV).

A scan that assembles into a *solvable* cube is not proof it read YOUR cube: a
couple of borderline stickers can misclassify in a way that still happens to
form a valid cube, giving a confident trace of the wrong state. So before the
solve, the user must be able to (a) SEE the detected cube and compare it to the
one in their hand, and (b) know which stickers were read with low confidence.

This module provides the testable pieces:
  * a coloured ASCII/ANSI net of the facelet string (works in any terminal,
    even headless -- the OpenCV window is an optional nicety on top);
  * a solvability / parity summary line; and
  * low-confidence sticker analysis from the per-sample classification margins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .colors import FACE_NAMES, RGB, ColorClassifier

# Face letter -> (human colour name, ANSI background code). The ANSI codes are
# 256-colour backgrounds chosen to look like the sticker colours in a terminal.
_FACE_INFO = {
    "U": ("W", 231),   # white
    "R": ("R", 196),   # red
    "F": ("G", 46),    # green
    "D": ("Y", 226),   # yellow
    "L": ("O", 208),   # orange
    "B": ("B", 21),    # blue
}


def _cell(face: str, color: bool) -> str:
    """Render one sticker cell: a coloured block (ANSI) or a plain letter."""
    letter, code = _FACE_INFO.get(face, ("?", 240))
    if not color:
        return f" {letter} "
    # Black text on the sticker-coloured background.
    return f"\x1b[48;5;{code}m\x1b[38;5;16m {letter} \x1b[0m"


def net_lines(facelets: str, color: bool = True) -> List[str]:
    """Return the cube as an unfolded cross net, one string per text row.

        U
      L F R B
        D

    `color` toggles ANSI colour (set False for plain text / tests). Each face
    is read in reading order (row-major); the net lets the user compare the
    detected cube to the physical one at a glance.
    """
    if len(facelets) != 54:
        raise ValueError(f"facelets must be 54 chars, got {len(facelets)}")
    f = {name: facelets[i * 9:(i + 1) * 9] for i, name in enumerate(FACE_NAMES)}

    def row(face: str, r: int) -> str:
        return "".join(_cell(f[face][r * 3 + c], color) for c in range(3))

    pad = "   " * 3  # width of one face block
    lines: List[str] = []
    for r in range(3):                      # U on top
        lines.append(pad + row("U", r))
    for r in range(3):                      # L F R B middle band
        lines.append(row("L", r) + row("F", r) + row("R", r) + row("B", r))
    for r in range(3):                      # D on bottom
        lines.append(pad + row("D", r))
    return lines


@dataclass
class LowConfidenceSticker:
    face: str          # which captured face (U R F D L B)
    index: int         # 0..8 reading-order position on that face
    rgb: RGB
    assigned: str      # colour it classified as
    margin: float      # dE gap to the 2nd-nearest colour (smaller = shakier)


def low_confidence_stickers(
    faces: Dict[str, Sequence[RGB]],
    classifier: ColorClassifier,
    min_margin: float = 4.0,
) -> List[LowConfidenceSticker]:
    """Find stickers whose classification was a close call.

    `faces` is the captured {face: 9 RGB samples} (BEFORE orientation, so the
    samples still line up with their captured positions). A sticker is "low
    confidence" when the gap between its nearest and second-nearest calibrated
    colour is under `min_margin` dE -- i.e. it could plausibly be either colour.
    These are exactly the stickers a user should eyeball on the net.
    """
    out: List[LowConfidenceSticker] = []
    for face in FACE_NAMES:
        samples = faces.get(face)
        if not samples:
            continue
        for i, sample in enumerate(samples):
            assigned, margin = classifier.classify_with_confidence(sample)
            if margin < min_margin:
                out.append(
                    LowConfidenceSticker(
                        face=face, index=i, rgb=tuple(sample),
                        assigned=assigned, margin=margin,
                    )
                )
    return out


@dataclass
class ScanReport:
    """Everything the user needs to judge a scan before accepting it."""

    facelets: str
    solvable: bool
    parity: bool                     # OP parity flag (odd edge-target count)
    low_confidence: List[LowConfidenceSticker] = field(default_factory=list)

    def summary_lines(self) -> List[str]:
        lines = []
        lines.append(
            "cube check: "
            + ("VALID & solvable" if self.solvable else "NOT a solvable cube")
        )
        lines.append(f"OP parity: {'yes' if self.parity else 'no'}")
        if self.low_confidence:
            lines.append(
                f"{len(self.low_confidence)} sticker(s) read with LOW confidence "
                "-- check these on the net:"
            )
            for s in self.low_confidence:
                lines.append(
                    f"  face {s.face} pos {s.index}: "
                    f"({s.rgb[0]:.0f},{s.rgb[1]:.0f},{s.rgb[2]:.0f}) "
                    f"-> {s.assigned} (margin {s.margin:.1f} dE)"
                )
        else:
            lines.append("all stickers read with clear confidence")
        return lines


def build_report(
    facelets: str,
    faces: Dict[str, Sequence[RGB]],
    classifier: ColorClassifier,
    min_margin: float = 4.0,
) -> ScanReport:
    """Assemble a ScanReport: solvability, parity, and low-confidence stickers.

    `facelets` is the FINAL (post-orientation) string; `faces` are the captured
    RGB samples used for the confidence analysis.
    """
    from cube.orientation import is_solvable

    solvable = is_solvable(facelets)
    parity = False
    if solvable:
        from cube.state import CubeState
        from solver import trace

        parity = trace(CubeState.from_facelets(facelets)).parity

    return ScanReport(
        facelets=facelets,
        solvable=solvable,
        parity=parity,
        low_confidence=low_confidence_stickers(faces, classifier, min_margin),
    )
