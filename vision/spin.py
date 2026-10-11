"""Spin-to-scan collection logic (pure, no camera / no OpenCV).

The low-friction capture flow: instead of pressing a key per face, you slowly
rotate the cube in front of the camera and the app auto-detects each face.

This module holds the TESTABLE decision logic for that flow. The camera layer
feeds it a stream of "candidate" readings (each = 9 color samples for a face
currently centered in the overlay). The collector:

  * identifies which physical face a candidate is by its CENTER sticker color
    (centers never move, so center color == face identity under a fixed color
    scheme),
  * only accepts a candidate once it has been held STEADY for a few frames
    (so we capture a square-on, non-blurry face, not a mid-rotation smear),
  * keeps one accepted reading per distinct face, and
  * reports when all six faces have been collected.

Face identity here is the CENTER's face letter (U R F D L B) as classified by
the calibrated ColorClassifier. Because we calibrate on the six centers first,
"which face" is just "which calibrated color is this center nearest".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .colors import FACE_NAMES, RGB, ColorClassifier, color_distance


@dataclass
class FaceCandidate:
    """One frame's reading of the face currently in view: 9 RGB samples."""

    samples: List[RGB]

    def center(self) -> RGB:
        return self.samples[4]


# How steady a face must be before we accept it, and how close two frames'
# center colors must be to count as "the same face held still".
DEFAULT_STEADY_FRAMES = 6
DEFAULT_STEADY_DIST = 0.08


@dataclass
class SpinCollector:
    """Collects one steady reading per face as the cube is rotated.

    Feed candidates with `offer()` each frame. When `is_complete()` is True,
    call `faces()` to get the {face: 9-samples} dict for assembly.
    """

    classifier: ColorClassifier
    steady_frames: int = DEFAULT_STEADY_FRAMES
    steady_dist: float = DEFAULT_STEADY_DIST

    _collected: Dict[str, List[RGB]] = field(default_factory=dict)
    _streak_face: Optional[str] = None
    _streak_center: Optional[RGB] = None
    _streak_count: int = 0
    _last_accepted: Optional[str] = None

    # --- query ---

    def collected_faces(self) -> List[str]:
        return [f for f in FACE_NAMES if f in self._collected]

    def remaining_faces(self) -> List[str]:
        return [f for f in FACE_NAMES if f not in self._collected]

    def is_complete(self) -> bool:
        return len(self._collected) == 6

    def faces(self) -> Dict[str, List[RGB]]:
        if not self.is_complete():
            raise RuntimeError(
                f"only {len(self._collected)}/6 faces collected: "
                f"missing {self.remaining_faces()}"
            )
        return dict(self._collected)

    def progress(self) -> str:
        got = "".join(self.collected_faces()) or "-"
        return f"{len(self._collected)}/6 faces [{got}] steady={self._streak_count}"

    # --- feed ---

    def offer(self, candidate: FaceCandidate) -> Optional[str]:
        """Offer the current frame's face reading.

        Returns the face letter if this call completed a steady capture of a
        NEW face (i.e. the moment a face is accepted); otherwise None.
        """
        center = candidate.center()
        face = self.classifier.classify(center)

        # Is this the same face, held roughly as still, as the running streak?
        same_as_streak = (
            self._streak_face == face
            and self._streak_center is not None
            and color_distance(center, self._streak_center) <= self.steady_dist
        )
        if same_as_streak:
            self._streak_count += 1
        else:
            # New face in view (or it moved too much): restart the streak.
            self._streak_face = face
            self._streak_center = center
            self._streak_count = 1

        # Accept only when steady long enough AND not already collected.
        if (
            self._streak_count >= self.steady_frames
            and face not in self._collected
        ):
            self._collected[face] = list(candidate.samples)
            self._last_accepted = face
            # Reset the streak so re-showing the same face doesn't re-trigger
            # and so moving to the next face starts fresh.
            self._streak_count = 0
            self._streak_face = None
            self._streak_center = None
            return face
        return None

    def reset(self) -> None:
        self._collected.clear()
        self._streak_face = None
        self._streak_center = None
        self._streak_count = 0
        self._last_accepted = None
