"""Static cube scanner: orchestrates calibrate -> scan -> classify -> validate.

`FaceletSource` is the single contract the rest of the app depends on: it
yields a 54-character facelet string. `StaticScanner` implements it on top of
the OpenCV capture layer (vision/capture.py) and the pure color logic
(vision/colors.py).

A future `LiveTurnDetector` can implement the same `FaceletSource` contract
(plus a turn-signal callback feeding tps.TPSCounter.record_turn) without any
downstream changes — the seam that was designed in Phase 1.

The pure-logic core still imports this module without OpenCV installed: OpenCV
is only touched when you actually call read_facelets() / run a scan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence

from .colors import (
    FACE_NAMES,
    RGB,
    ColorClassifier,
    assemble_facelets,
    validate_facelets,
)


class FaceletSource(ABC):
    """Anything that can produce a 54-char facelet string."""

    @abstractmethod
    def read_facelets(self) -> str:
        """Return a 54-character facelet string (Kociemba face order)."""
        raise NotImplementedError


class StaticScanner(FaceletSource):
    """Scan a stationary cube face-by-face via a webcam.

    Flow when read_facelets() is called:
      1. Calibrate: show each center sticker once so classification is tuned
         to your cube under your lighting.
      2. Scan: show all six faces with the 3x3 overlay; sample 9 colors each.
      3. Classify + assemble into a 54-char facelet string.
      4. Validate; if invalid, report the errors and (unless disabled) let you
         re-scan rather than feeding garbage to the solver.

    `on_turn` is accepted now (unused for static scanning) so the same object
    shape can later drive a TPS counter when live detection is added.
    """

    def __init__(
        self,
        camera_index: int = 0,
        on_turn: Optional[Callable[[], None]] = None,
        max_rescans: int = 3,
        debug: bool = False,
    ):
        self._camera_index = camera_index
        self._on_turn = on_turn
        self._max_rescans = max_rescans
        self._debug = debug
        self._classifier: Optional[ColorClassifier] = None

    # --- Classification is injectable so it can be tested without a camera. ---

    def set_classifier(self, classifier: ColorClassifier) -> None:
        self._classifier = classifier

    def classify_scan(self, faces: Dict[str, Sequence[RGB]]) -> str:
        """Turn captured face grids into a validated facelet string.

        Pure (no camera): used both by the live flow and by tests. Raises
        ValueError with the validation errors if the scan is structurally
        invalid.
        """
        if self._classifier is None:
            raise RuntimeError("calibrate (or set_classifier) before classifying")
        facelets = assemble_facelets(faces, self._classifier)
        result = validate_facelets(facelets)
        if not result.ok:
            raise ValueError("invalid scan: " + "; ".join(result.errors))
        return facelets

    def read_facelets(self) -> str:
        """Default scan: low-friction spin-to-scan (calibrate -> spin -> confirm).

        Rotate the cube slowly through all six faces; faces are auto-detected
        by center color. See read_facelets_manual() for the press-a-key backup.
        """
        return self._scan(spin=True)

    def read_facelets_manual(self) -> str:
        """Backup scan: show each face and press a key (robust under bad light)."""
        return self._scan(spin=False)

    def _scan(self, spin: bool) -> str:
        from . import capture  # lazy: only import OpenCV when scanning

        attempts = 0
        while True:
            if spin:
                # One continuous window: show white, then spin -- no window
                # switch. The classifier is built from the white sample inside
                # the same session.
                self._classifier, faces = capture.spin_scan_session(
                    camera_index=self._camera_index, debug=self._debug
                )
            else:
                # Manual backup: full six-center calibration (handles any scheme).
                centers = capture.calibrate_centers(self._camera_index)
                self._classifier = ColorClassifier.from_centers(centers)
                faces = capture.scan_faces(self._camera_index)

            facelets = assemble_facelets(faces, self._classifier)
            result = validate_facelets(facelets)

            if result.ok:
                # One-key confirm so a subtle misread doesn't silently trace.
                if capture.confirm_facelets(facelets, self._classifier):
                    return facelets
                print("Re-scanning at your request...")
            else:
                print("Scan did not validate:")
                for err in result.errors:
                    print(f"  - {err}")

            attempts += 1
            if attempts >= self._max_rescans:
                raise ValueError(
                    f"scan not accepted after {attempts} attempts; "
                    "try even lighting and avoid glare on stickers"
                )
            print(f"Re-scanning (attempt {attempts + 1}/{self._max_rescans})...")
