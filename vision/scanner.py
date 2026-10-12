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
      1. In one continuous camera/window session, show six distinct labeled
         centers (U/R/F/D/L/B) to learn a CIEDE2000 palette. Calibration waits
         for a real face transition and rejects duplicate colors.
      2. Without reopening the camera, spin through all six faces; sample nine
         colors from each detected 3x3 sticker grid.
      3. Classify + assemble into a 54-char facelet string.
      4. Validate; if invalid, report the errors and let the user re-scan.

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
        facelets = self._orient(facelets)
        result = validate_facelets(facelets)
        if not result.ok:
            raise ValueError("invalid scan: " + "; ".join(result.errors))
        return facelets

    @staticmethod
    def _orient(facelets: str) -> str:
        """Fix per-face rotation if a unique solvable orientation exists.

        Returns the re-oriented 54-char string when solve_orientation finds a
        unique solution; otherwise returns the input unchanged so the normal
        validation/error path reports the problem (an ambiguous or genuinely
        unsolvable scan is better surfaced than silently reshuffled).
        """
        from cube.orientation import FACE_ORDER, solve_orientation

        face_grids = {
            FACE_ORDER[i]: facelets[i * 9:(i + 1) * 9] for i in range(6)
        }
        result = solve_orientation(face_grids)
        if result.ok and result.facelets is not None:
            # (orientation note intentionally not printed: keep the terminal
            # clean; all scan feedback happens in the on-screen window)
            return result.facelets
        return facelets

    def read_facelets(self) -> str:
        """Default: one-window six-color calibration -> spin -> confirm."""
        return self._scan(mode="auto")

    def read_facelets_box(self) -> str:
        """Fallback: fixed 3x3 box you size to the cube with +/- keys."""
        return self._scan(mode="box")

    def read_facelets_manual(self) -> str:
        """Backup scan: show each face and press a key (robust under bad light)."""
        return self._scan(mode="manual")

    def _scan(self, mode: str) -> str:
        from . import capture  # lazy: only import OpenCV when scanning

        attempts = 0
        while True:
            if mode == "auto":
                # One uninterrupted camera/window state machine: calibrate six
                # distinct centers, build the CIEDE2000 palette, then scan.
                # Keeping one camera handle also keeps exposure/white-balance
                # consistent between calibration and classification.
                from .session import calibrate_and_spin

                self._classifier, faces = calibrate_and_spin(
                    camera_index=self._camera_index,
                    debug=self._debug,
                )
            elif mode == "box":
                # Fixed box the user sizes with +/- (previous default).
                self._classifier, faces = capture.spin_scan_session(
                    camera_index=self._camera_index, debug=self._debug
                )
            else:
                # Manual: full six-center calibration (handles any scheme).
                centers = capture.calibrate_centers(self._camera_index)
                self._classifier = ColorClassifier.from_centers(centers)
                faces = capture.scan_faces(self._camera_index)

            facelets = assemble_facelets(faces, self._classifier)
            # Recover per-face rotation: faces are captured at whatever angle
            # the cube was held, which otherwise yields a cube with correct
            # colour counts but impossible pieces. solve_orientation finds the
            # unique set of per-face rotations that make a solvable cube.
            facelets = self._orient(facelets)
            result = validate_facelets(facelets)

            if result.ok:
                # Build a report (solvability, parity, low-confidence stickers)
                # from the FINAL string plus the captured RGB samples, so the
                # user can tell a correct read from a plausible-but-wrong one.
                from .confirm import build_report

                report = build_report(facelets, faces, self._classifier)
                # In-window confirm so a subtle misread doesn't silently trace.
                # Nothing is printed: the terminal stays clean, the user accepts
                # or re-scans with a keypress in the net window.
                if capture.confirm_facelets(facelets, self._classifier, report):
                    return facelets
            # Invalid scan or user re-scan: loop silently and try again. The
            # terminal is kept clean on purpose (feedback is on-screen); the
            # final give-up below still raises so the caller can report it.

            attempts += 1
            if attempts >= self._max_rescans:
                raise ValueError(
                    f"scan not accepted after {attempts} attempts; "
                    "try even lighting and avoid glare on stickers"
                )
