"""Vision interface boundary (static scan now, live detection later).

`FaceletSource` is the single contract the rest of the app depends on: it
yields a 54-character facelet string. The static scanner will implement it on
top of OpenCV; a future `LiveTurnDetector` can implement the same contract
(plus a turn-signal callback feeding tps.TPSCounter.record_turn) without any
downstream changes.

Kept dependency-free for now so the pure-logic core imports and tests without
requiring OpenCV. The OpenCV pipeline lands in a later phase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional


class FaceletSource(ABC):
    """Anything that can produce a 54-char facelet string."""

    @abstractmethod
    def read_facelets(self) -> str:
        """Return a 54-character facelet string (Kociemba face order)."""
        raise NotImplementedError


class StaticScanner(FaceletSource):
    """Scan a stationary cube face-by-face via a camera.

    Placeholder: the OpenCV capture + color-classification pipeline is added
    in the vision phase. The constructor accepts an optional turn-signal hook
    now so the same object can later drive a TPS counter when live detection
    is implemented.
    """

    def __init__(self, on_turn: Optional[Callable[[], None]] = None):
        self._on_turn = on_turn

    def read_facelets(self) -> str:  # pragma: no cover - not yet implemented
        raise NotImplementedError(
            "Camera scanning is not implemented yet. For now, construct "
            "CubeState.from_facelets(...) with a facelet string directly. "
            "The OpenCV pipeline will implement this method in the vision phase."
        )
