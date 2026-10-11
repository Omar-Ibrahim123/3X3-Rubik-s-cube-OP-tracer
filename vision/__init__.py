"""Vision layer: turn camera frames into a 54-char facelet string.

Current scope: STATIC scanning (show each face, classify colors). The module
is organized so that a future live, camera-based turn-detection pipeline can
be added alongside the static scanner without changing anything downstream:
both ultimately feed the same facelet-string / turn-signal interfaces that the
cube model, solver, and TPS counter already consume.

Only the pure color-classification logic (colors.py) and the scanner's
orchestration import at package load. OpenCV (capture.py) is imported lazily
so this package and the test suite work without OpenCV installed.
"""

from .colors import (
    STANDARD_SCHEME,
    ColorClassifier,
    ValidationResult,
    assemble_facelets,
    validate_facelets,
)
from .scanner import FaceletSource, StaticScanner

__all__ = [
    "FaceletSource",
    "StaticScanner",
    "ColorClassifier",
    "ValidationResult",
    "STANDARD_SCHEME",
    "assemble_facelets",
    "validate_facelets",
]
