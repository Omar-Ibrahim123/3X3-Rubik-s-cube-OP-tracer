"""Vision layer: turn camera frames into a 54-char facelet string.

Current scope: STATIC scanning (show each face, classify colors). The module
is organized so that a future live, camera-based turn-detection pipeline can
be added alongside the static scanner without changing anything downstream:
both ultimately feed the same facelet-string / turn-signal interfaces that the
cube model, solver, and TPS counter already consume.

Implementation of the actual OpenCV pipeline is intentionally deferred; this
package currently defines the interface boundary only.
"""

from .scanner import FaceletSource, StaticScanner

__all__ = ["FaceletSource", "StaticScanner"]
