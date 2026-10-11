"""Pure state machine for reliable six-center palette calibration.

The UI prompts for U/R/F/D/L/B in order. A face is accepted only after a stable
run of center samples, and the next prompt cannot begin until the previous face
has left view (detection lost or a perceptually different center appears).
This prevents one held face from silently being recorded under several labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .colorsci import rgb_distance

RGB = tuple[float, float, float]
CALIBRATION_ORDER = ("U", "R", "F", "D", "L", "B")


def median_rgb(samples: Sequence[RGB]) -> RGB:
    """Per-channel median of several stable center samples."""
    if not samples:
        return (0.0, 0.0, 0.0)
    result = []
    for channel in range(3):
        values = sorted(sample[channel] for sample in samples)
        middle = len(values) // 2
        if len(values) % 2:
            result.append(values[middle])
        else:
            result.append((values[middle - 1] + values[middle]) / 2.0)
    return tuple(result)


@dataclass
class PaletteCalibrationTracker:
    """Collect six distinct, stable center colors under fixed semantic labels."""

    order: tuple[str, ...] = CALIBRATION_ORDER
    stable_frames: int = 6
    stable_distance: float = 4.0
    min_palette_distance: float = 7.0
    transition_frames: int = 3

    palette: dict[str, RGB] = field(default_factory=dict)
    _stable: list[RGB] = field(default_factory=list)
    _waiting_for_transition: bool = False
    _transition_count: int = 0
    _last_accepted: RGB | None = None
    last_message: str = ""

    @property
    def complete(self) -> bool:
        return len(self.palette) == len(self.order)

    @property
    def expected_face(self) -> str | None:
        if self.complete:
            return None
        return self.order[len(self.palette)]

    @property
    def waiting_for_transition(self) -> bool:
        return self._waiting_for_transition

    @property
    def stable_count(self) -> int:
        return len(self._stable)

    def offer(self, center: RGB | None) -> str | None:
        """Offer a detected center (or None when no face is detected).

        Returns the semantic face label accepted on this call, otherwise None.
        """
        if self.complete:
            return None

        if self._waiting_for_transition:
            transitioned = center is None
            if center is not None and self._last_accepted is not None:
                transitioned = (
                    rgb_distance(center, self._last_accepted)
                    >= self.min_palette_distance
                )
            self._transition_count = self._transition_count + 1 if transitioned else 0
            if self._transition_count >= self.transition_frames:
                self._waiting_for_transition = False
                self._transition_count = 0
                self._stable.clear()
                self.last_message = "new face detected; hold steady"
            else:
                self.last_message = "turn to the next requested face"
            return None

        if center is None:
            self._stable.clear()
            self.last_message = "no face detected"
            return None

        # A new palette entry must be perceptually distinct from every accepted
        # center. This catches duplicate/reused faces before they corrupt labels.
        nearest = min(
            (rgb_distance(center, reference) for reference in self.palette.values()),
            default=float("inf"),
        )
        if nearest < self.min_palette_distance:
            self._stable.clear()
            self.last_message = "this color is already calibrated; show requested face"
            return None

        if self._stable and rgb_distance(center, self._stable[-1]) > self.stable_distance:
            self._stable = [center]
        else:
            self._stable.append(center)
        self.last_message = f"hold steady ({len(self._stable)}/{self.stable_frames})"

        if len(self._stable) < self.stable_frames:
            return None

        accepted = median_rgb(self._stable[-self.stable_frames :])
        face = self.expected_face
        self.palette[face] = accepted
        self._last_accepted = accepted
        self._stable.clear()
        self._transition_count = 0
        self._waiting_for_transition = not self.complete
        self.last_message = f"calibrated {face}"
        return face
