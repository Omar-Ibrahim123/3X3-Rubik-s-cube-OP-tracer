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
from typing import Dict, List, Optional, Sequence, Tuple

from .colors import FACE_NAMES, RGB, ColorClassifier


@dataclass
class FaceCandidate:
    """One frame's reading of the face currently in view: 9 RGB samples."""

    samples: List[RGB]

    def center(self) -> RGB:
        return self.samples[4]


# ---------------------------------------------------------------------------
# Face-coherence gate (pure, testable).
#
# Root cause of the mixed-up scans seen on real hardware: a face was identified
# by its CENTER sticker alone, so if detection grabbed a wrong / rotated /
# partial region whose center happened to classify as some face, the whole
# garbled 9-sticker grid was accepted. Real debug data showed single "faces"
# containing stickers from four or five different real faces.
#
# A genuine cube face, by contrast, is internally consistent: its center is
# clearly one color (confident classification), and its 9 stickers do not span
# nearly every face of the cube. These two pure checks catch the mixed-region
# captures before they corrupt the facelet string.
# ---------------------------------------------------------------------------

# The gate's core signal (found by analysing real captures): on a GENUINE
# face, every one of the 9 stickers IS one of the six calibrated colors, so
# each sample sits close to its nearest calibrated reference in CIEDE2000. A
# mis-detected mixed region instead samples gaps / edges / halfway-between
# pixels, producing stickers that are FAR from every calibrated color. Real
# clean stickers measured well under 15 dE from their reference; the real
# mixed-region captures had several stickers at 20-29 dE.
#
# `max_sticker_distance` is how far one sticker may sit from its nearest
# calibrated color before it counts as "off" (gap/edge/blur). `max_off_stickers`
# is how many such off stickers a face may contain before it's judged a mixed
# region rather than one face. One off sticker is tolerated (occasional glare /
# a sample clipping a gap); two or more means the region isn't a clean face.
#
# `hard_ceiling_distance` is a stricter, single-sticker trip wire: a sticker
# this far from EVERY calibrated color cannot plausibly be a real sticker under
# normal glare (which stays well under this perceptually), so even ONE such
# sticker rejects the reading. This catches an otherwise-mostly-clean mixed
# region that happens to contain just one wildly off-cube sample.
DEFAULT_MAX_STICKER_DISTANCE = 18.0
DEFAULT_MAX_OFF_STICKERS = 1
DEFAULT_HARD_CEILING_DISTANCE = 22.0

# How many GLARE-blown stickers (clipped highlight + unmatched -- see
# is_glare_blown) a face may contain and still be accepted. Glare is distinct
# from a mixed-region mis-detection: a face with its center confident and all
# other stickers coherent, spoiled only by a couple of specular highlights, is
# still a real face held square-on -- rejecting it just makes the glossy-cube
# case impossible to scan. Non-glare off-stickers are still capped by
# max_off_stickers and the hard ceiling exactly as before; only clipped-and-
# unmatched highlights get this separate allowance.
DEFAULT_MAX_GLARE_STICKERS = 2


@dataclass
class FaceCoherence:
    """Diagnostic for how plausible 9 samples are as one real cube face."""

    face: str                       # face letter the CENTER classifies as
    center_margin: float            # confidence gap of the center classification
    sticker_distances: List[float]  # each sticker's distance to nearest color
    off_stickers: int               # non-glare stickers far from every color
    color_counts: Dict[str, int]    # per-face classification counts over the 9
    ok: bool
    reason: str = ""
    glare_stickers: List[int] = field(default_factory=list)  # indices blown by glare


def _nearest_reference_distance(
    sample: RGB, classifier: ColorClassifier
) -> float:
    """CIEDE2000 distance from `sample` to its NEAREST calibrated color."""
    return min(
        classifier._dist(sample, ref) for ref in classifier.references.values()
    )


# A glare-blown sticker is one the camera over-exposed: a specular highlight on
# a glossy sticker clips one or more channels toward 255 and washes the colour
# out, so the sample lands FAR from every calibrated colour. Real captured data
# showed these as values like (238,249,255) / (230,244,255) -- a channel pinned
# near 255 while the reading matches nothing (dE > 20).
#
# This is deliberately a TWO-part test so it can never flag a genuine sticker:
#   1. clipped  -- at least one channel is at/near the 255 ceiling, AND
#   2. unmatched -- the sample is far from EVERY calibrated colour.
# A real bright sticker (even white) is not clipped on a well-exposed frame and
# matches its calibrated colour closely, so condition (2) protects it. Both
# conditions must hold, so a bright-but-real sticker is never called glare.
DEFAULT_CLIP_CHANNEL = 250.0        # a channel this high is treated as clipped
DEFAULT_GLARE_MIN_DISTANCE = 18.0   # and this far from any colour = washed out


def is_glare_blown(
    sample: RGB,
    classifier: ColorClassifier,
    clip_channel: float = DEFAULT_CLIP_CHANNEL,
    min_distance: float = DEFAULT_GLARE_MIN_DISTANCE,
) -> bool:
    """True if `sample` looks like an over-exposed (glare) reading, not colour.

    Pure and conservative: requires BOTH a clipped channel AND a poor match to
    every calibrated colour, so a genuine bright sticker is never misflagged.
    """
    clipped = any(c >= clip_channel for c in sample)
    if not clipped:
        return False
    return _nearest_reference_distance(sample, classifier) >= min_distance


def face_coherence(
    samples: Sequence[RGB],
    classifier: ColorClassifier,
    max_sticker_distance: float = DEFAULT_MAX_STICKER_DISTANCE,
    max_off_stickers: int = DEFAULT_MAX_OFF_STICKERS,
    hard_ceiling_distance: float = DEFAULT_HARD_CEILING_DISTANCE,
    max_glare_stickers: int = DEFAULT_MAX_GLARE_STICKERS,
) -> FaceCoherence:
    """Judge whether 9 sticker samples plausibly belong to ONE cube face.

    The physical check: every sticker on a real face is one of the six
    calibrated cube colors, so every sample should sit CLOSE to its nearest
    calibrated reference. Count the stickers that are far from EVERY calibrated
    color -- those are gap / edge / mid-rotation samples. If too many stickers
    are "off", the 9 points did not land on one clean face (the mis-detection
    failure mode) and the reading is rejected.

    This deliberately does NOT limit how many distinct colors a face shows: a
    legally scrambled face can show all six, and every one of those stickers is
    still close to a calibrated color, so it passes.

    Pure: no camera, no OpenCV. Returns a FaceCoherence diagnostic so the
    caller can show the user exactly why a face was rejected.
    """
    if len(samples) != 9:
        return FaceCoherence(
            face="", center_margin=0.0, sticker_distances=[], off_stickers=0,
            color_counts={}, ok=False,
            reason=f"expected 9 samples, got {len(samples)}",
        )

    center_face, center_margin = classifier.classify_with_confidence(samples[4])

    counts: Dict[str, int] = {}
    distances: List[float] = []
    off = 0                      # non-glare stickers far from every colour
    glare_idx: List[int] = []    # stickers blown out by specular glare
    off_distances: List[float] = []  # distances of the NON-glare off stickers
    for i, sample in enumerate(samples):
        label = classifier.classify(sample)
        counts[label] = counts.get(label, 0) + 1
        d = _nearest_reference_distance(sample, classifier)
        distances.append(d)
        if d > max_sticker_distance:
            # Separate a washed-out specular highlight (recoverable: the face
            # is still real) from a genuine mixed-region/off-cube sample (a
            # mis-detection we must still reject).
            if is_glare_blown(sample, classifier):
                glare_idx.append(i)
            else:
                off += 1
                off_distances.append(d)

    worst = max(distances)
    # The hard ceiling only trips on NON-glare stickers: a glare highlight can
    # legitimately be very far from every colour, so it must not by itself kill
    # an otherwise-clean face. Mixed-region protection is unchanged.
    worst_non_glare = max(off_distances) if off_distances else 0.0
    too_many_off = off > max_off_stickers
    one_wildly_off = worst_non_glare > hard_ceiling_distance
    too_much_glare = len(glare_idx) > max_glare_stickers
    ok = not (too_many_off or one_wildly_off or too_much_glare)
    reason = ""
    if not ok:
        if too_much_glare and not (too_many_off or one_wildly_off):
            reason = (
                f"{len(glare_idx)} of 9 stickers are blown out by glare "
                f"(worst {worst:.1f} dE); tilt the cube or soften the light so "
                "the shine moves off the stickers, then hold it square-on"
            )
        else:
            reason = (
                f"{off} of 9 stickers don't match any calibrated color "
                f"(worst {worst_non_glare:.1f} dE); "
                "looks like a mixed region or off-cube sample, not one clean face"
            )

    return FaceCoherence(
        face=center_face,
        center_margin=center_margin,
        sticker_distances=distances,
        off_stickers=off,
        color_counts=counts,
        ok=ok,
        reason=reason,
        glare_stickers=glare_idx,
    )


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

    # When True, reject candidates whose 9 samples don't look like one real
    # face (see face_coherence). This is the sanity gate that stops a
    # mis-detected mixed region from being accepted and corrupting the scan.
    coherence_gate: bool = True
    max_sticker_distance: float = DEFAULT_MAX_STICKER_DISTANCE
    max_off_stickers: int = DEFAULT_MAX_OFF_STICKERS
    hard_ceiling_distance: float = DEFAULT_HARD_CEILING_DISTANCE
    max_glare_stickers: int = DEFAULT_MAX_GLARE_STICKERS

    _collected: Dict[str, List[RGB]] = field(default_factory=dict)
    _streak_face: Optional[str] = None
    _streak_center: Optional[RGB] = None
    _streak_count: int = 0
    _last_accepted: Optional[str] = None
    # Why the most recent candidate was rejected (empty if accepted/steadying).
    # The camera layer surfaces this so the user knows to re-show a face.
    last_rejection: str = ""

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

        # Sanity gate: a real face is internally coherent. Reject mixed-region
        # captures (the mis-detection failure mode) BEFORE they can form a
        # steady streak or be collected -- otherwise a garbled grid whose center
        # merely happens to classify as some face corrupts the scan.
        if self.coherence_gate and self.classifier.perceptual:
            coherence = face_coherence(
                candidate.samples,
                self.classifier,
                max_sticker_distance=self.max_sticker_distance,
                max_off_stickers=self.max_off_stickers,
                hard_ceiling_distance=self.hard_ceiling_distance,
                max_glare_stickers=self.max_glare_stickers,
            )
            if not coherence.ok:
                self.last_rejection = coherence.reason
                self._streak_face = None
                self._streak_center = None
                self._streak_count = 0
                return None
        self.last_rejection = ""

        # Is this the same face, held roughly as still, as the running streak?
        # Use the classifier's own metric: CIEDE2000 for a calibrated palette,
        # legacy HSV otherwise. Their scales differ, so use an appropriate
        # threshold for each.
        stable_limit = 5.0 if self.classifier.perceptual else self.steady_dist
        same_as_streak = (
            self._streak_face == face
            and self._streak_center is not None
            and self.classifier.distance(center, self._streak_center)
            <= stable_limit
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
        self.last_rejection = ""
