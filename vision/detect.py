"""Automatic cube-face detection geometry (pure, no camera / no OpenCV).

Replaces the fixed-box overlay: instead of asking the user to fit the cube into
a grid, we FIND the face in the frame. The fragile part was always sampling the
wrong pixels; detecting the stickers removes that at the source.

Split, as always, into:
  * PURE geometry here (testable): given candidate sticker squares (each a
    center (x, y) plus a size), decide whether they form a cube face and, if so,
    order their 9 centers into reading order (row-major, top-left first).
  * Thin OpenCV extraction in vision/capture.py: turn a frame into candidate
    squares (contours) and warp/sample at the ordered centers.

A "square" candidate is represented as (cx, cy, size) in pixel units. We do not
need full contours here -- centers and sizes are enough to recover the 3x3
grid, which is what color sampling needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Square:
    cx: float
    cy: float
    size: float  # side length in pixels (approx)


def pick_white_reference(
    samples: Sequence[Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """Pick the best 'white' from a face's 9 sticker samples.

    During white calibration the user shows the white face, but on a SCRAMBLED
    cube only the CENTER is guaranteed white -- the other 8 are random colors.
    So we must not just average/median the 9. A white sticker is the one that is
    BRIGHT and NEUTRAL (low saturation: its channels are close together). We
    score each sample by brightness minus a penalty for channel spread, and
    return the best. Robust to a center logo too, since samples are already
    trimmed-median per cell.

    Channel order in = channel order out.
    """
    if not samples:
        return (255.0, 255.0, 255.0)
    # Score mostly by BRIGHTNESS, lightly penalizing RELATIVE color (saturation)
    # so a bright white with a mild lighting cast (e.g. slightly blue) still
    # beats a dim flat grey. We only strongly avoid vividly colored cells.
    #   saturation = spread / brightness  (0 = neutral, ~1 = vivid)
    # White stickers are bright AND low-saturation; colored stickers are vivid.
    best = None
    best_score = float("-inf")
    for s in samples:
        brightness = sum(s) / 3.0
        if brightness <= 0:
            continue
        spread = max(s) - min(s)
        saturation = spread / brightness
        # Brightness dominates; subtract a brightness-scaled saturation penalty.
        score = brightness * (1.0 - 1.2 * saturation)
        if score > best_score:
            best_score = score
            best = s
    return tuple(best) if best is not None else (255.0, 255.0, 255.0)


def trimmed_median_color(
    pixels: Sequence[Tuple[float, float, float]],
    drop_dark: float = 0.25,
    drop_bright: float = 0.10,
) -> Tuple[float, float, float]:
    """Robust color of a sticker patch, ignoring a logo and glare.

    `pixels` is a list of (b, g, r) or (r, g, b) tuples from the sample patch.
    We sort by brightness, drop the darkest `drop_dark` fraction (logo marks /
    shadow) and the brightest `drop_bright` fraction (specular glare), and
    return the per-channel median of what's left. A small central logo is a
    minority of (dark) pixels and is discarded, so the sticker's true color
    dominates. Channel order in = channel order out.

    Pure Python so it's unit-tested; vision/capture.py uses the numpy
    equivalent on real frames.
    """
    if not pixels:
        return (0.0, 0.0, 0.0)
    order = sorted(range(len(pixels)), key=lambda i: sum(pixels[i]))
    lo = int(len(order) * drop_dark)
    hi = int(len(order) * (1.0 - drop_bright))
    keep = order[lo:hi] if hi > lo else order
    out = []
    for ch in range(3):
        vals = sorted(pixels[i][ch] for i in keep)
        n = len(vals)
        mid = n // 2
        out.append(vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0)
    return tuple(out)


def score_face_candidate_debug(
    rcx: float, rcy: float, rw: float, rh: float, extent: float,
    frame_w: float, frame_h: float,
):
    """Like score_face_candidate but also returns a human reject reason.

    Returns (score, reason). reason is '' if accepted, else which gate failed.
    Used by the batch harness to show WHY frames are rejected.
    """
    if rw <= 0 or rh <= 0 or frame_w <= 0 or frame_h <= 0:
        return 0.0, "degenerate size"
    frame_area = frame_w * frame_h
    area = rw * rh
    size_frac = area / frame_area
    aspect = max(rw, rh) / min(rw, rh)
    if aspect > 1.4:
        return 0.0, f"aspect {aspect:.2f} > 1.4 (not square)"
    if extent < 0.68:
        return 0.0, f"extent {extent:.2f} < 0.68 (poorly filled)"
    if not (0.015 <= size_frac <= 0.95):
        return 0.0, f"size_frac {size_frac:.3f} out of [0.015, 0.95]"
    return score_face_candidate(rcx, rcy, rw, rh, extent, frame_w, frame_h), ""


def score_face_candidate(
    rcx: float, rcy: float, rw: float, rh: float, extent: float,
    frame_w: float, frame_h: float,
) -> float:
    """Score a candidate cube-face rect; higher is more cube-like. 0 = reject.

    A real cube face held up to scan is: roughly SQUARE, well-FILLED by its
    region (extent), a reasonable SIZE (not tiny, not the whole frame), and
    near the frame CENTER (you hold it up centered; posters/shelves/doors sit
    at the edges). This single score replaces fixed thresholds + 'largest wins',
    so detection works across the full close->far distance range and ignores
    busy-background false positives.

    Returns 0.0 if the candidate is disqualified (wrong aspect, too small/large,
    poorly filled, or hugging a frame edge).
    """
    if rw <= 0 or rh <= 0 or frame_w <= 0 or frame_h <= 0:
        return 0.0
    frame_area = frame_w * frame_h
    area = rw * rh
    size_frac = area / frame_area

    # Hard rejects.
    aspect = max(rw, rh) / min(rw, rh)
    if aspect > 1.4:
        return 0.0
    if extent < 0.68:
        return 0.0
    if not (0.015 <= size_frac <= 0.95):  # ~12%..97% of a side length
        return 0.0

    # Centrality: 1.0 at frame center, decreasing toward edges.
    nx = (rcx - frame_w / 2) / (frame_w / 2)
    ny = (rcy - frame_h / 2) / (frame_h / 2)
    offcenter = (nx * nx + ny * ny) ** 0.5  # 0 center .. ~1.4 corner
    centrality = max(0.0, 1.0 - offcenter)

    # Prefer a square shape and a well-filled region.
    squareness = 1.0 / aspect          # 1.0 = perfect square
    fill = min(1.0, extent)

    # Mild size preference peaking around a comfortably-held cube (~1/9 of frame
    # area, i.e. ~1/3 of a side). Avoids both tiny specks and whole-frame grabs.
    import math
    size_pref = math.exp(-((size_frac - 0.11) ** 2) / (2 * 0.10 ** 2))

    return centrality * 1.4 + squareness + fill + size_pref


def subdivide_face(
    cx: float, cy: float, size: float, angle: float = 0.0, inset: float = 0.72
) -> List[Tuple[float, float]]:
    """Given a cube FACE square, return its 9 sticker-cell centers.

    This is the robust approach: rather than detecting 9 separate stickers
    (which keep merging in the mask), detect the one big cube square and divide
    it into a 3x3 grid. `(cx, cy)` is the face center, `size` the full face side
    length, `angle` an optional rotation (radians) for a tilted cube.

    `inset` (0..1) pulls the outer cells INWARD toward the face center so sample
    points stay well inside each sticker and away from the dark gaps / cube
    edge, even if the detected square slightly overshoots the cube. The step for
    the outer ring is (size/3)*inset; the center cell is always the face center.

    Returns 9 (x, y) centers in reading order (row-major, top-left first).
    """
    import math

    step = (size / 3.0) * inset
    ca, sa = math.cos(angle), math.sin(angle)
    centers: List[Tuple[float, float]] = []
    for row in (-1, 0, 1):
        for col in (-1, 0, 1):
            dx = col * step
            dy = row * step
            rx = dx * ca - dy * sa
            ry = dx * sa + dy * ca
            centers.append((cx + rx, cy + ry))
    return centers


def is_sticker_blob(
    area: float,
    bbox_w: float,
    bbox_h: float,
    frame_area: float,
    min_frac: float = 0.004,
    max_frac: float = 0.09,
    aspect_range: Tuple[float, float] = (0.6, 1.6),
    min_solidity: float = 0.6,
) -> bool:
    """Decide whether a detected color blob is a plausible sticker.

    Pure acceptance test shared by the OpenCV blob extractor, so the fiddly
    thresholds are unit-tested rather than buried in camera code. A sticker blob
    is sticker-sized (area within [min_frac, max_frac] of the frame), roughly
    square (bounding-box aspect in range), and reasonably SOLID (filled area vs
    bounding-box area) -- solidity tolerates rounded corners while rejecting
    thin background streaks.
    """
    if bbox_w <= 0 or bbox_h <= 0 or frame_area <= 0:
        return False
    if not (frame_area * min_frac <= area <= frame_area * max_frac):
        return False
    aspect = bbox_w / bbox_h
    if not (aspect_range[0] <= aspect <= aspect_range[1]):
        return False
    solidity = area / (bbox_w * bbox_h)
    if solidity < min_solidity:
        return False
    return True


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def filter_similar_squares(
    squares: Sequence[Square],
    size_tol: float = 0.5,
) -> List[Square]:
    """Keep only squares whose size is close to the median size.

    Sticker squares on one face are all about the same size; this drops stray
    contours (background, hands, reflections) that are much bigger/smaller.
    `size_tol` is the allowed fractional deviation from the median size.
    """
    if not squares:
        return []
    med = _median([s.size for s in squares])
    if med <= 0:
        return []
    lo, hi = med * (1 - size_tol), med * (1 + size_tol)
    return [s for s in squares if lo <= s.size <= hi]


def _cluster_1d(values: Sequence[float], gap: float) -> List[List[int]]:
    """Group indices whose values fall into 3 bands separated by >= gap.

    Returns a list of index-groups ordered by ascending value. Used to split
    the 9 centers into 3 rows (by y) and 3 columns (by x).
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups: List[List[int]] = []
    cur: List[int] = []
    last: Optional[float] = None
    for i in order:
        v = values[i]
        if last is not None and (v - last) > gap:
            groups.append(cur)
            cur = []
        cur.append(i)
        last = v
    if cur:
        groups.append(cur)
    return groups


def detect_face_grid(
    squares: Sequence[Square],
    grid_tol: float = 0.6,
) -> Optional[List[Tuple[float, float]]]:
    """Find a 3x3 cube face among candidate squares; return 9 ordered centers.

    Returns the 9 (x, y) sticker centers in reading order (row-major, top-left
    to bottom-right) if the squares form a plausible 3x3 face, else None.

    Approach: filter to similar-sized squares, then require exactly 9 that
    arrange into 3 rows and 3 columns (clustered by y and x, with gaps scaled
    to the sticker size). This is rotation-tolerant for small tilts; large
    perspective skew is corrected upstream (capture layer) before this runs.
    """
    cand = filter_similar_squares(squares)
    if len(cand) < 9:
        return None
    # If more than 9 survive, keep the 9 closest to their collective centroid
    # (the face is the tight cluster; extras are usually scattered noise).
    if len(cand) > 9:
        mx = _median([s.cx for s in cand])
        my = _median([s.cy for s in cand])
        cand = sorted(cand, key=lambda s: (s.cx - mx) ** 2 + (s.cy - my) ** 2)[:9]
    if len(cand) != 9:
        return None

    med_size = _median([s.size for s in cand])
    gap = med_size * grid_tol  # min separation between rows / columns

    ys = [s.cy for s in cand]
    xs = [s.cx for s in cand]
    rows = _cluster_1d(ys, gap)
    cols = _cluster_1d(xs, gap)
    if len(rows) != 3 or len(cols) != 3:
        return None
    if any(len(r) != 3 for r in rows) or any(len(c) != 3 for c in cols):
        return None

    # Build reading order: for each row (top->bottom), sort its 3 by x.
    ordered: List[Tuple[float, float]] = []
    for row in rows:  # rows already ascending by y
        row_sorted = sorted(row, key=lambda i: cand[i].cx)
        for i in row_sorted:
            ordered.append((cand[i].cx, cand[i].cy))
    return ordered
