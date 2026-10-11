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


def subdivide_face(
    cx: float, cy: float, size: float, angle: float = 0.0
) -> List[Tuple[float, float]]:
    """Given a cube FACE square, return its 9 sticker-cell centers.

    This is the robust approach: rather than detecting 9 separate stickers
    (which keep merging in the mask), detect the one big cube square and divide
    it into a 3x3 grid. `(cx, cy)` is the face center, `size` the full face side
    length, `angle` an optional rotation (radians) for a tilted cube.

    Returns 9 (x, y) centers in reading order (row-major, top-left first). Cell
    centers sit at +/- size/3 from the face center along the (rotated) axes.
    """
    import math

    step = size / 3.0
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
