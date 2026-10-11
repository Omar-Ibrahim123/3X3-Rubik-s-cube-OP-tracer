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


def has_logo_signature(
    pixels: Sequence[Tuple[float, float, float]],
    min_white_frac: float = 0.35,
    min_logo_frac: float = 0.08,
) -> bool:
    """Detect the 'white sticker with a printed logo' signature.

    On many cubes (e.g. GAN) the white CENTER sticker carries a colored logo.
    Rather than fight to read white through it, we use the logo as a POSITIVE
    signal: if a center patch contains BOTH a bright near-neutral majority
    (the white sticker) AND a saturated minority blob (the logo), it is the
    white face. This distinguishes it from a plain colored sticker (uniform,
    no bright-neutral region) -- so a blue logo is not confused with a blue
    sticker.

    `pixels` are (r, g, b) in 0..255. Returns True if the white+logo signature
    is present.
    """
    if not pixels:
        return False
    n = len(pixels)
    white_n = 0
    logo_n = 0
    for (r, g, b) in pixels:
        brightness = (r + g + b) / 3.0
        spread = max(r, g, b) - min(r, g, b)
        sat = spread / brightness if brightness > 0 else 0.0
        # White sticker: BRIGHT (even a blue-cast white reads ~190 bright) with
        # only mild color. Plain colored stickers are dimmer, so brightness is
        # the key discriminator that keeps a blue sticker from counting here.
        if brightness > 160 and sat < 0.5:
            white_n += 1
        elif sat > 0.7:
            logo_n += 1            # strongly saturated => logo mark
    return (white_n / n) >= min_white_frac and (logo_n / n) >= min_logo_frac


def logo_signature_fractions(pixels):
    """Diagnostic: return (white_frac, logo_frac, mean_brightness) for a patch.

    Lets us see WHY has_logo_signature does or doesn't fire on a real center
    patch, so the thresholds can be tuned to actual camera values.
    """
    if not pixels:
        return (0.0, 0.0, 0.0)
    n = len(pixels)
    white_n = logo_n = 0
    bright_sum = 0.0
    for (r, g, b) in pixels:
        brightness = (r + g + b) / 3.0
        bright_sum += brightness
        sat = (max(r, g, b) - min(r, g, b)) / brightness if brightness > 0 else 0.0
        if brightness > 160 and sat < 0.5:
            white_n += 1
        elif sat > 0.7:
            logo_n += 1
    return (white_n / n, logo_n / n, bright_sum / n)


def dominant_color(
    pixels: Sequence[Tuple[float, float, float]], bin_size: int = 48
) -> Tuple[float, float, float]:
    """Return the most common color of a patch by coarse-bin majority vote.

    For a CENTER sticker with a printed logo (a colored blob in the middle),
    the sticker's true color is the MAJORITY of the patch and the logo is a
    minority. Quantize pixels into coarse RGB bins, find the most populated bin,
    and return the average of the pixels in it -- so a central logo (or glare)
    that is a minority of the area is ignored. Channel order in = order out.
    """
    if not pixels:
        return (0.0, 0.0, 0.0)
    buckets = {}
    for p in pixels:
        key = (int(p[0]) // bin_size, int(p[1]) // bin_size, int(p[2]) // bin_size)
        buckets.setdefault(key, []).append(p)
    best = max(buckets.values(), key=len)
    n = len(best)
    return tuple(sum(p[ch] for p in best) / n for ch in range(3))


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


def find_cube_by_neighbors(squares):
    """Find a 3x3 cube face among candidate sticker squares; return 9 ordered.

    Adapted from the Qbr project (MIT, (c) 2016 Kim Koomen,
    https://github.com/kkoomen/qbr): rather than hunting for the cube as one big
    region (which proved fragile), find small sticker-sized squares, then locate
    the one that has exactly 9 neighbors arranged in a 3x3 grid around it (the
    center sticker). Those 9 neighbors ARE the cube face. Random background
    squares won't form a 3x3 grid, so this rejects clutter naturally.

    `squares` is a list of (x, y, w, h) axis-aligned sticker boxes. Returns the
    9 boxes sorted in reading order (top-left to bottom-right), or None.
    """
    if len(squares) < 9:
        return None

    radius = 1.5  # neighbor slots sit at +/- 1.5 * sticker size from a center

    def neighbors_of(cx, cy, w, h):
        return [
            (cx - w * radius, cy - h * radius), (cx, cy - h * radius),
            (cx + w * radius, cy - h * radius), (cx - w * radius, cy),
            (cx, cy), (cx + w * radius, cy),
            (cx - w * radius, cy + h * radius), (cx, cy + h * radius),
            (cx + w * radius, cy + h * radius),
        ]

    for (x, y, w, h) in squares:
        cx, cy = x + w / 2.0, y + h / 2.0
        slots = neighbors_of(cx, cy, w, h)
        matched = []
        for (x2, y2, w2, h2) in squares:
            for (sx, sy) in slots:
                # slot center falls inside this square's box?
                if x2 < sx and y2 < sy and x2 + w2 > sx and y2 + h2 > sy:
                    matched.append((x2, y2, w2, h2))
                    break
        if len(matched) == 9:
            # Sort into reading order: by y, then split into 3 rows by x.
            ys = sorted(matched, key=lambda b: b[1])
            rows = [
                sorted(ys[0:3], key=lambda b: b[0]),
                sorted(ys[3:6], key=lambda b: b[0]),
                sorted(ys[6:9], key=lambda b: b[0]),
            ]
            return rows[0] + rows[1] + rows[2]
    return None


def is_sticker_square(w, h, area, min_w=20, max_w=120):
    """Qbr-style sticker filter: near-square box, sticker-sized, well-filled.

    Pure, testable. `area` is the contour area; w,h the bounding box. Width
    bounds are wider than Qbr's fixed 30-60px so a cube held at varying
    distance still works.
    """
    if h <= 0 or w <= 0:
        return False
    ratio = w / float(h)
    if not (0.8 <= ratio <= 1.2):
        return False
    if not (min_w <= w <= max_w):
        return False
    if area / float(w * h) <= 0.4:
        return False
    return True


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
    if extent < 0.55:
        return 0.0, f"extent {extent:.2f} < 0.55 (poorly filled)"
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
    if extent < 0.55:
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
