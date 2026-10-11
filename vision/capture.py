"""OpenCV camera capture layer for the static cube scanner.

THIS IS THE LAYER YOU RUN LOCALLY WITH YOUR WEBCAM. It is intentionally thin:
all the color-decision logic lives in vision/colors.py (and is unit-tested).
This file only:
  * opens the webcam,
  * draws a 3x3 sampling grid overlay so you can line up each face,
  * samples the average color inside each of the 9 grid cells, and
  * walks you through calibration (show each center once) then the 6 faces.

OpenCV and numpy are imported lazily inside functions so that importing the
`vision` package (and running the pure-logic tests) never requires them. Install
them locally with:

    pip install opencv-python numpy

Controls during scanning:
    SPACE / ENTER  capture the current face (or center during calibration)
    r              re-capture the last face
    q / ESC        abort

Guided flow (default face order U R F D L B): the on-screen prompt tells you
which face to show. Hold the cube so the face fills the grid, matching the
standard orientation (white U / green F etc. — any fixed scheme works as long
as you keep the cube oriented consistently while turning it to show each face).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .colors import FACE_NAMES, RGB

# Default guidance for how to present each face to the camera. These are hints
# shown on screen; the only hard requirement is that you show the correct face
# for each prompt.
_FACE_PROMPTS = {
    "U": "UP face (hold top face toward camera)",
    "R": "RIGHT face",
    "F": "FRONT face",
    "D": "DOWN face",
    "L": "LEFT face",
    "B": "BACK face",
}


def _lazy_imports():
    """Import OpenCV/numpy only when actually running the camera."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "The camera scanner needs OpenCV and numpy. Install them with:\n"
            "    pip install opencv-python numpy\n"
            f"(original import error: {exc})"
        ) from exc
    return cv2, np


def save_frames(camera_index: int = 0, out_dir: str = "frames"):
    """Live preview; press SPACE to save the current frame as a PNG.

    Diagnostic tool for tuning cube detection: capture a few real frames of your
    cube (different faces/angles) and share them. Each SPACE writes
    frames/frameNN.png. Also overlays what the CURRENT detector finds (circles
    on detected stickers), so you can see live whether detection fires at all.
    Press q/ESC to finish.
    """
    import os

    cv2, _ = _lazy_imports()
    from .detect import detect_face_grid

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")
    saved = []
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("failed to read camera frame")
            disp = frame.copy()
            # Show what the current detector sees, so detection is visible live.
            centers = _detect_face_centers(frame)
            cv2.putText(disp, "hold cube CLOSE, square to camera",
                        (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                        2, cv2.LINE_AA)
            if centers is not None:
                for (x, y) in centers:
                    cv2.circle(disp, (int(x), int(y)), 10, (0, 255, 0), 2)
                cv2.putText(disp, "FACE DETECTED", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2,
                            cv2.LINE_AA)
            cv2.putText(disp, "SPACE=save frame   q=done", (20, disp.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                        cv2.LINE_AA)
            cv2.imshow("save-frame (share these PNGs)", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord(" "), 13):
                path = os.path.join(out_dir, f"frame{n:02d}.png")
                cv2.imwrite(path, frame)  # save the CLEAN frame, no overlay
                saved.append(path)
                print(f"saved {path}")
                n += 1
            elif key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return saved


def analyze_frame(path: str):
    """Run detection on a SAVED image and dump diagnostics + debug images.

    Offline tuning tool: no camera needed. Loads `path`, runs the exact mask +
    blob + grid pipeline, prints what it found (counts, blob sizes, whether a
    3x3 face was detected), and writes two images next to the input:
      <name>_mask.png       -- the binary sticker mask
      <name>_annotated.png  -- blobs boxed, detected grid circled
    Share those two PNGs so detection can be tuned against real output.
    """
    import os

    cv2, np = _lazy_imports()

    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"could not read image: {path}")
    h, w = frame.shape[:2]
    print(f"image: {path}  ({w}x{h})")

    mask = _sticker_mask(frame)
    white_px = int((mask > 0).sum())
    print(f"mask: {100.0 * white_px / (h * w):.1f}% of pixels are 'sticker-like'")

    face = _find_cube_face(frame)
    if face is None:
        print("no cube face found")
    else:
        fcx, fcy, fsize, fang = face
        import math
        print(f"cube face: center=({fcx:.0f},{fcy:.0f}) size={fsize:.0f}px "
              f"angle={math.degrees(fang):.1f}deg")

    centers = _detect_face_centers(frame)
    print("FACE DETECTED (3x3 grid placed)" if centers else "no face detected")

    # Annotated image: cube square + the 9 subdivided cell centers.
    disp = frame.copy()
    if face is not None:
        cv2.circle(disp, (int(face[0]), int(face[1])), int(face[2] / 2),
                   (255, 0, 255), 2)
    if centers:
        for (x, y) in centers:
            cv2.circle(disp, (int(x), int(y)), 10, (0, 255, 0), 3)

    base, _ = os.path.splitext(path)
    mask_path = base + "_mask.png"
    annot_path = base + "_annotated.png"
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(annot_path, disp)
    print(f"\nwrote:\n  {mask_path}\n  {annot_path}\nShare these two images.")
    return mask_path, annot_path


def analyze_dir(folder: str):
    """Run detection on every image in a folder; print a pass/fail summary.

    Batch-tuning harness: lets detection be evaluated against many real frames
    at once (close/medium/far, different faces, some tilt) so a change can be
    checked for not breaking some distances while fixing others. For each image
    prints whether a cube face was detected and its size/center; writes an
    _annotated.png for each. Skips files that already end in _mask/_annotated.
    """
    import glob
    import os

    cv2, _ = _lazy_imports()

    paths = sorted(
        p for p in glob.glob(os.path.join(folder, "*.png"))
        if not p.endswith(("_mask.png", "_annotated.png"))
    )
    if not paths:
        print(f"no .png frames found in {folder}")
        return
    from .detect import score_face_candidate_debug

    detected = 0
    print(f"Analyzing {len(paths)} frame(s) in {folder}:\n")
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"  {os.path.basename(path)}: could not read")
            continue
        h, w = frame.shape[:2]
        name = os.path.basename(path)
        face = _find_cube_face(frame)
        centers = _detect_face_centers(frame)

        if face is not None and centers is not None:
            detected += 1
            fcx, fcy, fsize, _ = face
            print(f"  {name}: DETECTED  size={fsize:.0f}px "
                  f"({100.0 * fsize / w:.0f}% of width)")
        else:
            # Diagnostic: show the top contours by area and why each was
            # rejected, so detection can be tuned against reality.
            print(f"  {name}: NOT detected")
            mask = _cube_edge_mask(frame)[0]
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cs = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
            if not cs:
                print("      (no contours in mask at all)")
            for c in cs:
                a = cv2.contourArea(c)
                (rcx, rcy), (rw, rh), _ = cv2.minAreaRect(c)
                if rw <= 0 or rh <= 0:
                    continue
                extent = a / (rw * rh)
                _, reason = score_face_candidate_debug(
                    rcx, rcy, rw, rh, extent, float(w), float(h)
                )
                szpct = 100.0 * ((rw + rh) / 2.0) / w
                print(f"      contour: size={szpct:.0f}%w "
                      f"aspect={max(rw,rh)/min(rw,rh):.2f} extent={extent:.2f} "
                      f"center=({rcx:.0f},{rcy:.0f}) -> "
                      f"{reason or 'ACCEPTED?'}")

        disp = frame.copy()
        if face is not None:
            cv2.circle(disp, (int(face[0]), int(face[1])), int(face[2] / 2),
                       (255, 0, 255), 2)
        if centers is not None:
            for (x, y) in centers:
                cv2.circle(disp, (int(x), int(y)), 8, (0, 255, 0), 2)
        base, _ = os.path.splitext(path)
        cv2.imwrite(base + "_annotated.png", disp)
        cv2.imwrite(base + "_mask.png", _cube_edge_mask(frame)[0])
    print(f"\n{detected}/{len(paths)} frames detected. "
          f"Annotated + mask images written next to each frame.")


def _cube_edge_mask(frame):
    """Mask of the cube via EDGE DENSITY (robust to a bright background).

    Key finding from real masks: the user scans in a bright room (white walls,
    door, light skin), so brightness/saturation masking detects the BACKGROUND,
    not the cube. What actually distinguishes the cube is DETAIL: it's packed
    with edges (sticker borders + dark gaps), while walls/doors/skin are smooth.

    So: Canny edges -> heavy dilate+close so the cube's dense edges merge into
    one solid blob while sparse background edges stay small -> that blob is the
    cube. Returns a binary mask where the cube region is white.
    """
    cv2, np = _lazy_imports()
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    # Bridge the sticker-border lattice into a filled blob -- but modestly, so
    # the cube does NOT merge with nearby face/hair/poster detail into one giant
    # region (that caused far-away false positives covering the whole head).
    k = max(5, int(round(min(h, w) * 0.025)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.dilate(edges, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Erode back to undo the dilation's growth and snap thin bridges to other
    # detail, leaving the cube as its own compact blob.
    mask = cv2.erode(mask, kernel, iterations=1)
    # Fill interior holes (the lattice gaps) so the blob is solid, WITHOUT a
    # convex hull (hulling tended to swallow neighbouring detail).
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask, edges


def _sticker_mask(frame):
    """Binary mask of sticker-like pixels (vivid OR bright), gaps removed.

    Shared by the blob detector and the debug mask view so what you SEE is
    exactly what detection uses.
    """
    cv2, np = _lazy_imports()
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    # Vivid stickers: high saturation. White stickers: low saturation but high
    # brightness. Combine both; exclude the dark gaps (low V).
    saturated = s > 70
    bright_white = (s <= 70) & (v > 150)
    mask = ((saturated | bright_white) & (v > 60)).astype(np.uint8) * 255
    # Merge each sticker into a solid blob (close gaps/glare), then open to drop
    # thin noise.
    k = max(3, int(round((min(h, w)) * 0.012)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Fill interior holes so a center LOGO (a dark mark inside a white sticker)
    # doesn't punch a hole in that sticker's blob or split it. We refill each
    # external contour solid: holes inside a blob vanish, outer shape unchanged.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask


def _find_cube_face(frame, min_frac=0.02, max_frac=0.75):
    """Find the cube FACE as one big square, return (cx, cy, size, angle_rad).

    The robust approach: the mask renders the whole face as a single solid
    square (the 9 stickers fused). Instead of fighting that to separate
    stickers, we embrace it -- find the largest square-ish region (the cube) and
    subdivide it into a 3x3 grid elsewhere. Returns None if no plausible face.

    Picks the largest contour whose minimum-area rotated rect is roughly square
    and sized like a close-held cube (a fraction of the frame). Returns the
    rect's center, side length (mean of the two sides), and rotation in radians.
    """
    import math

    cv2, np = _lazy_imports()
    from .detect import score_face_candidate

    h, w = frame.shape[:2]
    mask, edges = _cube_edge_mask(frame)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    best = None
    best_score = 0.0
    for c in contours:
        a = cv2.contourArea(c)
        rect = cv2.minAreaRect(c)  # ((cx,cy),(rw,rh),angle_deg)
        (rcx, rcy), (rw, rh), ang = rect
        if rw <= 0 or rh <= 0:
            continue
        extent = a / (rw * rh)
        s = score_face_candidate(rcx, rcy, rw, rh, extent, float(w), float(h))
        if s <= 0 or s <= best_score:
            continue
        # Validate by EDGE DENSITY inside the candidate box: a real cube is
        # uniformly packed with edges; a far-away false positive that spans the
        # head/wall is mostly empty. This rejects the 'cube+face+wall' blobs.
        x, y, bw, bh = cv2.boundingRect(c)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        sub = edges[y0:y1, x0:x1]
        if sub.size == 0:
            continue
        density = float((sub > 0).sum()) / sub.size
        if density < 0.06:  # too sparse to be a detail-packed cube face
            continue
        best_score = s
        size = (rw + rh) / 2.0
        best = (rcx, rcy, size, math.radians(ang))
    return best


def _detect_face_centers(frame):
    """Return the 9 sticker-cell centers of the detected cube face, or None.

    Finds the cube's outer square and subdivides it into a 3x3 grid. This is the
    detection used by the scan flows. `angle` from minAreaRect is normalized to
    a small tilt so a near-upright cube subdivides correctly.
    """
    import math

    face = _find_cube_face(frame)
    if face is None:
        return None
    cx, cy, size, angle = face
    # Normalize the rect angle to [-45, 45] deg: a square's orientation is
    # ambiguous mod 90, and we only need the small tilt from upright.
    deg = math.degrees(angle)
    deg = ((deg + 45) % 90) - 45
    from .detect import subdivide_face
    return subdivide_face(cx, cy, size, math.radians(deg))


def _find_sticker_blobs(frame, min_frac=0.004, max_frac=0.09):
    """Detect sticker candidates by COLOR REGION (robust to rounded/glossy cubes).

    Why not square contours: the user's stickerless glossy cube has rounded
    sticker shapes, thin gaps, and glare -- edges don't form clean 4-point
    squares. Instead we find blobs that are either vividly colored OR bright
    (white stickers), separated from the dark gaps and most backgrounds.

    Pipeline: HSV -> mask of (saturated OR bright) pixels -> morphological close
    to merge each sticker into one blob and seal glare holes -> contours ->
    keep blobs of sticker-ish AREA that are reasonably SOLID (filled) and
    roughly square by bounding box. Returns detect.Square(cx, cy, size).

    Area thresholds are fractions of frame area; defaults assume the cube is
    held fairly CLOSE (fills a good part of the frame), which is the reliable
    way to use this.
    """
    cv2, np = _lazy_imports()
    from .detect import Square, is_sticker_blob

    h, w = frame.shape[:2]
    frame_area = float(h * w)

    mask = _sticker_mask(frame)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    blobs = []
    for c in contours:
        a = cv2.contourArea(c)
        x, y, bw, bh = cv2.boundingRect(c)
        if not is_sticker_blob(a, bw, bh, frame_area, min_frac, max_frac):
            continue
        cx, cy = x + bw / 2.0, y + bh / 2.0
        blobs.append(Square(cx=cx, cy=cy, size=(bw + bh) / 2.0))
    return blobs


def _find_sticker_squares(frame, min_frac=0.0015, max_frac=0.08):
    """Detect candidate sticker squares in a frame -> list of detect.Square.

    Legacy square-contour detector (kept as a fallback). Thin OpenCV layer that
    finds square-ish contours of plausible sticker size. The pure grid logic
    (vision/detect.py) decides which 9 form a face.
    """
    cv2, np = _lazy_imports()
    from .detect import Square

    h, w = frame.shape[:2]
    area = h * w
    min_area, max_area = area * min_frac, area * max_frac

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edges + dilate to close sticker borders, then find contours.
    edges = cv2.Canny(gray, 40, 110)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    squares = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.08 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        if bh == 0:
            continue
        aspect = bw / float(bh)
        if not (0.7 <= aspect <= 1.4):  # roughly square
            continue
        cx, cy = x + bw / 2.0, y + bh / 2.0
        squares.append(Square(cx=cx, cy=cy, size=(bw + bh) / 2.0))
    return squares


def _grid_cell_centers(frame_w: int, frame_h: int, box: int) -> List[Tuple[int, int]]:
    """Return the 9 (x, y) pixel centers of a centered 3x3 grid.

    `box` is the full grid side length in pixels; cells are read in reading
    order (row-major), matching the facelet convention.
    """
    cx, cy = frame_w // 2, frame_h // 2
    step = box // 3
    # Cell centers sit at the MIDDLE of each 3x3 cell. The grid spans `box`
    # centered on (cx, cy), so cell centers are offset by one step from center.
    # (This must match the grid lines drawn in _draw_overlay, which start at
    # cx - 3*step/2 -- cell centers are at that start + step/2 + col*step.)
    start = -3 * step // 2
    centers = []
    for row in range(3):
        for col in range(3):
            x = cx + start + step // 2 + col * step
            y = cy + start + step // 2 + row * step
            centers.append((x, y))
    return centers


def _sample_cell(frame, center: Tuple[int, int], half: int) -> RGB:
    """Robustly sample a sticker color around `center`. Returns RGB.

    Logo/glare robust: a center sticker may have a printed LOGO right where we
    sample, and glossy stickers have specular highlights. We take the patch,
    drop the darkest and brightest pixels (logo marks and glare are the
    extremes), and return the per-channel MEDIAN of what remains -- so the
    sticker's true color dominates even when a logo sits dead center.
    """
    _, np = _lazy_imports()
    from .detect import trimmed_median_color

    # Detected cell centers are floats; NumPy slicing needs ints. Round, clamp
    # to the frame, and ensure a non-empty patch.
    fh, fw = frame.shape[:2]
    x, y = int(round(center[0])), int(round(center[1]))
    half = max(1, int(round(half)))
    x0, x1 = max(0, x - half), min(fw, x + half)
    y0, y1 = max(0, y - half), min(fh, y + half)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return (0.0, 0.0, 0.0)
    # Downsample large patches so the pure trimmed-median stays fast (9 patches
    # per frame). Patch is BGR; convert to RGB tuples for the shared helper.
    pix = patch.reshape(-1, 3)
    if len(pix) > 400:
        step = len(pix) // 400
        pix = pix[::step]
    rgb_pixels = [(float(p[2]), float(p[1]), float(p[0])) for p in pix]
    return trimmed_median_color(rgb_pixels)


def _draw_overlay(frame, cell_centers, box, half, prompt, captured_colors=None):
    cv2, _ = _lazy_imports()
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    step = box // 3
    # Grid lines.
    top_left = (cx - 3 * step // 2, cy - 3 * step // 2)
    for i in range(4):
        x = top_left[0] + i * step
        y = top_left[1] + i * step
        cv2.line(frame, (x, top_left[1]), (x, top_left[1] + box), (255, 255, 255), 1)
        cv2.line(frame, (top_left[0], y), (top_left[0] + box, y), (255, 255, 255), 1)
    # Sample squares.
    for i, (x, y) in enumerate(cell_centers):
        cv2.rectangle(frame, (x - half, y - half), (x + half, y + half),
                      (0, 255, 0), 1)
        if captured_colors is not None:
            r, g, b = captured_colors[i]
            cv2.rectangle(frame, (x - half, y - half), (x + half, y + half),
                          (int(b), int(g), int(r)), -1)
    # Prompt text.
    cv2.putText(frame, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "SPACE=capture  r=redo  q=quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def calibrate_centers(
    camera_index: int = 0, box_frac: float = 0.6
) -> Dict[str, RGB]:
    """Interactively capture the six center colors for calibration.

    Returns {face_letter: RGB}. Show each center sticker when prompted and
    press SPACE. This learns YOUR cube's actual colors under YOUR lighting.
    """
    cv2, _ = _lazy_imports()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")
    centers: Dict[str, RGB] = {}
    try:
        for face in FACE_NAMES:
            prompt = f"CALIBRATE: show {face} CENTER sticker, press SPACE"
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("failed to read camera frame")
                h, w = frame.shape[:2]
                box = int(min(w, h) * box_frac)
                single = [(w // 2, h // 2)]
                half = max(4, box // 12)
                disp = frame.copy()
                _draw_overlay(disp, single, box // 3, half, prompt)
                cv2.imshow("calibrate", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord(" "), 13):  # SPACE / ENTER
                    centers[face] = _sample_cell(frame, (w // 2, h // 2), half)
                    break
                if key in (ord("q"), 27):  # q / ESC
                    raise KeyboardInterrupt("calibration aborted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return centers


def calibrate_white(camera_index: int = 0, box_frac: float = 0.6) -> RGB:
    """Capture a single WHITE sample for the low-friction spin flow.

    Show the white (U) face centered in the overlay and press SPACE. The
    returned RGB is used by ColorClassifier.from_white_anchor to adapt the
    built-in standard-scheme references to your lighting. Assumes a BOY/Western
    cube; use calibrate_centers() if your cube uses a different scheme.
    """
    cv2, _ = _lazy_imports()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")
    prompt = "Show the WHITE face, press SPACE  (then spin from there)"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("failed to read camera frame")
            h, w = frame.shape[:2]
            box = int(min(w, h) * box_frac)
            half = max(4, box // 12)
            single = [(w // 2, h // 2)]
            disp = frame.copy()
            _draw_overlay(disp, single, box // 3, half, prompt)
            cv2.imshow("calibrate", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord(" "), 13):
                return _sample_cell(frame, (w // 2, h // 2), half)
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("calibration aborted")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def scan_faces(
    camera_index: int = 0,
    box_frac: float = 0.6,
    face_order: Sequence[str] = FACE_NAMES,
) -> Dict[str, List[RGB]]:
    """Interactively capture all six faces' 9-sticker color grids.

    Returns {face_letter: [9 RGB samples in reading order]}.
    """
    cv2, _ = _lazy_imports()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")
    faces: Dict[str, List[RGB]] = {}
    try:
        for face in face_order:
            prompt = f"Scan {_FACE_PROMPTS.get(face, face)}"
            last: Optional[List[RGB]] = None
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("failed to read camera frame")
                h, w = frame.shape[:2]
                box = int(min(w, h) * box_frac)
                half = max(4, box // 12)
                cells = _grid_cell_centers(w, h, box)
                disp = frame.copy()
                _draw_overlay(disp, cells, box, half, prompt, last)
                cv2.imshow("scan", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord(" "), 13):
                    last = [_sample_cell(frame, c, half) for c in cells]
                    faces[face] = last
                    break
                if key == ord("r") and last is not None:
                    last = None  # clear preview, keep scanning this face
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt("scan aborted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return faces


def _draw_spin_overlay(frame, cells, box, half, collector, just_got=None):
    """Overlay for the spin flow: live grid + collected-faces progress bar."""
    cv2, _ = _lazy_imports()
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    step = box // 3
    top_left = (cx - 3 * step // 2, cy - 3 * step // 2)
    for i in range(4):
        x = top_left[0] + i * step
        y = top_left[1] + i * step
        cv2.line(frame, (x, top_left[1]), (x, top_left[1] + box), (255, 255, 255), 1)
        cv2.line(frame, (top_left[0], y), (top_left[0] + box, y), (255, 255, 255), 1)
    for (x, y) in cells:
        cv2.rectangle(frame, (x - half, y - half), (x + half, y + half),
                      (0, 255, 0), 1)
    # Instruction + progress.
    cv2.putText(frame, "Slowly rotate the cube through all 6 faces",
                (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                cv2.LINE_AA)
    collected = set(collector.collected_faces())
    label = "Collected: "
    cv2.putText(frame, label, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    x0 = 20 + 110
    for face in FACE_NAMES:
        got = face in collected
        color = (0, 220, 0) if got else (90, 90, 90)
        cv2.putText(frame, face, (x0, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2, cv2.LINE_AA)
        x0 += 28
    if just_got:
        cv2.putText(frame, f"captured {just_got}!", (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "q/ESC = quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def spin_scan(
    classifier,
    camera_index: int = 0,
    box_frac: float = 0.6,
    steady_frames: int = 6,
):
    """Low-friction capture: rotate the cube; auto-detect each face by center.

    `classifier` must already be calibrated (see calibrate_centers). Returns
    {face_letter: [9 RGB samples]} once all six faces have been captured.
    """
    cv2, _ = _lazy_imports()
    from .spin import FaceCandidate, SpinCollector

    collector = SpinCollector(classifier=classifier, steady_frames=steady_frames)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")
    just_got = None
    just_got_ttl = 0
    try:
        while not collector.is_complete():
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("failed to read camera frame")
            h, w = frame.shape[:2]
            box = int(min(w, h) * box_frac)
            half = max(4, box // 12)
            cells = _grid_cell_centers(w, h, box)
            samples = [_sample_cell(frame, c, half) for c in cells]
            got = collector.offer(FaceCandidate(samples=samples))
            if got:
                just_got = got
                just_got_ttl = 20
            if just_got_ttl > 0:
                just_got_ttl -= 1
            else:
                just_got = None
            disp = frame.copy()
            _draw_spin_overlay(disp, cells, box, half, collector, just_got)
            cv2.imshow("spin-scan", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("spin scan aborted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return collector.faces()


def confirm_facelets(facelets: str, classifier=None) -> bool:
    """Show the detected cube as text and ask for a one-key confirm.

    Returns True to accept, False to re-scan. Printed to the console (no extra
    window) so it works even on minimal setups.
    """
    print("\nDetected cube (facelet string):")
    print(f"  {facelets}")
    print("Faces (U R F D L B), each read top-left to bottom-right:")
    for i, face in enumerate(FACE_NAMES):
        chunk = facelets[i * 9:(i + 1) * 9]
        print(f"  {face}: {chunk[0:3]} {chunk[3:6]} {chunk[6:9]}")
    while True:
        resp = input("Accept this scan? [y]es / [n]o re-scan: ").strip().lower()
        if resp in ("y", "yes", ""):
            return True
        if resp in ("n", "no"):
            return False


# Short names for a compact on-screen color readout.
_FACE_LABEL = {"U": "Wht", "D": "Yel", "F": "Grn", "B": "Blu", "R": "Red", "L": "Org"}


def _draw_live_classification(frame, cells, half, classifier):
    """Label each sampled cell with the color it currently classifies as.

    This is the diagnostic that makes color problems visible: you see, live,
    what the app thinks each sticker is, so misreads are obvious on screen.
    """
    cv2, _ = _lazy_imports()
    for (x, y) in cells:
        sample = _sample_cell(frame, (x, y), half)
        face = classifier.classify(sample)
        cv2.putText(frame, _FACE_LABEL.get(face, face), (x - 18, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, _FACE_LABEL.get(face, face), (x - 18, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def spin_scan_session(
    camera_index: int = 0,
    box_frac: float = 0.6,
    steady_frames: int = 6,
    debug: bool = False,
):
    """One continuous window: show white -> (no window switch) -> spin all faces.

    Phase A (white): prompts for the white face; SPACE builds the classifier
    from that single sample (ColorClassifier.from_white_anchor).
    Phase B (spin): immediately, in the SAME window, collect all six faces by
    slowly rotating the cube; faces are auto-detected by center color.

    Returns (classifier, {face: 9 samples}). With debug=True, each sticker is
    labeled live with its classified color and the per-face samples are printed
    when captured, so color-detection issues are easy to diagnose.
    """
    cv2, _ = _lazy_imports()
    from .colors import ColorClassifier
    from .spin import FaceCandidate, SpinCollector

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")

    win = "cube scan"
    classifier = None
    collector = None
    just_got = None
    just_got_ttl = 0
    # Live-adjustable grid size so the 9 sample points land on the 9 stickers.
    box_scale = box_frac
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("failed to read camera frame")
            h, w = frame.shape[:2]
            box = int(min(w, h) * box_scale)
            half = max(3, box // 14)  # sample patch stays inside a sticker
            disp = frame.copy()

            if classifier is None:
                # ---- Phase A: white calibration ----
                center = (w // 2, h // 2)
                _draw_overlay(
                    disp, [center], box // 3, half,
                    "Show WHITE, SPACE to set.  +/- resize grid to your cube",
                )
                cv2.imshow(win, disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("+"), ord("=")):
                    box_scale = min(0.95, box_scale + 0.03)
                elif key in (ord("-"), ord("_")):
                    box_scale = max(0.25, box_scale - 0.03)
                if key in (ord(" "), 13):
                    white = _sample_cell(frame, center, half)
                    classifier = ColorClassifier.from_white_anchor(white)
                    collector = SpinCollector(
                        classifier=classifier, steady_frames=steady_frames
                    )
                    if debug:
                        print(f"[debug] white sample RGB = "
                              f"({white[0]:.0f}, {white[1]:.0f}, {white[2]:.0f})")
                elif key in (ord("q"), 27):
                    raise KeyboardInterrupt("scan aborted")
                continue

            # ---- Phase B: spin collection (same window) ----
            cells = _grid_cell_centers(w, h, box)
            samples = [_sample_cell(frame, c, half) for c in cells]
            got = collector.offer(FaceCandidate(samples=samples))
            if got:
                just_got = got
                just_got_ttl = 20
                if debug:
                    rgb = [f"({s[0]:.0f},{s[1]:.0f},{s[2]:.0f})" for s in samples]
                    print(f"[debug] captured {got}: {rgb}")
            if just_got_ttl > 0:
                just_got_ttl -= 1
            else:
                just_got = None

            _draw_spin_overlay(disp, cells, box, half, collector, just_got)
            if debug:
                _draw_live_classification(disp, cells, half, classifier)
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("+"), ord("=")):
                box_scale = min(0.95, box_scale + 0.03)
            elif key in (ord("-"), ord("_")):
                box_scale = max(0.25, box_scale - 0.03)
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("scan aborted")
            if collector.is_complete():
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return classifier, collector.faces()


def _draw_detection(frame, centers, classifier=None, debug=False):
    """Draw the 9 detected sticker centers (and classified colors in debug)."""
    cv2, _ = _lazy_imports()
    for i, (x, y) in enumerate(centers):
        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
        if debug and classifier is not None:
            sample = _sample_cell(frame, (x, y), 6)
            face = classifier.classify(sample)
            cv2.putText(frame, _FACE_LABEL.get(face, face), (x - 16, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, _FACE_LABEL.get(face, face), (x - 16, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)


def spin_scan_auto(
    camera_index: int = 0,
    steady_frames: int = 6,
    debug: bool = False,
):
    """Automatic face detection: no box to fit the cube into.

    One window. Phase A: hold the WHITE face up until it's detected (9 squares
    found), press SPACE to calibrate. Phase B: slowly rotate the cube; each face
    is auto-detected and auto-captured by its center color when steady.

    Detection finds the 9 sticker squares in the frame (vision/detect.py orders
    them), so sampling always lands on the stickers -- removing the alignment
    problem the fixed box had. Returns (classifier, {face: 9 samples}).
    """
    cv2, _ = _lazy_imports()
    from .colors import ColorClassifier
    from .detect import detect_face_grid
    from .spin import FaceCandidate, SpinCollector

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")

    win = "cube scan (auto-detect)"
    classifier = None
    collector = None
    just_got = None
    just_got_ttl = 0
    patch_half = 6
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("failed to read camera frame")
            disp = frame.copy()
            face = _find_cube_face(frame)
            centers = _detect_face_centers(frame)
            detected = centers is not None
            # Scale the sampling patch to the detected sticker (cell) size.
            if face is not None:
                patch_half = max(4, int((face[2] / 3.0) * 0.3))

            # In debug, show the detection mask + the found cube square, so we
            # can see what the detector is keying on.
            if debug:
                cv2.imshow("detection mask", _cube_edge_mask(frame)[0])
                if face is not None:
                    cv2.circle(disp, (int(face[0]), int(face[1])),
                               int(face[2] / 2), (255, 0, 255), 2)

            if classifier is None:
                # ---- Phase A: white calibration via detection ----
                msg = ("WHITE face detected - press SPACE" if detected
                       else "Show WHITE face CLOSE to camera...")
                color = (0, 220, 0) if detected else (0, 200, 255)
                cv2.putText(disp, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            color, 2, cv2.LINE_AA)
                if detected:
                    _draw_detection(disp, centers)
                cv2.putText(disp, "q/ESC = quit", (20, disp.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                            cv2.LINE_AA)
                cv2.imshow(win, disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord(" "), 13) and detected:
                    # White = the brightest, most-neutral of the 9 samples. On a
                    # scrambled white face only the center is guaranteed white,
                    # so we must NOT median all 9 (that mixes random colors).
                    from .detect import pick_white_reference
                    samples = [_sample_cell(frame, c, patch_half) for c in centers]
                    white = pick_white_reference(samples)
                    classifier = ColorClassifier.from_white_anchor(white)
                    collector = SpinCollector(
                        classifier=classifier, steady_frames=steady_frames
                    )
                    if debug:
                        print(f"[debug] white sample RGB = "
                              f"({white[0]:.0f}, {white[1]:.0f}, {white[2]:.0f})")
                elif key in (ord("q"), 27):
                    raise KeyboardInterrupt("scan aborted")
                continue

            # ---- Phase B: spin collection via detection ----
            if detected:
                samples = [_sample_cell(frame, c, patch_half) for c in centers]
                got = collector.offer(FaceCandidate(samples=samples))
                if got:
                    just_got = got
                    just_got_ttl = 20
                    if debug:
                        rgb = [f"({s[0]:.0f},{s[1]:.0f},{s[2]:.0f})"
                               for s in samples]
                        print(f"[debug] captured {got}: {rgb}")
                _draw_detection(disp, centers, classifier, debug)

            if just_got_ttl > 0:
                just_got_ttl -= 1
            else:
                just_got = None

            # Progress bar.
            cv2.putText(disp, "Slowly rotate the cube through all 6 faces",
                        (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                        2, cv2.LINE_AA)
            got_set = set(collector.collected_faces())
            x0 = 20
            cv2.putText(disp, "Collected:", (x0, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1, cv2.LINE_AA)
            x0 += 105
            for face in FACE_NAMES:
                col = (0, 220, 0) if face in got_set else (90, 90, 90)
                cv2.putText(disp, face, (x0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            col, 2, cv2.LINE_AA)
                x0 += 26
            if not detected:
                cv2.putText(disp, "(no face detected - adjust angle/lighting)",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 170, 255), 1, cv2.LINE_AA)
            if just_got:
                cv2.putText(disp, f"captured {just_got}!",
                            (20, disp.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 220, 0), 2, cv2.LINE_AA)

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("scan aborted")
            if collector.is_complete():
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return classifier, collector.faces()
