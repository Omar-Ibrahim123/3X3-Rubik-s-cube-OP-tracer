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

    Uses the MEDIAN over the patch (not mean) so the dark grid lines between
    stickers, small specular highlights, or a stray edge don't drag the color.
    """
    _, np = _lazy_imports()
    x, y = center
    patch = frame[y - half : y + half, x - half : x + half]
    if patch.size == 0:
        return (0.0, 0.0, 0.0)
    med_bgr = np.median(patch.reshape(-1, 3), axis=0)
    b, g, r = (float(v) for v in med_bgr)
    return (r, g, b)


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
