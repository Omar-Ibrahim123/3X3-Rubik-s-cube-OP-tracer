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
    start_x = cx - step  # center of top-left cell
    start_y = cy - step
    centers = []
    for row in range(3):
        for col in range(3):
            centers.append((start_x + col * step, start_y + row * step))
    return centers


def _sample_cell(frame, center: Tuple[int, int], half: int) -> RGB:
    """Average the color in a small square around `center`. Returns RGB."""
    _, np = _lazy_imports()
    x, y = center
    patch = frame[y - half : y + half, x - half : x + half]
    # OpenCV frames are BGR; return RGB.
    mean_bgr = patch.reshape(-1, 3).mean(axis=0)
    b, g, r = (float(v) for v in mean_bgr)
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
