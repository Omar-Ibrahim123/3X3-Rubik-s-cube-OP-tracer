"""One-window calibration + scan session.

Owns one OpenCV camera and one window for the entire workflow:
  calibrate six distinct labeled centers -> build perceptual palette -> scan.
"""

from __future__ import annotations


def calibrate_and_spin(camera_index=0, steady_frames=6, debug=False):
    """Run six-color calibration and spin scanning in one camera window.

    Returns ``(classifier, faces)``. OpenCV is imported lazily through capture.
    """
    import math

    from . import capture
    from .calibration import PaletteCalibrationTracker
    from .colors import FACE_NAMES, ColorClassifier
    from .spin import FaceCandidate, SpinCollector

    cv2, _ = capture._lazy_imports()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")

    labels = {
        "U": "WHITE (U)",
        "R": "RED (R)",
        "F": "GREEN (F)",
        "D": "YELLOW (D)",
        "L": "ORANGE (L)",
        "B": "BLUE (B)",
    }
    tracker = PaletteCalibrationTracker(stable_frames=steady_frames)
    classifier = None
    collector = None
    phase = "calibrate"
    window = "Rubik's Cube OP Tracer"
    just_captured = None
    just_captured_ttl = 0
    patch_half = 6

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("failed to read camera frame")

            display = frame.copy()
            centers = capture._detect_face_centers(frame)
            detected = centers is not None

            if detected:
                distances = [
                    math.hypot(
                        centers[i][0] - centers[j][0],
                        centers[i][1] - centers[j][1],
                    )
                    for i, j in ((0, 1), (1, 2), (3, 4), (4, 5))
                ]
                spacing = sorted(distances)[len(distances) // 2]
                patch_half = max(4, int(spacing * 0.28))
                capture._draw_detection(
                    display,
                    centers,
                    classifier if phase == "scan" else None,
                    debug and phase == "scan",
                )

            if debug:
                for x, y, width, height in capture._find_sticker_boxes(frame):
                    cv2.rectangle(
                        display,
                        (x, y),
                        (x + width, y + height),
                        (255, 0, 255),
                        1,
                    )

            if phase == "calibrate":
                center = (
                    capture._sample_center_sticker(frame, centers[4], patch_half)
                    if detected
                    else None
                )
                accepted = tracker.offer(center)
                expected = tracker.expected_face
                capture._draw_closer_guide(display, detected)

                if tracker.waiting_for_transition:
                    title = f"Turn to {labels[expected]}"
                elif expected is not None:
                    title = f"CALIBRATE: show {labels[expected]}"
                else:
                    title = "Calibration complete"
                cv2.putText(
                    display,
                    title,
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    tracker.last_message or "hold face steady inside target",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                done = " ".join(tracker.palette.keys()) or "none"
                cv2.putText(
                    display,
                    f"calibrated: {done}",
                    (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 0),
                    1,
                    cv2.LINE_AA,
                )

                if accepted and debug:
                    rgb = tracker.palette[accepted]
                    print(
                        f"[debug] calibrated {accepted}: "
                        f"({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f})"
                    )

                if tracker.complete:
                    classifier = ColorClassifier.from_palette(tracker.palette)
                    collector = SpinCollector(
                        classifier=classifier,
                        steady_frames=steady_frames,
                    )
                    if debug:
                        nearest = classifier.nearest_palette_pair()
                        print(
                            "[debug] palette complete; nearest pair "
                            f"{nearest[0]}/{nearest[1]} distance={nearest[2]:.2f}"
                        )
                    phase = "scan"

            else:
                if detected:
                    samples = capture._sample_face(frame, centers, patch_half)
                    accepted = collector.offer(FaceCandidate(samples=samples))
                    if accepted:
                        just_captured = accepted
                        just_captured_ttl = 20
                        if debug:
                            values = [
                                f"({r:.0f},{g:.0f},{b:.0f})"
                                for r, g, b in samples
                            ]
                            print(f"[debug] captured {accepted}: {values}")

                cv2.putText(
                    display,
                    "SCAN: slowly rotate through all six faces",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                done = set(collector.collected_faces())
                x = 20
                cv2.putText(
                    display,
                    "collected:",
                    (x, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                x += 105
                for face in FACE_NAMES:
                    color = (0, 220, 0) if face in done else (90, 90, 90)
                    cv2.putText(
                        display,
                        face,
                        (x, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    x += 27
                if not detected:
                    capture._draw_closer_guide(display, False)
                    cv2.putText(
                        display,
                        "hold cube closer and square to camera",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )
                if just_captured_ttl > 0:
                    just_captured_ttl -= 1
                    cv2.putText(
                        display,
                        f"captured {just_captured}",
                        (20, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    just_captured = None

                if collector.is_complete():
                    return classifier, collector.faces()

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("scan aborted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
