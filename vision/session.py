"""One-window calibration + scan session.

Owns one OpenCV camera and one window for the entire workflow:
  calibrate six distinct labeled centers -> build perceptual palette -> scan.
"""

from __future__ import annotations


def _nearest_calibrated(center, palette):
    """Nearest already-calibrated face to `center`, or (None, inf) if empty.

    Returns (face_letter, dE). Used for the live calibration readout so the user
    sees when the centre they're about to capture is actually a duplicate of a
    face already recorded (the red-as-white / blue-as-wrong failure mode).
    """
    from .colorsci import rgb_distance

    best = (None, float("inf"))
    for face, ref in palette.items():
        d = rgb_distance(center, ref)
        if d < best[1]:
            best = (face, d)
    return best


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
    # Key pressed on the PREVIOUS frame, consumed by the calibrate phase this
    # frame (one-frame latency is imperceptible and lets us keep a single
    # waitKey at the bottom of the loop). Calibration is SPACE-to-capture: the
    # user presses SPACE when the requested face is clearly in view, so the
    # program records exactly the intended centre instead of guessing when a
    # face has "settled" -- the root cause of the mis-sampled red/blue centres.
    pending_key = 255  # 255 == no key
    # Rolling buffer of the most recent detected centre samples. On SPACE we
    # commit the MEDIAN of this buffer, not the single current frame, so one
    # glitchy frame (motion blur, glare, a slightly-off detected centre) at the
    # instant of the key press can't poison a colour's reference and make the
    # gate reject every real instance of that face. This restores the
    # robustness of the old auto-steady calibration while keeping SPACE control.
    recent_centers: list = []
    RECENT_MAX = 6

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("failed to read camera frame")

            display = frame.copy()
            # Restrict detection to the on-screen guide box: ignore everything
            # outside it, which rejects background clutter and is much more
            # robust in awkward lighting. The user keeps the cube in the box.
            roi = capture._guide_roi(frame, capture.ROI_FRAC)
            centers = capture._detect_face_centers(frame, roi=roi)
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
                for x, y, width, height in capture._find_sticker_boxes(frame, roi=roi):
                    cv2.rectangle(
                        display,
                        (x, y),
                        (x + width, y + height),
                        (255, 0, 255),
                        1,
                    )

            if phase == "calibrate":
                # Sample the centre every frame for the LIVE readout, but only
                # RECORD it when the user presses SPACE (and a face is detected).
                center = (
                    capture._sample_center_sticker(frame, centers[4], patch_half)
                    if detected
                    else None
                )
                # Keep a short rolling buffer of detected centres; drop it when
                # the face isn't detected so stale samples can't leak across.
                if center is not None:
                    recent_centers.append(center)
                    if len(recent_centers) > RECENT_MAX:
                        recent_centers.pop(0)
                else:
                    recent_centers.clear()
                space_pressed = pending_key in (ord(" "), 13)
                accepted = None
                if space_pressed:
                    # Commit the MEDIAN of the recent steady samples, not the
                    # single current frame (robust to a one-frame glitch).
                    from .calibration import median_rgb
                    sample = median_rgb(recent_centers) if recent_centers else center
                    accepted = tracker.commit(sample)
                    if accepted is None and debug:
                        print(f"[debug] SPACE ignored: {tracker.last_message}")
                    elif accepted is not None:
                        recent_centers.clear()
                expected = tracker.expected_face
                capture._draw_closer_guide(display, detected)

                if tracker.waiting_for_transition:
                    title = f"Turn to {labels[expected]}, then press SPACE"
                elif expected is not None:
                    title = f"CALIBRATE: show {labels[expected]}, press SPACE"
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
                # Live readout: which already-calibrated colour the centre is
                # nearest to (or 'new' if far from all), so a mis-aligned/wrong
                # face is caught BEFORE it is recorded. Early on, with few
                # colours calibrated, this just shows the raw swatch.
                if detected and center is not None and expected is not None:
                    guess, guess_dist = _nearest_calibrated(center, tracker.palette)
                    swatch = (int(center[2]), int(center[1]), int(center[0]))
                    cv2.rectangle(display, (display.shape[1] - 60, 24),
                                  (display.shape[1] - 20, 64), swatch, -1)
                    if guess is not None and guess_dist < tracker.min_palette_distance:
                        readout = f"center matches already-calibrated {guess}!"
                        colour = (0, 165, 255)  # warn: duplicate of another face
                    else:
                        readout = f"center ready for {labels[expected]} - press SPACE"
                        colour = (0, 220, 0)
                    cv2.putText(
                        display, readout, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2, cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        display,
                        tracker.last_message or "line up the cube, then press SPACE",
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
                    elif collector.last_rejection:
                        # The coherence gate refused this reading. Show WHY, on
                        # screen and (throttled) in debug, so the user knows to
                        # re-show the face squarely instead of silently getting
                        # a corrupt scan.
                        cv2.putText(
                            display,
                            "re-show face squarely (mixed reading)",
                            (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 165, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        if debug:
                            tick = getattr(calibrate_and_spin, "_rej_tick", 0) + 1
                            calibrate_and_spin._rej_tick = tick
                            if tick % 15 == 0:
                                print(f"[debug] rejected: {collector.last_rejection}")
                                # Dump WHAT was sampled + each sticker's nearest
                                # calibrated color and distance, so a bad
                                # calibration (vs a bad frame) is diagnosable.
                                from .spin import face_coherence
                                coh = face_coherence(samples, classifier)
                                for i, (s, d) in enumerate(
                                    zip(samples, coh.sticker_distances)
                                ):
                                    near = classifier.classify(s)
                                    flag = "  <-- OFF" if d > 18 else ""
                                    print(
                                        f"    sticker {i}: "
                                        f"({s[0]:.0f},{s[1]:.0f},{s[2]:.0f}) "
                                        f"-> {near} dE={d:.1f}{flag}"
                                    )

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
            # Remember this frame's key for the calibrate phase to consume next
            # frame (SPACE-to-capture). Reset once no longer calibrating.
            pending_key = key if phase == "calibrate" else 255
    finally:
        cap.release()
        cv2.destroyAllWindows()
