# 3x3 Rubik's Cube — Old Pochmann Blindfolded Tracer

Use computer vision to read a scrambled 3x3 cube and return an **Old Pochmann
(OP)** blindfolded trace for corners and edges, using the **Speffz** lettering
scheme — plus a live **turns-per-second (TPS)** counter during the solve.

## Status

This repository currently contains the **pure-logic core**, fully tested:

- ✅ **Cube model** — parses a standard 54-char facelet string into a
  Speffz-level permutation.
- ✅ **Speffz lettering** — single source of truth for sticker ↔ letter maps,
  piece-companion tables, and OP buffer constants.
- ✅ **OP tracer** — produces separate corner and edge letter-pair traces, with
  cycle-breaking, in-place twist/flip handling, and parity detection.
- ✅ **TPS counter** — `total turns / elapsed time`, driven by a simple
  `record_turn()` signal.
- ✅ **Vision (static scan)** — automatic cube-face detection (finds the 9
  sticker squares and orders them into a 3×3 grid) feeding a white-balanced
  HSV color classifier, assembling a validated facelet string. Detection and
  color logic are dependency-free and unit-tested; OpenCV is only needed to run
  a live scan. Fixed-box and manual flows remain as fallbacks. The future live
  turn detector plugs into the same `FaceletSource` seam.

## Architecture

```
cube/      cube state model + Speffz lettering (pure logic)
solver/    Old Pochmann trace generation (corners + edges)
tps/       turns-per-second counter
vision/    camera input boundary (static scan now, live detection later)
app/       CLI tying it together
tests/     test suite + a self-contained move engine for scrambles
```

The design keeps a clean seam for the future **live camera-based turn
detection**: everything downstream consumes a 54-char facelet string and a
`record_turn()` turn signal, so the live detector can be added without touching
the solver or TPS logic.

## Lettering & buffers

Speffz scheme, face order **U, L, F, R, B, D**, letters A–X assigned clockwise
per face (corners and edges independently). Standard OP buffers:
corner = `A` (UBL), edge = `B` (UR). Buffers are named constants in
`cube/speffz.py` so they can be changed in one place.

Reference: [Speffz on the Speedsolving wiki](https://www.speedsolving.com/wiki/index.php/Speffz).
*(Scheme described there; implementation here is original.)*

## Usage

Requires Python 3.10+. The core (trace + TPS) has **no dependencies**.

```bash
# Trace a scramble from its facelet string (Kociemba order U R F D L B):
python -m app.main --facelets UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB

# TPS counter demo (playback-driven; same interface the live detector will use):
python -m app.main --demo-tps
```

### Scanning a real cube with your webcam

The scanner needs OpenCV + numpy (only for scanning, not the core):

```bash
pip install -r requirements-scan.txt      # opencv-python, numpy
python -m app.main --scan                  # add --camera 1 if the wrong cam opens
```

**Default: spin-to-scan (low friction).** Assumes a standard **BOY/Western**
color scheme (white opposite yellow, green opposite blue, red opposite orange).

The cube face is **detected automatically** — there's no box to line the cube
up with. It all happens in one window:

1. **Show white** — hold the **white** face toward the camera until it's
   detected (its 9 stickers get circled), then press **SPACE**. That single
   sample anchors the standard-scheme colors to your lighting.
2. **Spin** — slowly rotate the cube; each face is detected and captured
   automatically once held steady. The progress bar lights up U R F D L B as
   they're collected — no key presses, no alignment.
3. **Confirm** — it prints the detected cube; press **y** to accept (or **n** to
   re-scan). Then it prints the OP trace.

How detection works: each frame, the app finds vivid/bright color regions (the
stickers), keeps the sticker-sized, roughly-square, solid ones, and groups 9 of
them into a 3×3 face — robust to glossy/stickerless cubes with rounded stickers.
**Hold the cube fairly close** so it fills a good part of the frame; that makes
detection far more reliable (and crowds out a busy background).

Add `--debug` to see the detection mask and candidate regions live, which helps
if detection is struggling with your lighting.

**Fallbacks** if auto-detection struggles with your lighting/background:

```bash
python -m app.main --scan-box      # fixed 3x3 box you size with +/- keys
python -m app.main --scan-manual   # show each face and press SPACE
```

**Diagnosing color issues:** add `--debug` to any scan to label each sticker
live with its detected color and print the raw RGB samples per face:

```bash
python -m app.main --scan --debug
```

**Backup: manual scan** (more robust under poor/uneven lighting, and works
with any color scheme since it calibrates all six centers):

```bash
python -m app.main --scan-manual
```

Calibrate by showing each center, then show each face in turn and press
**SPACE** to capture it (controls: **SPACE** capture · **r** redo · **q/ESC**
quit).

Tips: even, diffuse lighting; avoid glare on the stickers; keep the face
square-on and filling the grid. If the spin flow struggles with your lighting,
`--scan-manual` is the reliable fallback. Recalibrate under the same light you
scan in.

Example output for a scramble:

```
Corners: OP JG IT B
Edges:   VH JD FA GN CW K  [PARITY]
```

## Facelet string format

54 characters in Kociemba face order — `U` ×9, `R` ×9, `F` ×9, `D` ×9, `L` ×9,
`B` ×9 — each face read left-to-right, top-to-bottom, with sticker 5 the
center. The vision layer produces exactly this string.

## Development

```bash
pip install pytest
python -m pytest tests/ -q
```

The tracer's correctness is verified by an independent cycle-decomposition
check and coverage invariants (`tests/test_core.py`), not just well-formedness.

## Roadmap

1. ✅ Pure-logic core (cube model, Speffz, OP tracer, TPS counter).
2. ✅ Static camera scanning (OpenCV): calibrate → show each face → facelet
   string → validate → trace.
3. ⏳ Live, camera-based turn detection feeding the TPS counter in real time.

## Credits

Cube-face detection (finding the sticker squares and locating the one whose
3×3 neighbor grid is complete) is adapted from the
[Qbr project](https://github.com/kkoomen/qbr) by Kim Koomen, used under the MIT
License. The adaptation lives in `vision/detect.py`
(`find_cube_by_neighbors`, `is_sticker_square`) and `vision/capture.py`
(`_find_sticker_boxes`).
