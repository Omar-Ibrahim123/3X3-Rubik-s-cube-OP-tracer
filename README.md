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
  sticker squares and orders them into a 3×3 grid) feeding a **CIEDE2000**
  perceptual color classifier (calibrated on all six centers, compared in CIE
  L\*a\*b\*), assembling a validated facelet string. Detection and
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

## Quick start — scan your cube

Everything runs locally; there is no website or server. You need **Python
3.10+** and a **webcam**.

```bash
# 1. Get the code
git clone https://github.com/Omar-Ibrahim123/3X3-Rubik-s-cube-OP-tracer.git
cd 3X3-Rubik-s-cube-OP-tracer

# 2. Install the scanning dependencies (OpenCV + numpy)
pip install -r requirements-scan.txt

# 3. Scan — calibrate, spin through all six faces, confirm, done
python -m app.main --scan          # add --camera 1 if the wrong camera opens
```

Everything happens **in the on-screen windows** — the terminal stays clean
(the only thing in your shell is the command you ran):

- When a face is read, the detected cube is shown as a coloured net. Press
  **Y** in that window to accept, or **N** to re-scan (Q/Esc also re-scans).
- After you accept, a **result window** opens with your Old Pochmann trace,
  **corners and edges in separate panels** (plus a parity banner if parity
  applies). Press any key in that window to close it.

> Tip: hold the cube **close** so it fills the on-screen box, use even lighting,
> and avoid glare. If auto-detection struggles, see the fallback scan modes
> below.

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

The cube face is **detected automatically** — there's no sampling grid to line
up manually. Calibration and scanning run in **one continuous camera window**
with one camera session, so exposure and white balance do not change between
phases:

1. **Calibrate** — show each face when prompted (U white, R red, F green,
   D yellow, L orange, B blue), filling the target box; each is captured when
   steady. After a capture, turn to the next requested face. The app refuses to
   reuse the same center under a second label and rejects near-duplicate palette
   colors. This learns your cube's **real** colors.
2. **Spin** — without switching windows, slowly rotate the cube through all six
   faces; each is detected and captured once held steady.
3. **Confirm (in the window)** — it shows the detected cube as a coloured net
   with a status line; press **Y** in the window to accept, or **N** to re-scan
   (Q/Esc also re-scan). Nothing is typed in the terminal.
4. **Result (in the window)** — a clean window pops up with your Old Pochmann
   trace, **corners and edges split** into separate panels, plus a parity
   banner when parity is present. Press any key in the window to close it. The
   terminal stays clean throughout.

Colors are classified by **CIEDE2000** perceptual distance in CIE L\*a\*b\*
against your calibrated palette — comparing your real colors to your real colors
rather than fixed thresholds, which is far more robust under colored lighting
(this is what finally fixed the orange↔red / green↔white confusion).

How detection works: each frame the app finds the individual sticker squares,
then locates the one whose 3×3 neighbor grid is complete (the center sticker);
its 9 neighbors are the face (adapted from Qbr — see Credits).

**Hold the cube CLOSE — fill the on-screen target box.** Detection is reliable
when the stickers are big in the frame; the app draws a corner-bracket box and
prompts "hold the cube closer" until a face is detected (box turns green). This
is the key to good scans.

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

For a `--scan`, this trace is shown in the **result window** (corners and edges
in separate panels, with a parity banner), and the terminal stays clean. The
`--facelets` command below prints the same trace to the terminal instead. The
trace looks like:

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

Color classification uses calibration on all six colors + perceptual
**CIEDE2000** distance in CIE L\*a\*b\*, the same approach Qbr uses. The LAB
conversion and CIEDE2000 routines in `vision/colorsci.py` are adapted from a
widely-used sRGB→LAB routine and from
[lovro-i/CIEDE2000](https://github.com/lovro-i/CIEDE2000), as used by Qbr.
