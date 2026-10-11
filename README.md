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
- 🚧 **Vision** — interface boundary is defined (`vision/`). The OpenCV static
  scanner and the future live, camera-based turn detector plug into the same
  interfaces without changing the core.

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

Requires Python 3.10+.

```bash
# Trace a scramble from its facelet string (Kociemba order U R F D L B):
python -m app.main --facelets UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB

# TPS counter demo (playback-driven; same interface the live detector will use):
python -m app.main --demo-tps
```

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
2. 🚧 Static camera scanning (OpenCV): show each face → facelet string.
3. ⏳ Live, camera-based turn detection feeding the TPS counter in real time.
