"""CLI entry point for the OP blindfolded tracer.

Current capabilities (pure-logic core):
  * Given a 54-char facelet string, print the Old Pochmann corner and edge
    trace (Speffz letter-pairs) plus a parity flag.
  * A simple TPS-counter demo driven by solution playback, showing how the
    live camera-based turn detector will plug into the same interface later.

Usage:
    python -m app.main --facelets <54-char-string>
    python -m app.main --demo-tps

The camera-based static scanner and (future) live turn detection live in the
`vision` package and feed the same CubeState / TPSCounter interfaces.
"""

from __future__ import annotations

import argparse
import sys
import time

from cube.state import CubeState
from solver import trace
from tps import TPSCounter


def cmd_trace(facelets: str) -> int:
    try:
        state = CubeState.from_facelets(facelets)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = trace(state)
    print(result)
    print()
    print(f"Corner targets: {len(result.corners)}  "
          f"Edge targets: {len(result.edges)}  "
          f"Parity: {'yes' if result.parity else 'no'}")
    return 0


def cmd_demo_tps() -> int:
    """Demonstrate the TPS counter by 'playing back' a short move sequence.

    This is scenario (2) from the design discussion: counting playback turns.
    The future live detector would call counter.record_turn() from the camera
    pipeline instead of this loop, using the exact same counter.
    """
    moves = "R U R' U R U2 R'".split()
    counter = TPSCounter()
    counter.start()
    print("Playing back solution, counting TPS live:\n")
    for mv in moves:
        time.sleep(0.15)  # simulate the time a turn takes
        counter.record_turn()
        print(f"  {mv:3s}  ->  {counter.status()}")
    print(f"\nFinal: {counter.stop():.2f} TPS over {counter.elapsed:.2f}s "
          f"({counter.turns} turns)")
    return 0


def cmd_scan(camera_index: int, manual: str, debug: bool) -> int:
    """Scan a physical cube with the webcam and print its OP trace.

    Default (auto): the cube face is detected automatically -- hold white up,
    then slowly rotate through all six faces; no box to align.
    --scan-box: fixed 3x3 box you size with +/- (fallback).
    --scan-manual: calibrate centers, show each face, press SPACE (bad light).
    With --debug, each sticker is labeled live with its classified color and
    per-face RGB samples are printed, to diagnose color issues.
    Requires OpenCV locally (pip install -r requirements-scan.txt).
    """
    from vision import StaticScanner  # lazy: only needs OpenCV here

    scanner = StaticScanner(camera_index=camera_index, debug=debug)
    try:
        if manual == "manual":
            facelets = scanner.read_facelets_manual()
        elif manual == "box":
            facelets = scanner.read_facelets_box()
        else:
            facelets = scanner.read_facelets()
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nscan aborted.", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\nScanned facelets: {facelets}\n")
    return cmd_trace(facelets)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Old Pochmann BLD cube tracer")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--facelets",
        help="54-char facelet string (Kociemba order U R F D L B)",
    )
    g.add_argument(
        "--scan", action="store_true",
        help="scan by auto-detecting the cube face while you spin it, then trace",
    )
    g.add_argument(
        "--scan-box", action="store_true",
        help="fallback scan using a fixed 3x3 box you size with +/-",
    )
    g.add_argument(
        "--scan-manual", action="store_true",
        help="backup scan: show each face and press SPACE (robust in bad light)",
    )
    g.add_argument(
        "--demo-tps", action="store_true",
        help="run the TPS counter playback demo",
    )
    p.add_argument(
        "--camera", type=int, default=0, metavar="INDEX",
        help="webcam index for scanning (default 0; try 1 if you have multiple)",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="label each sticker live with its detected color + print RGB "
             "samples (use to diagnose color-detection issues)",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.demo_tps:
        return cmd_demo_tps()
    if args.scan or args.scan_box or args.scan_manual:
        mode = "manual" if args.scan_manual else "box" if args.scan_box else "auto"
        return cmd_scan(args.camera, manual=mode, debug=args.debug)
    return cmd_trace(args.facelets)


if __name__ == "__main__":
    raise SystemExit(main())
