"""Old Pochmann (OP) blindfolded trace generation.

Given a CubeState, produce the ordered list of Speffz "targets" for corners
and edges independently. A target is the sticker you shoot the buffer to; the
sequence of targets is the memo a blindfolded solver executes.

Algorithm (OP shot simulation):

  OP solves a cube by repeatedly swapping the piece in the buffer slot with the
  piece in a target slot. We trace by *simulating those swaps on the state*:

  1. Look at the sticker in the buffer slot. That sticker IS the next target
     (shoot there). Emit it and apply the swap to our working copy of the
     state. Repeat while the buffer slot is unsolved.
  2. When the buffer slot is solved but pieces remain, "break in": shoot a
     still-unsolved slot (which parks the buffer piece there and brings a new
     piece into the buffer), then resume step 1.
  3. Simulating the swap means a piece that returns to its home slot *twisted
     or flipped* is still seen as unsolved by step 1, so it is automatically
     re-oriented by two further shots (shoot the home sticker, then the sticker
     sitting in the home slot). This yields the cubing-theory cost of exactly
     TWO targets for a flipped edge and TWO for a twisted corner, and needs no
     separate orientation pass.

Why simulate instead of a plain cycle walk? A plain permutation cycle walk
only tracks *which piece* lands home, not its final orientation, so pieces
cycled into place can be left twisted/flipped and never fixed, and a broken-in
cycle leaves the buffer threaded out of place. Simulating the actual swaps
makes orientation and buffer-threading fall out correctly, which an independent
executor (tests/test_op_execution.py) confirms by replaying the memo and
checking the cube ends solved.

Parity: if the number of edge targets is odd, OP parity is present and the
executor must apply the parity algorithm. We only report the flag here;
execution is a separate concern (and leaves room for the future live/vision
layers to consume the same trace).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from cube import speffz
from cube.state import CubeState

# Corner slots whose Speffz companion ring is listed in the opposite rotational
# sense from the others. A shot swaps the buffer slot's sticker-ring with the
# target slot's sticker-ring; for all corner pieces except this one the two
# rings align in companion order, but the D-F-L corner's companions are listed
# the other way round, so when that slot is on either side of a swap its ring
# must be read in reverse (keeping the pivot sticker fixed) for the swap to
# match the physical cube. This is a fixed geometric property of the Speffz
# corner lettering in cube/speffz.py (we do not modify that table); it is
# verified exhaustively by the ground-truth executor in
# tests/test_op_execution.py, which derives every swap from the move engine.
_REVERSED_CORNER_SLOTS = frozenset(speffz.CORNER_COMPANIONS["U"])  # {'U','L','G'}


@dataclass
class OPTrace:
    corners: List[str] = field(default_factory=list)
    edges: List[str] = field(default_factory=list)
    parity: bool = False

    def corner_pairs(self) -> List[str]:
        return _pairs(self.corners)

    def edge_pairs(self) -> List[str]:
        return _pairs(self.edges)

    def __str__(self) -> str:
        c = " ".join(self.corner_pairs()) or "(none)"
        e = " ".join(self.edge_pairs()) or "(none)"
        p = "  [PARITY]" if self.parity else ""
        return f"Corners: {c}\nEdges:   {e}{p}"


def _pairs(letters: List[str]) -> List[str]:
    """Group a flat target list into memo letter-pairs."""
    out = []
    for i in range(0, len(letters), 2):
        out.append("".join(letters[i : i + 2]))
    return out


def _shoot(
    at: Dict[str, str],
    buffer: str,
    target: str,
    home: Dict[str, str],
    companions: Dict[str, Tuple[str, ...]],
) -> None:
    """Apply one OP shot (swap buffer slot with target slot) to `at` in place.

    The buffer piece and the piece in `target`'s slot trade places. We pair the
    buffer slot's sticker-ring (companion ring, started at the buffer sticker)
    with the target slot's sticker-ring (started at `target`) element by
    element and exchange the stickers currently sitting in those positions.
    For corners the two rings align in companion order, except that a ring on a
    `_REVERSED_CORNER_SLOTS` slot must be read in reverse (pivot sticker fixed).
    """
    buf_ring = list(companions[buffer])
    tgt_slot_ring = companions[home[target]]
    i = tgt_slot_ring.index(target)
    tgt_ring = list(tgt_slot_ring[i:] + tgt_slot_ring[:i])
    if len(buf_ring) == 3:  # corner orientation correction (see note above)
        if home[buffer] in _REVERSED_CORNER_SLOTS:
            buf_ring = [buf_ring[0]] + buf_ring[1:][::-1]
        if home[target] in _REVERSED_CORNER_SLOTS:
            tgt_ring = [tgt_ring[0]] + tgt_ring[1:][::-1]
    snapshot = dict(at)
    for b_pos, t_pos in zip(buf_ring, tgt_ring):
        at[b_pos] = snapshot[t_pos]
        at[t_pos] = snapshot[b_pos]


def _trace_orbit(
    at: Dict[str, str],
    buffer: str,
    home: Dict[str, str],
    companions: Dict[str, Tuple[str, ...]],
) -> List[str]:
    """Trace one orbit (corners or edges) and return the target list.

    `at[pos]` = the specific sticker currently occupying position `pos`.
    `home[letter]` = canonical home letter of the piece a letter belongs to.
    `companions[letter]` = cyclic tuple of the piece's stickers starting at
    `letter`.

    We work on a copy of `at` and simulate the OP shots (see module docstring),
    so the caller's state is untouched and the emitted memo is exactly the
    sequence of swaps that drives the copy to solved.
    """
    work = dict(at)
    targets: List[str] = []
    buffer_piece = home[buffer]
    all_positions = sorted(companions.keys())
    # Stickers of the buffer piece itself; shooting to one of these is a no-op
    # (you cannot shoot the buffer at itself), so they are never valid targets.
    buffer_stickers = set(companions[buffer_piece])

    def first_breakin() -> str | None:
        """First slot holding a shootable (non-buffer) out-of-place piece."""
        for pos in all_positions:
            if work[pos] != pos and work[pos] not in buffer_stickers:
                return pos
        return None

    # A single orbit has 24 stickers; this caps any pathological walk (e.g. a
    # malformed state) so tracing can never loop forever.
    max_steps = 100
    steps = 0
    while True:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("trace did not terminate; malformed cube state")
        occupant = work[buffer]  # sticker now sitting in the buffer slot
        if occupant == buffer:
            # Buffer slot solved: break into a remaining cycle, or finish.
            start = first_breakin()
            if start is None:
                break
            target = start
        elif occupant not in buffer_stickers:
            # Normal shot: send the buffer piece to where its sticker belongs.
            target = occupant
        else:
            # Buffer piece is home but mis-oriented and nothing else can be
            # shot first; a legal cube state cannot reach this (orientation is
            # balanced across the orbit), but guard against it to avoid looping.
            start = first_breakin()
            if start is None:
                break
            target = start
        targets.append(target)
        _shoot(work, buffer, target, home, companions)

    return targets


def trace_corners(state: CubeState) -> List[str]:
    return _trace_orbit(
        state.corner_at,
        speffz.CORNER_BUFFER,
        speffz.CORNER_HOME,
        speffz.CORNER_COMPANIONS,
    )


def trace_edges(state: CubeState) -> List[str]:
    edge_companions = {
        k: (k, speffz.EDGE_COMPANION[k]) for k in speffz.EDGE_COMPANION
    }
    return _trace_orbit(
        state.edge_at,
        speffz.EDGE_BUFFER,
        speffz.EDGE_HOME,
        edge_companions,
    )


def trace(state: CubeState) -> OPTrace:
    corners = trace_corners(state)
    edges = trace_edges(state)
    return OPTrace(corners=corners, edges=edges, parity=(len(edges) % 2 == 1))
