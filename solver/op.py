"""Old Pochmann (OP) blindfolded trace generation.

Given a CubeState, produce the ordered list of Speffz "targets" for corners
and edges independently. A target is the sticker you shoot the buffer to; the
sequence of targets is the memo a blindfolded solver executes.

Algorithm (pure permutation cycle-tracing):

  1. Start at the buffer. Look at the piece in the buffer slot; the sticker it
     carries tells you the next target. Emit it, move there, repeat.
  2. When the chain returns to the buffer's piece, the cycle is closed.
  3. Break into a new cycle: pick the first still-unsolved, unvisited sticker
     (in canonical A..X order), emit it, and continue tracing.
  4. In-place mis-oriented pieces (twisted corners / flipped edges) are
     permutation fixed-points; a second pass emits the short cycles that
     re-orient them.

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
    """
    targets: List[str] = []
    buffer_piece = home[buffer]
    # Pieces whose cycle we've already handled (by home letter).
    solved_pieces = set()

    def is_solved(pos: str) -> bool:
        """A position is solved iff its occupying sticker is exactly itself."""
        return at[pos] == pos

    # A single orbit has at most 24 stickers; this caps any pathological walk
    # (e.g. a malformed state) so tracing can never loop forever.
    max_steps = 100

    # ---- Main walk: follow the cycle(s) containing the buffer. ----
    current = buffer
    steps = 0
    while True:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("trace did not terminate; malformed cube state")
        occupant = at[current]  # sticker now sitting in `current`
        # Where does that sticker belong? Shoot to that exact sticker.
        target = occupant
        if home[target] == buffer_piece:
            # Chain has returned to the buffer piece: cycle closed.
            solved_pieces.add(buffer_piece)
            break
        targets.append(target)
        solved_pieces.add(home[target])
        # Continue from the position this piece came from: move to the slot
        # the target sticker names.
        current = target

    # ---- Break into remaining cycles. ----
    # Canonical scan order: A..X by position letter.
    all_positions = sorted(companions.keys())
    while True:
        start = None
        for pos in all_positions:
            piece = home[pos]
            if piece in solved_pieces:
                continue
            if is_solved(pos):
                # Permutation fixed-point; handled in orientation pass unless
                # the whole piece is already solved.
                continue
            start = pos
            break
        if start is None:
            break
        # Break-in: emit the entering sticker, then trace its cycle.
        targets.append(start)
        current = start
        start_piece = home[start]
        steps = 0
        while True:
            steps += 1
            if steps > max_steps:
                raise RuntimeError(
                    "trace did not terminate; malformed cube state"
                )
            occupant = at[current]
            target = occupant
            if home[target] == start_piece:
                solved_pieces.add(start_piece)
                break
            targets.append(target)
            solved_pieces.add(home[target])
            current = target

    # ---- Orientation pass: in-place twisted/flipped pieces. ----
    for pos in all_positions:
        piece = home[pos]
        if piece in solved_pieces:
            continue
        if pos != piece:
            # Only act from the canonical home letter of the piece.
            continue
        occupant = at[pos]
        if occupant == pos:
            solved_pieces.add(piece)
            continue
        # Piece is at home but mis-oriented: emit its non-home stickers in
        # orientation order to twist/flip it in place.
        cycle = companions[pos]
        # occupant is one of the companions; emit the others following the
        # cyclic order so the piece is re-oriented.
        idx = cycle.index(occupant)
        rotated = cycle[idx:] + cycle[:idx]
        # Shoot to each non-home sticker needed to restore orientation.
        for sticker in rotated[1:]:
            targets.append(sticker)
        solved_pieces.add(piece)

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
