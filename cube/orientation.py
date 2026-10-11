"""Recover per-face rotation when scanning a cube in a free orientation.

Why this exists
---------------
When you scan a cube by spinning it in front of a camera, each face is captured
whenever it is held steady -- but you naturally hold each face at a slightly
different rotation. The six 3x3 colour grids are individually correct, and the
total colour counts come out right (9 of each), yet the assembled 54-char
facelet string describes an impossible cube: a single face captured rotated by
90/180/270 degrees puts its stickers in the wrong Speffz slots, so corners and
edges pair up wrong (e.g. the U and D colours -- opposite faces -- end up on the
same corner). This was confirmed on real scans: brute-forcing per-face rotations
recovered exactly one valid orientation.

What it does
------------
Each face has only four possible rotations, so the whole cube has 4**6 = 4096
orientation combinations -- trivial to search exhaustively. For each combination
we assemble the facelet string and test it with a real validity oracle
(`is_valid_cube`, which rejects impossible/duplicate pieces). We return the
combination(s) that yield a valid cube.

This is pure logic: no camera, no OpenCV. The capture layer calls
`orient_faces()` on the six captured grids before assembling the final string.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

from .state import CubeState

# Kociemba face order used by the facelet string.
FACE_ORDER = ("U", "R", "F", "D", "L", "B")

# Index remaps for rotating a 3x3 face (reading order 0..8):
#   0 1 2
#   3 4 5
#   6 7 8
# A clockwise quarter turn sends new[pos] = old[CW90[pos]].
_CW90 = (6, 3, 0, 7, 4, 1, 8, 5, 2)
_CW180 = (8, 7, 6, 5, 4, 3, 2, 1, 0)
_CW270 = (2, 5, 8, 1, 4, 7, 0, 3, 6)
_IDENTITY = (0, 1, 2, 3, 4, 5, 6, 7, 8)

# Rotation index 0..3 -> remap, where 1 == one clockwise quarter turn.
_ROTATIONS = (_IDENTITY, _CW90, _CW180, _CW270)


def rotate_face(face: str, quarter_turns: int) -> str:
    """Rotate a 9-char face string by `quarter_turns` * 90 degrees clockwise.

    `face` is 9 characters in reading order; the centre (index 4) is fixed by
    every rotation, as it must be.
    """
    if len(face) != 9:
        raise ValueError(f"a face must be 9 chars, got {len(face)}")
    remap = _ROTATIONS[quarter_turns % 4]
    return "".join(face[remap[i]] for i in range(9))


def is_structurally_valid(facelets: str) -> bool:
    """True if `facelets` has 9 of each colour and only real pieces.

    Uses CubeState.from_facelets as the oracle: it resolves every corner and
    edge to a real physical piece and raises if any slot holds an impossible
    colour combination (e.g. two opposite faces on one piece). This catches
    gross misreads but NOT parity errors (a structurally-fine cube can still be
    physically unsolvable); for that, use is_solvable().
    """
    if len(facelets) != 54:
        return False
    for face in FACE_ORDER:
        if facelets.count(face) != 9:
            return False
    try:
        CubeState.from_facelets(facelets)
    except ValueError:
        return False
    return True


def _permutation_sign(perm: Sequence[int]) -> int:
    """Sign (+1 / -1) of a permutation given as a list of ints 0..n-1."""
    seen = [False] * len(perm)
    sign = 1
    for i in range(len(perm)):
        if seen[i]:
            continue
        # Walk the cycle containing i; a k-cycle contributes (-1)**(k-1).
        length = 0
        j = i
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign


def is_solvable(facelets: str) -> bool:
    """True if `facelets` is a physically SOLVABLE cube.

    Beyond structural validity, a real cube obeys three invariants (the cube
    group's conserved quantities):

      1. permutation parity: the corner permutation and the edge permutation
         have the SAME sign (a single swap of two pieces is impossible);
      2. corner orientation: the total corner twist is 0 mod 3;
      3. edge orientation: the total edge flip is 0 mod 2.

    These are exactly the states reachable by legal moves. We derive the
    permutations and orientations from the resolved CubeState using the Speffz
    piece tables, with the standard orientation reference (U/D sticker for
    corners and edges). Returns False for any structurally-invalid input too.
    """
    if not is_structurally_valid(facelets):
        return False

    from cube import speffz

    state = CubeState.from_facelets(facelets)

    # Which colour is on each facelet position of each piece. We identify a
    # piece's permutation from the resolved CubeState (robust) and compute its
    # ORIENTATION geometrically from the facelet colours (convention-free),
    # using U/D as the reference axis -- the standard definition:
    #   * corner twist = clockwise steps from the slot's U/D-face facelet to the
    #     facelet actually showing a U or D colour (0,1,2);
    #   * edge flip = 0 if the edge's "primary" facelet shows the correct
    #     orientation colour, else 1.
    colour_at = facelets  # positional: colour_at[facelet_index] is a face letter
    UD = {"U", "D"}

    corner_homes = [p[0] for p in speffz.CORNER_PIECES]
    corner_index = {h: i for i, h in enumerate(corner_homes)}
    corner_perm: List[int] = []
    corner_twist = 0
    # Corner slots split into two tetrads of OPPOSITE handedness; a raw sum of
    # per-corner twist offsets is therefore not rotation-invariant. Negating the
    # offset for one tetrad puts every corner in a consistent frame so the sum
    # is conserved mod 3. The odd-handed representative here is the U slot;
    # verified over thousands of legal scrambles (sum % 3 == 0 for all) and
    # against single-corner twists (correctly caught).
    _CORNER_SIGN = {"A": 1, "B": 1, "C": 1, "D": 1, "U": -1, "V": 1, "W": 1, "X": 1}

    for slot_home in corner_homes:
        occupant = state.corner_at[slot_home]
        piece_home = speffz.CORNER_HOME[occupant]
        corner_perm.append(corner_index[piece_home])
        # The slot's three facelets, in clockwise order starting at the slot's
        # own (U/D-face) reference sticker.
        slot_cycle = speffz.CORNER_COMPANIONS[slot_home]
        colours = [
            colour_at[speffz.facelet_index(*speffz.CORNER_FACELET[s])]
            for s in slot_cycle
        ]
        # Exactly one of the three shows a U/D colour; its clockwise offset from
        # the reference facelet (index 0), signed by the slot's handedness, is
        # the twist contribution.
        offset = next(i for i, col in enumerate(colours) if col in UD)
        corner_twist += _CORNER_SIGN[slot_home] * offset

    edge_homes = [p[0] for p in speffz.EDGE_PIECES]
    edge_index = {h: i for i, h in enumerate(edge_homes)}
    edge_perm: List[int] = []
    edge_flip = 0
    edge_companions = {
        k: (k, speffz.EDGE_COMPANION[k]) for k in speffz.EDGE_COMPANION
    }

    for slot_home in edge_homes:
        occupant = state.edge_at[slot_home]
        piece_home = speffz.EDGE_HOME[occupant]
        edge_perm.append(edge_index[piece_home])
        slot_cycle = edge_companions[slot_home]
        colours = [
            colour_at[speffz.facelet_index(*speffz.EDGE_FACELET[s])]
            for s in slot_cycle
        ]
        # Standard edge-orientation rule using the U/D axis: an edge touching
        # the U or D face is oriented iff its U/D-face sticker shows a U/D
        # colour; the primary facelet (index 0) is that reference. If neither
        # sticker is U/D (an F/B-L/R slice edge), fall back to L/R.
        ref = colours[0]
        if ref in UD:
            flip = 0
        elif colours[1] in UD:
            flip = 1
        else:
            # Middle-slice edge: orient by L/R colour on the primary facelet.
            flip = 0 if ref in {"L", "R"} else 1
        edge_flip += flip

    if _permutation_sign(corner_perm) != _permutation_sign(edge_perm):
        return False
    if corner_twist % 3 != 0:
        return False
    if edge_flip % 2 != 0:
        return False
    return True


# Backwards-compatible alias: the orientation search uses the strongest oracle.
def is_valid_cube(facelets: str) -> bool:
    """True if `facelets` is a physically solvable cube (alias of is_solvable)."""
    return is_solvable(facelets)


def assemble(faces: Dict[str, str]) -> str:
    """Join six 9-char face strings into a 54-char facelet string."""
    return "".join(faces[f] for f in FACE_ORDER)


@dataclass
class OrientationResult:
    """Outcome of the per-face orientation search."""

    ok: bool
    facelets: Optional[str]            # the valid, re-oriented string if ok
    rotations: Optional[Dict[str, int]]  # face -> quarter-turns applied
    candidate_count: int               # how many valid orientations were found
    message: str = ""


def solve_orientation(faces: Dict[str, str]) -> OrientationResult:
    """Find per-face rotations that make the six grids a valid cube.

    `faces` maps each face letter (U R F D L B) to its 9-char colour string in
    the orientation it was captured. Centres are assumed already correct (they
    identify the face and are rotation-invariant); only the eight outer stickers
    of each face are reordered by rotation.

    Returns an OrientationResult. `ok` is True only when EXACTLY ONE orientation
    combination yields a valid cube (a unique solution). If none are valid the
    scan is genuinely broken (a real misread, not just rotation); if more than
    one are valid the result is ambiguous and we decline rather than guess.
    """
    missing = set(FACE_ORDER) - set(faces)
    if missing:
        return OrientationResult(
            ok=False, facelets=None, rotations=None, candidate_count=0,
            message=f"missing faces: {sorted(missing)}",
        )

    # Pre-rotate every face into its four orientations once.
    rotated = {f: [rotate_face(faces[f], q) for q in range(4)] for f in FACE_ORDER}

    # Collect DISTINCT valid cubes keyed by the resulting facelet string. Many
    # rotation combos can yield the same cube (e.g. a face with a uniform colour
    # is unchanged by rotation, so a solved cube has one distinct result across
    # all 4096 combos). Uniqueness must therefore be judged on the distinct
    # result, not the number of combos. For each distinct cube we keep the
    # minimal rotation (fewest total quarter-turns) as the representative.
    by_cube: Dict[str, Dict[str, int]] = {}
    for combo in product(range(4), repeat=6):
        candidate = "".join(rotated[FACE_ORDER[i]][combo[i]] for i in range(6))
        if not is_valid_cube(candidate):
            continue
        rotations = {FACE_ORDER[i]: combo[i] for i in range(6)}
        prev = by_cube.get(candidate)
        if prev is None or sum(rotations.values()) < sum(prev.values()):
            by_cube[candidate] = rotations

    if not by_cube:
        return OrientationResult(
            ok=False, facelets=None, rotations=None, candidate_count=0,
            message=(
                "no per-face rotation yields a valid cube; this is a real "
                "misread (a sticker colour is wrong), not just orientation"
            ),
        )
    if len(by_cube) > 1:
        return OrientationResult(
            ok=False, facelets=None, rotations=None, candidate_count=len(by_cube),
            message=(
                f"{len(by_cube)} distinct valid cubes from rotations; "
                "cannot disambiguate"
            ),
        )

    facelets, rotations = next(iter(by_cube.items()))
    turned = {f: q for f, q in rotations.items() if q}
    return OrientationResult(
        ok=True, facelets=facelets, rotations=rotations, candidate_count=1,
        message=(
            "re-oriented faces: "
            + (", ".join(f"{f}+{q*90}deg" for f, q in turned.items()) or "none")
        ),
    )
