"""Minimal facelet-level move engine, for tests only.

Applies face turns to a 54-char Kociemba facelet string so tests can build
real scrambles and verify tracer behavior against known states. This is NOT
part of the library surface; it exists so tests don't need an external cube
dependency.

Facelet order: U(0-8) R(9-17) F(18-26) D(27-35) L(36-44) B(45-53), each face
read 1..9:
    0 1 2
    3 4 5
    6 7 8
"""

from __future__ import annotations

SOLVED = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9

# Index of each face's 9 stickers.
_U = list(range(0, 9))
_R = list(range(9, 18))
_F = list(range(18, 27))
_D = list(range(27, 36))
_L = list(range(36, 45))
_B = list(range(45, 54))


def _rotate_face_cw(perm, face):
    """Rotate a face's own 9 stickers clockwise (in-place on perm map)."""
    # positions: 0 1 2 / 3 4 5 / 6 7 8  ->  CW
    a = face
    mapping = {
        a[0]: a[6], a[1]: a[3], a[2]: a[0],
        a[3]: a[7], a[4]: a[4], a[5]: a[1],
        a[6]: a[8], a[7]: a[5], a[8]: a[2],
    }
    return mapping


def _cycle(perm_map, cycle):
    """Add a 4-cycle (list of equal-length index groups) to a perm map."""
    # cycle is a list of index-lists; stickers move group[i] -> group[i+1].
    n = len(cycle)
    for i in range(n):
        src = cycle[i]
        dst = cycle[(i + 1) % n]
        for s, d in zip(src, dst):
            perm_map[d] = s


# Each move = full index permutation applied to the facelet string.
# Built by composing the face self-rotation with the 4 side-strip cycles.
def _build_move(face_indices, side_cycle):
    perm = {i: i for i in range(54)}
    perm.update(_rotate_face_cw(perm, face_indices))
    _cycle(perm, side_cycle)
    return perm


# Side strips for each clockwise face turn (the 3 adjacent stickers that cycle).
_MOVES = {
    "U": _build_move(_U, [
        [_F[0], _F[1], _F[2]],
        [_L[0], _L[1], _L[2]],
        [_B[0], _B[1], _B[2]],
        [_R[0], _R[1], _R[2]],
    ]),
    "D": _build_move(_D, [
        [_F[6], _F[7], _F[8]],
        [_R[6], _R[7], _R[8]],
        [_B[6], _B[7], _B[8]],
        [_L[6], _L[7], _L[8]],
    ]),
    "R": _build_move(_R, [
        [_U[2], _U[5], _U[8]],
        [_B[6], _B[3], _B[0]],
        [_D[2], _D[5], _D[8]],
        [_F[2], _F[5], _F[8]],
    ]),
    "L": _build_move(_L, [
        [_U[0], _U[3], _U[6]],
        [_F[0], _F[3], _F[6]],
        [_D[0], _D[3], _D[6]],
        [_B[8], _B[5], _B[2]],
    ]),
    "F": _build_move(_F, [
        [_U[6], _U[7], _U[8]],
        [_R[0], _R[3], _R[6]],
        [_D[2], _D[1], _D[0]],
        [_L[8], _L[5], _L[2]],
    ]),
    "B": _build_move(_B, [
        [_U[2], _U[1], _U[0]],
        [_L[0], _L[3], _L[6]],
        [_D[6], _D[7], _D[8]],
        [_R[8], _R[5], _R[2]],
    ]),
}


def _apply_perm(s, perm):
    out = list(s)
    for dst in range(54):
        out[dst] = s[perm[dst]]
    return "".join(out)


def apply_move(facelets, move):
    """Apply a single move token like 'R', "R'", 'R2'."""
    face = move[0]
    perm = _MOVES[face]
    times = 1
    if len(move) > 1:
        times = 3 if move[1] == "'" else 2 if move[1] == "2" else 1
    s = facelets
    for _ in range(times):
        s = _apply_perm(s, perm)
    return s


def apply_sequence(sequence, start=SOLVED):
    s = start
    for move in sequence.split():
        s = apply_move(s, move)
    return s
