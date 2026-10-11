"""Speffz lettering scheme definitions for a 3x3 Rubik's cube.

The Speffz scheme assigns a letter A-X to every corner sticker-position and,
separately, a letter A-X to every edge sticker-position. Letters are assigned
one face at a time in the face order U, L, F, R, B, D; within a face the four
stickers of the relevant piece type are lettered clockwise when looking
directly at that face.

This module is the single source of truth for how physical sticker positions
map to Speffz letters. The vision layer (added later) only needs to produce a
standard 54-character facelet string; everything downstream speaks Speffz.

Reference: https://www.speedsolving.com/wiki/index.php/Speffz
(scheme described in the wiki; implementation here is original.)
"""

from __future__ import annotations

# Face order used throughout the project.
FACE_ORDER = ("U", "L", "F", "R", "B", "D")

# Speffz letters A..X (24 letters) for corners and, independently, for edges.
LETTERS = tuple(chr(ord("A") + i) for i in range(24))

# ---------------------------------------------------------------------------
# Facelet-string indexing.
#
# We use the Kociemba facelet convention: a 54-char string ordered
#   U1..U9  R1..R9  F1..F9  D1..D9  L1..L9  B1..B9
# where each face's 9 stickers are indexed 1..9 in reading order:
#     1 2 3
#     4 5 6
#     7 8 9
# with sticker 5 being the center.
#
# NOTE: the Kociemba face order (U R F D L B) differs from the Speffz face
# order (U L F R B D). We keep the facelet string in Kociemba order because it
# is the universal interchange format, and translate to Speffz here.
# ---------------------------------------------------------------------------

# Starting index of each face within the 54-char Kociemba facelet string.
_KOCIEMBA_FACE_START = {"U": 0, "R": 9, "F": 18, "D": 27, "L": 36, "B": 45}


def facelet_index(face: str, sticker: int) -> int:
    """Return the 0-based index into a 54-char facelet string.

    `face` is one of U R F D L B; `sticker` is 1..9 in reading order.
    """
    if face not in _KOCIEMBA_FACE_START:
        raise ValueError(f"unknown face {face!r}")
    if not 1 <= sticker <= 9:
        raise ValueError(f"sticker must be 1..9, got {sticker}")
    return _KOCIEMBA_FACE_START[face] + (sticker - 1)


# ---------------------------------------------------------------------------
# Speffz sticker-position -> facelet (face, sticker) mapping.
#
# Each entry maps a Speffz letter to the (face, sticker) of the physical
# facelet it labels. Corner and edge maps are independent.
#
# Corner stickers occupy sticker slots {1,3,7,9} of a face; edge stickers
# occupy slots {2,4,6,8}. The clockwise ordering below follows the Speffz
# convention for each face.
# ---------------------------------------------------------------------------

# Corners: letter -> (face, sticker). Clockwise from the standard start sticker
# of each face, faces in order U, L, F, R, B, D.
CORNER_FACELET = {
    "A": ("U", 1), "B": ("U", 3), "C": ("U", 9), "D": ("U", 7),
    "E": ("L", 1), "F": ("L", 3), "G": ("L", 9), "H": ("L", 7),
    "I": ("F", 1), "J": ("F", 3), "K": ("F", 9), "L": ("F", 7),
    "M": ("R", 1), "N": ("R", 3), "O": ("R", 9), "P": ("R", 7),
    "Q": ("B", 1), "R": ("B", 3), "S": ("B", 7), "T": ("B", 9),
    "U": ("D", 1), "V": ("D", 3), "W": ("D", 9), "X": ("D", 7),
}

# Edges: letter -> (face, sticker).
EDGE_FACELET = {
    "A": ("U", 2), "B": ("U", 6), "C": ("U", 8), "D": ("U", 4),
    "E": ("L", 2), "F": ("L", 6), "G": ("L", 8), "H": ("L", 4),
    "I": ("F", 2), "J": ("F", 6), "K": ("F", 8), "L": ("F", 4),
    "M": ("R", 2), "N": ("R", 6), "O": ("R", 8), "P": ("R", 4),
    "Q": ("B", 2), "R": ("B", 6), "S": ("B", 8), "T": ("B", 4),
    "U": ("D", 2), "V": ("D", 6), "W": ("D", 8), "X": ("D", 4),
}

# ---------------------------------------------------------------------------
# Standard Old Pochmann buffers.
#
# Corner buffer: Speffz 'A' (the U-face sticker of the UBL corner).
# Edge buffer:   Speffz 'B' (the U-face sticker of the UB... see note).
#
# These are the conventional OP buffers. They are defined here as named
# constants so the solver never hard-codes a position and so they can be
# changed in one place if a different buffer is ever desired.
# ---------------------------------------------------------------------------
CORNER_BUFFER = "A"
EDGE_BUFFER = "B"

# ---------------------------------------------------------------------------
# Piece groupings and sticker-companion tables.
#
# A physical corner has 3 stickers; a physical edge has 2. For OP tracing we
# must know, for any Speffz letter, the other letters on the SAME physical
# piece, in cyclic order, so that we can:
#   - resolve a piece's orientation (which sticker faces where), and
#   - "shoot to the sticker" (emit the specific sticker that lands the piece
#     both permuted and oriented correctly).
#
# CORNER_COMPANIONS[x] = (x, next_cw, next_next_cw): the three stickers of the
# corner, in clockwise order starting at x. Rotating the tuple by one position
# corresponds to one physical clockwise twist of that corner.
#
# EDGE_COMPANION[x] = the single other sticker on the same edge. Flipping an
# edge swaps its two stickers.
#
# These tables are derived directly from CORNER_FACELET / EDGE_FACELET above
# and match the community-standard Speffz grouping.
# ---------------------------------------------------------------------------

CORNER_COMPANIONS = {
    "A": ("A", "E", "R"), "B": ("B", "Q", "N"),
    "C": ("C", "M", "J"), "D": ("D", "I", "F"),
    "E": ("E", "R", "A"), "F": ("F", "D", "I"),
    "G": ("G", "U", "L"), "H": ("H", "X", "T"),
    "I": ("I", "F", "D"), "J": ("J", "C", "M"),
    "K": ("K", "P", "V"), "L": ("L", "G", "U"),
    "M": ("M", "J", "C"), "N": ("N", "B", "Q"),
    "O": ("O", "S", "W"), "P": ("P", "V", "K"),
    "Q": ("Q", "N", "B"), "R": ("R", "A", "E"),
    "S": ("S", "W", "O"), "T": ("T", "H", "X"),
    "U": ("U", "L", "G"), "V": ("V", "K", "P"),
    "W": ("W", "O", "S"), "X": ("X", "T", "H"),
}

EDGE_COMPANION = {
    "A": "Q", "Q": "A", "B": "M", "M": "B", "C": "I", "I": "C",
    "D": "E", "E": "D", "F": "L", "L": "F", "J": "P", "P": "J",
    "N": "T", "T": "N", "R": "H", "H": "R", "U": "K", "K": "U",
    "V": "O", "O": "V", "W": "S", "S": "W", "X": "G", "G": "X",
}

# Physical pieces as tuples of their Speffz letters (first letter is the
# canonical "home" sticker used to identify the piece).
CORNER_PIECES = (
    ("A", "E", "R"), ("B", "Q", "N"), ("C", "M", "J"), ("D", "I", "F"),
    ("U", "L", "G"), ("V", "K", "P"), ("W", "O", "S"), ("X", "T", "H"),
)
EDGE_PIECES = (
    ("A", "Q"), ("B", "M"), ("C", "I"), ("D", "E"), ("F", "L"), ("J", "P"),
    ("N", "T"), ("R", "H"), ("U", "K"), ("V", "O"), ("W", "S"), ("X", "G"),
)

# letter -> the canonical home letter of the piece it belongs to.
CORNER_HOME = {letter: piece[0] for piece in CORNER_PIECES for letter in piece}
EDGE_HOME = {letter: piece[0] for piece in EDGE_PIECES for letter in piece}
