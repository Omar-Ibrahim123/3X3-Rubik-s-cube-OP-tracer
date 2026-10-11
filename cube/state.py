"""Cube state model.

Parses a standard 54-character facelet string (Kociemba order
U R F D L B, each face read 1..9) and exposes, for the solver, the mapping
"which physical sticker currently occupies each Speffz position".

The vision layer added later only needs to produce the 54-char facelet string;
this module turns raw colors into the Speffz-level permutation the tracer uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from . import speffz


@dataclass(frozen=True)
class CubeState:
    """An immutable cube state built from a 54-char facelet string.

    `facelets` is the raw string. `corner_at` / `edge_at` map each Speffz
    position letter to the Speffz home-sticker letter of the piece currently
    occupying that position, with orientation resolved: the value is the
    specific companion sticker that is physically sitting in that slot.
    """

    facelets: str
    corner_at: Dict[str, str]
    edge_at: Dict[str, str]

    @classmethod
    def from_facelets(cls, facelets: str) -> "CubeState":
        facelets = facelets.strip().upper()
        if len(facelets) != 54:
            raise ValueError(
                f"facelet string must be 54 chars, got {len(facelets)}"
            )

        # Map each face's center color -> face name, so we can translate the
        # raw colors on each sticker into face letters (U L F R B D).
        centers = {
            facelets[speffz.facelet_index(f, 5)]: f
            for f in ("U", "R", "F", "D", "L", "B")
        }
        if len(centers) != 6:
            raise ValueError("facelets must have 6 distinct center colors")

        def face_of(letter_map, pos_letter):
            """Resolve which piece-sticker sits at a Speffz position."""
            # The piece occupying `pos_letter` is identified by reading the
            # colors on all of that position's stickers and matching them to a
            # home piece.
            return letter_map[pos_letter]

        corner_at = cls._resolve(facelets, centers, kind="corner")
        edge_at = cls._resolve(facelets, centers, kind="edge")
        return cls(facelets=facelets, corner_at=corner_at, edge_at=edge_at)

    @staticmethod
    def _resolve(facelets, centers, kind: str) -> Dict[str, str]:
        """Build position-letter -> occupying-sticker-letter map.

        For each Speffz position, read the color on that facelet and the
        colors on its companion facelets, identify which physical piece those
        colors form, and record the specific sticker now sitting in the slot.
        """
        if kind == "corner":
            facelet_map = speffz.CORNER_FACELET
            companions = speffz.CORNER_COMPANIONS
            pieces = speffz.CORNER_PIECES
        else:
            facelet_map = speffz.EDGE_FACELET
            companions = {
                k: (k, speffz.EDGE_COMPANION[k]) for k in speffz.EDGE_COMPANION
            }
            pieces = speffz.EDGE_PIECES

        # Map a frozenset of face-letters -> the piece (ordered tuple).
        piece_by_faces = {}
        for piece in pieces:
            faces = frozenset(facelet_map[s][0] for s in piece)
            piece_by_faces[faces] = piece

        # The solved color of a face equals that face's center color. Invert.
        color_of_face = {f: c for c, f in centers.items()}

        result: Dict[str, str] = {}
        for pos_letter, (face, sticker) in facelet_map.items():
            # Read colors on this position's stickers (the position's own
            # facelet plus its companions' facelets).
            cycle = companions[pos_letter]
            colors = [
                facelets[speffz.facelet_index(*facelet_map[c])] for c in cycle
            ]
            # Faces these colors belong to (the home faces of the piece here).
            try:
                home_faces = frozenset(_face_for_color(colors, color_of_face))
            except KeyError as exc:  # pragma: no cover - defensive
                raise ValueError(f"unknown sticker color {exc}") from None
            piece = piece_by_faces.get(home_faces)
            if piece is None:
                raise ValueError(
                    f"facelets describe an invalid piece at position "
                    f"{pos_letter}: colors {colors}"
                )
            # The specific sticker occupying this slot is the piece sticker
            # whose home face matches the color currently on THIS facelet.
            this_color = colors[0]
            this_face = color_of_face[this_color]
            for s in piece:
                if facelet_map[s][0] == this_face:
                    result[pos_letter] = s
                    break
        return result


def _face_for_color(colors, color_of_face):
    return [color_of_face[c] for c in colors]
