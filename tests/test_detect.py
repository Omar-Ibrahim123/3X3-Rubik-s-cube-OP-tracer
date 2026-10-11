"""Tests for the pure cube-face detection geometry (no camera / no OpenCV)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import pytest

from vision.detect import (
    Square,
    detect_face_grid,
    filter_similar_squares,
)


def make_grid(cx=300, cy=300, spacing=60, size=50, angle=0.0):
    """9 squares arranged in a 3x3 grid, optionally rotated by `angle` rad."""
    squares = []
    for row in range(3):
        for col in range(3):
            dx = (col - 1) * spacing
            dy = (row - 1) * spacing
            # rotate (dx, dy) by angle
            rx = dx * math.cos(angle) - dy * math.sin(angle)
            ry = dx * math.sin(angle) + dy * math.cos(angle)
            squares.append(Square(cx=cx + rx, cy=cy + ry, size=size))
    return squares


def test_detects_clean_grid_and_orders_reading_order():
    squares = make_grid()
    centers = detect_face_grid(squares)
    assert centers is not None
    assert len(centers) == 9
    # Reading order: row 0 (smallest y) first, left-to-right within row.
    rows = [centers[0:3], centers[3:6], centers[6:9]]
    for row in rows:
        xs = [p[0] for p in row]
        assert xs == sorted(xs), "row not left-to-right"
    ys_top = [p[1] for p in rows[0]]
    ys_bot = [p[1] for p in rows[2]]
    assert max(ys_top) < min(ys_bot), "rows not top-to-bottom"


def test_detects_slightly_tilted_grid():
    squares = make_grid(angle=math.radians(8))
    centers = detect_face_grid(squares)
    assert centers is not None
    assert len(centers) == 9


def test_rejects_too_few_squares():
    squares = make_grid()[:7]
    assert detect_face_grid(squares) is None


def test_filters_out_wrong_sized_squares():
    squares = make_grid()
    # Add a huge background rectangle and a tiny speck.
    squares.append(Square(cx=50, cy=50, size=400))
    squares.append(Square(cx=600, cy=600, size=5))
    kept = filter_similar_squares(squares)
    assert len(kept) == 9  # the 9 real stickers only


def test_detects_face_despite_extra_noise_squares():
    squares = make_grid()
    # Scatter a few similar-sized noise squares far from the face centroid.
    squares += [
        Square(cx=20, cy=20, size=50),
        Square(cx=580, cy=40, size=50),
    ]
    centers = detect_face_grid(squares)
    assert centers is not None
    assert len(centers) == 9


def test_rejects_non_grid_arrangement():
    # 9 similar squares but in a line, not a 3x3 grid.
    squares = [Square(cx=100 + i * 40, cy=300, size=50) for i in range(9)]
    assert detect_face_grid(squares) is None


def test_empty_input():
    assert detect_face_grid([]) is None
    assert filter_similar_squares([]) == []
