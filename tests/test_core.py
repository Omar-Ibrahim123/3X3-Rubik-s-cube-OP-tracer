"""Tests for the pure-logic core: cube model, Speffz, OP tracer, TPS counter."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from cube import speffz
from cube.state import CubeState
from solver import trace, trace_corners, trace_edges
from tps import TPSCounter
from tests.cube_moves import SOLVED, apply_sequence


# --------------------------- Speffz integrity ---------------------------

def test_speffz_tables_are_consistent():
    # Every corner letter has a 3-cycle whose set matches its physical piece.
    for letter, cyc in speffz.CORNER_COMPANIONS.items():
        assert len(cyc) == 3
        assert cyc[0] == letter
        assert set(cyc) in [set(p) for p in speffz.CORNER_PIECES]
    # Edge companions are symmetric.
    for a, b in speffz.EDGE_COMPANION.items():
        assert speffz.EDGE_COMPANION[b] == a
    # 24 corner letters, 24 edge letters, all distinct.
    assert len(speffz.CORNER_COMPANIONS) == 24
    assert len(speffz.EDGE_COMPANION) == 24


def test_home_maps_cover_all_letters():
    assert set(speffz.CORNER_HOME) == set(speffz.LETTERS)
    assert set(speffz.EDGE_HOME) == set(speffz.LETTERS)


# --------------------------- Solved cube ---------------------------

def test_solved_cube_has_empty_trace():
    state = CubeState.from_facelets(SOLVED)
    result = trace(state)
    assert result.corners == []
    assert result.edges == []
    assert result.parity is False


def test_solved_state_is_identity_permutation():
    state = CubeState.from_facelets(SOLVED)
    for pos in speffz.CORNER_COMPANIONS:
        assert state.corner_at[pos] == pos
    for pos in speffz.EDGE_COMPANION:
        assert state.edge_at[pos] == pos


# --------------------------- Scrambled cubes ---------------------------

@pytest.mark.parametrize("scramble", [
    "R",
    "U",
    "R U R' U'",
    "R U R' U' R' F R2 U' R' U' R U R' F'",   # T-perm
    "F R U' R' U' R U R' F' R U R' U' R' F R F'",  # a longer alg
    "R2 U2 R2 U2 R2 U2",
    "D L F2 U' R B2 L' D2 F R' U B",
])
def test_trace_is_wellformed(scramble):
    facelets = apply_sequence(scramble)
    state = CubeState.from_facelets(facelets)
    result = trace(state)
    # Targets are valid Speffz letters.
    for t in result.corners:
        assert t in speffz.LETTERS
    for t in result.edges:
        assert t in speffz.LETTERS
    # Parity flag matches edge-count parity.
    assert result.parity == (len(result.edges) % 2 == 1)


def test_single_U_scramble_permutes_pieces():
    # A single U turn is a 4-cycle on both corners and edges in the U layer.
    facelets = apply_sequence("U")
    state = CubeState.from_facelets(facelets)
    result = trace(state)
    # U alone leaves orientation intact and permutes 4 corners / 4 edges, so
    # the trace must be non-empty and even-length per orbit (3-target cycle
    # for a 4-cycle after the buffer, etc.). Just assert non-trivial.
    assert result.corners or result.edges


def test_trace_pairs_formatting():
    facelets = apply_sequence("R U R' U'")
    result = trace(CubeState.from_facelets(facelets))
    pairs = result.corner_pairs()
    # Pairs reconstruct the flat list.
    flat = "".join(pairs)
    assert flat == "".join(result.corners)


# --------------------------- TPS counter ---------------------------

def test_tps_counter_basic():
    t = [0.0]
    c = TPSCounter(clock=lambda: t[0])
    c.start()
    t[0] = 2.0
    c.record_turn()
    c.record_turn()
    c.record_turn()
    # 3 turns over 2 seconds = 1.5 TPS.
    assert c.turns == 3
    assert c.elapsed == 2.0
    assert c.tps == 1.5


def test_tps_counter_stop_freezes_time():
    t = [10.0]
    c = TPSCounter(clock=lambda: t[0])
    c.start()
    t[0] = 14.0
    c.record_turn(8)
    assert c.stop() == 2.0
    t[0] = 100.0  # advancing the clock after stop must not change tps
    assert c.tps == 2.0
    assert c.elapsed == 4.0


def test_tps_requires_start():
    c = TPSCounter()
    with pytest.raises(RuntimeError):
        c.record_turn()


# --------------------------- Trace correctness ---------------------------

def _independent_trace(at, buffer, home, positions):
    """Independent cycle-decomposition of the permutation (see _verify_trace)."""
    targets, visited = [], set()
    bpiece = home[buffer]

    def follow(start, closing):
        cur = start
        guard = 0
        while True:
            guard += 1
            assert guard < 100, "non-terminating follow"
            occ = at[cur]
            if home[occ] == closing:
                return
            targets.append(occ)
            visited.add(home[occ])
            cur = occ

    visited.add(bpiece)
    follow(buffer, bpiece)
    for pos in sorted(positions):
        if home[pos] in visited or at[pos] == pos:
            continue
        targets.append(pos)
        visited.add(home[pos])
        follow(pos, home[pos])
    return targets


@pytest.mark.parametrize("scramble", [
    "U",
    "R U R' U'",
    "R U R' U R U2 R'",
    "R2 U2 R2 U2 R2 U2",
    "D L F2 U' R B2 L' D2 F R' U B",
    "F2 L2 D' R2 F2 D L2 U2 B2 U R U L' B F' D2 R' F'",
])
def test_trace_matches_independent_decomposition(scramble):
    from tests.cube_moves import apply_sequence as seq
    st = CubeState.from_facelets(seq(scramble))

    c_got = trace_corners(st)
    c_exp = _independent_trace(
        st.corner_at, speffz.CORNER_BUFFER, speffz.CORNER_HOME,
        list(speffz.CORNER_COMPANIONS),
    )
    assert c_got == c_exp, f"corner trace mismatch for {scramble}"

    e_got = trace_edges(st)
    e_exp = _independent_trace(
        st.edge_at, speffz.EDGE_BUFFER, speffz.EDGE_HOME,
        list(speffz.EDGE_COMPANION),
    )
    assert e_got == e_exp, f"edge trace mismatch for {scramble}"


@pytest.mark.parametrize("scramble", [
    "U", "R U R' U'", "D L F2 U' R B2 L' D2 F R' U B",
])
def test_every_out_of_place_piece_is_covered(scramble):
    """Coverage invariant: pieces not named in the trace must be solved."""
    from tests.cube_moves import apply_sequence as seq
    st = CubeState.from_facelets(seq(scramble))
    for at, buf, home, positions, targets in [
        (st.corner_at, speffz.CORNER_BUFFER, speffz.CORNER_HOME,
         list(speffz.CORNER_COMPANIONS), trace_corners(st)),
        (st.edge_at, speffz.EDGE_BUFFER, speffz.EDGE_HOME,
         list(speffz.EDGE_COMPANION), trace_edges(st)),
    ]:
        covered = {home[t] for t in targets} | {home[buf]}
        for pos in positions:
            if home[pos] not in covered:
                assert at[pos] == pos, f"uncovered unsolved piece at {pos}"


def test_malformed_state_does_not_hang():
    """A corrupted permutation must raise, not loop forever."""
    st = CubeState.from_facelets(SOLVED)
    bad = dict(st.corner_at)
    # Create an impossible mapping that never returns to the buffer.
    bad["A"] = "B"
    bad["B"] = "C"
    bad["C"] = "B"  # cycle that excludes the buffer piece
    from solver.op import _trace_orbit
    with pytest.raises(RuntimeError):
        _trace_orbit(bad, "A", speffz.CORNER_HOME, speffz.CORNER_COMPANIONS)
