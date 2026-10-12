"""Ground-truth OP executor + end-to-end solve verification.

This test file builds an INDEPENDENT, faithful model of how an Old Pochmann
(OP) blindfolded solve is executed, and uses it to check the invariant that was
missing from the project: *the generated trace must actually solve the cube.*

Why a separate executor?
------------------------
`solver/op.py` only *generates* a target list (the memo). Nothing in the repo
ever *executed* that memo against the cube state to confirm it solves. The
executor here closes that loop: it replays the memo as a sequence of OP swaps
on the cube STATE and asserts the result is the identity (solved).

State representation (matches cube/state.py `_resolve`)
-------------------------------------------------------
`at[pos]` = the specific Speffz sticker-letter physically sitting in position
`pos`, orientation resolved. `at[pos] == pos` means that slot is solved. A
physical piece occupies the companion positions of `pos`.

How a shot is modelled -- and why it is trustworthy
---------------------------------------------------
A single OP shot swaps the buffer piece with the piece in a target slot. The
shot's effect on `at` is a FIXED, state-independent permutation `s_T` (the
"swap table"): applying the shot to the solved cube yields exactly `s_T`
(solved `at` is the identity), and a shot on any state is `at_new = at o s_T`,
i.e. `at_new[p] = at[s_T[p]]`.

We derive every `s_T` directly from the facelet move engine (tests.cube_moves)
by realising the shot physically as `setup(T) . base . setup(T)^-1` and parsing
the result back through `CubeState`. The corner base is a Y-perm (swaps corner
buffer A with C); the edge base is a T-perm whose EDGE effect is a clean 2-swap
of edge buffer B with edge D. This makes the executor faithful to how
`cube/state.py` resolves orientation, with NO assumption about Speffz companion
ordering -- so it is an independent oracle for the solver's trace.

The executor is self-validated below: the solver's trace must already solve
pure 3-cycle scrambles under it, and `s_T[buffer] == T` must hold for every
target (the defining property of "shoot to T").
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from cube import speffz
from cube.state import CubeState
from solver import trace_corners, trace_edges
from tests.cube_moves import apply_sequence


# ---------------------------------------------------------------------------
# Derive the ground-truth swap tables once, from the facelet move engine.
#
# For target T, we realise a single OP shot physically as
#     setup(T) . base . setup(T)^-1
# on the solved cube and parse the result through CubeState. Because solved
# `at` is the identity, the resulting diff IS the fixed swap permutation s_T.
# `setup(T)` is the shortest face-turn sequence (BFS) that makes the shot land
# on T, i.e. produces a pure 2-piece swap whose buffer-slot sticker is T.
# ---------------------------------------------------------------------------

_MOVES = ["U", "D", "R", "L", "F", "B"]
_SUFFIX = ["", "'", "2"]
# Corner base: Y-perm (swaps corner buffer A with C).
_CORNER_BASE = "F R U' R' U' R U R' F' R U R' U' R' F R F'".split()
# Edge base: T-perm; its EDGE effect is a pure 2-swap of edge buffer B with D.
_EDGE_BASE = "R U R' U' R' F R2 U' R' U' R U R' F'".split()


def _inverse(seq):
    return [m[0] if m.endswith("'") else (m if m.endswith("2") else m + "'")
            for m in reversed(seq)]


def _at_diff(seq, kind):
    st = CubeState.from_facelets(apply_sequence(" ".join(seq)))
    at = st.corner_at if kind == "corner" else st.edge_at
    return {p: v for p, v in at.items() if p != v}


def _setup_sequences(max_len):
    """BFS over face-turn sequences, shortest first, skipping same-face repeats."""
    frontier = [[]]
    yield []
    for _ in range(max_len):
        nxt = []
        for s in frontier:
            for f in _MOVES:
                if s and s[-1][0] == f:
                    continue
                for suf in _SUFFIX:
                    cand = s + [f + suf]
                    nxt.append(cand)
                    yield cand
        frontier = nxt


def _derive_swaps(base, buffer, kind, need, max_setup=4):
    n = 6 if kind == "corner" else 4
    table = {}
    for setup in _setup_sequences(max_setup):
        diff = _at_diff(setup + base + _inverse(setup), kind)
        if len(diff) == n and buffer in diff:
            table.setdefault(diff[buffer], diff)
        if len(table) >= need:
            break
    return table


_CORNER_TARGETS = [c for c in speffz.LETTERS if c not in ("A", "E", "R")]
_EDGE_TARGETS = [c for c in speffz.LETTERS if c not in ("B", "M")]
CORNER_SWAP = _derive_swaps(_CORNER_BASE, "A", "corner", len(_CORNER_TARGETS))
EDGE_SWAP = _derive_swaps(_EDGE_BASE, "B", "edge", len(_EDGE_TARGETS))


# ---------------------------------------------------------------------------
# The OP executor
# ---------------------------------------------------------------------------

def _shoot(at, target, table):
    """Apply one OP shot (buffer <-> target) to `at`, returning the new map.

    `table[target]` is the fixed swap permutation `s_T` as a {pos: sticker}
    diff; positions not listed are fixed. The shot is `at_new = at o s_T`.
    """
    s_t = table[target]
    return {p: at[s_t.get(p, p)] for p in at}


def execute(at_initial, targets, table):
    at = dict(at_initial)
    for t in targets:
        at = _shoot(at, t, table)
    return at


def _is_solved(at):
    return all(at[p] == p for p in at)


def _solves(st):
    c = trace_corners(st)
    e = trace_edges(st)
    end_c = execute(st.corner_at, c, CORNER_SWAP)
    end_e = execute(st.edge_at, e, EDGE_SWAP)
    return _is_solved(end_c) and _is_solved(end_e), c, e, end_c, end_e


# ---------------------------------------------------------------------------
# Executor self-validation (trustworthiness)
# ---------------------------------------------------------------------------

def test_swap_tables_complete():
    assert sorted(CORNER_SWAP) == sorted(_CORNER_TARGETS)
    assert sorted(EDGE_SWAP) == sorted(_EDGE_TARGETS)


def test_shot_sends_buffer_to_target():
    """Defining property of "shoot to T": the sticker landing in the buffer
    slot is exactly T. (Equivalently s_T[buffer] == T.)"""
    for t, s in CORNER_SWAP.items():
        assert s.get("A", "A") == t
    for t, s in EDGE_SWAP.items():
        assert s.get("B", "B") == t


def test_shot_is_an_involution():
    """Shooting the same target twice is a no-op (OP shots are involutions)."""
    solved_c = {p: p for p in speffz.CORNER_COMPANIONS}
    for t in CORNER_SWAP:
        assert _shoot(_shoot(solved_c, t, CORNER_SWAP), t, CORNER_SWAP) == solved_c
    solved_e = {p: p for p in speffz.EDGE_COMPANION}
    for t in EDGE_SWAP:
        assert _shoot(_shoot(solved_e, t, EDGE_SWAP), t, EDGE_SWAP) == solved_e


# Pure-permutation 3-cycle scrambles (parity- and orientation-free). If the
# executor model is right, the solver's trace already solves these -- this
# validates the executor itself before we trust it to judge the fix.
_PURE_3CYCLE = [
    "R' F R' B2 R F' R' B2 R2",            # corner 3-cycle (A-perm)
    "R2 U' R' U' R U R U R U' R",          # edge 3-cycle (U-perm style)
]


@pytest.mark.parametrize("scramble", _PURE_3CYCLE)
def test_executor_solves_pure_3cycle_traces(scramble):
    st = CubeState.from_facelets(apply_sequence(scramble))
    ok, c, e, end_c, end_e = _solves(st)
    assert ok, (
        f"executor/trace disagree on pure 3-cycle {scramble}:\n"
        f"corners={c} leftover={ {p: v for p, v in end_c.items() if p != v} }\n"
        f"edges={e} leftover={ {p: v for p, v in end_e.items() if p != v} }"
    )


# ---------------------------------------------------------------------------
# End-to-end: the trace must SOLVE the cube (the missing invariant)
# ---------------------------------------------------------------------------

_MOVES = ["U", "D", "R", "L", "F", "B"]
_SUFFIX = ["", "'", "2"]


def _random_scramble(rng, n):
    out, last = [], None
    for _ in range(n):
        f = rng.choice(_MOVES)
        while f == last:
            f = rng.choice(_MOVES)
        last = f
        out.append(f + rng.choice(_SUFFIX))
    return " ".join(out)


@pytest.mark.parametrize("seed", range(40))
def test_trace_solves_random_scrambles(seed):
    rng = random.Random(seed)
    scramble = _random_scramble(rng, 25)
    st = CubeState.from_facelets(apply_sequence(scramble))
    ok, c, e, end_c, end_e = _solves(st)
    assert ok, (
        f"trace did not solve scramble (seed={seed}): {scramble}\n"
        f"corners={c} leftover={ {p: v for p, v in end_c.items() if p != v} }\n"
        f"edges={e} leftover={ {p: v for p, v in end_e.items() if p != v} }"
    )


def test_trace_solves_large_random_sample():
    """Large sample: the generated trace must solve every scramble."""
    rng = random.Random(12345)
    failures = 0
    total = 1000
    for _ in range(total):
        scramble = _random_scramble(rng, rng.randint(1, 30))
        st = CubeState.from_facelets(apply_sequence(scramble))
        ok, *_ = _solves(st)
        if not ok:
            failures += 1
    assert failures == 0, f"{failures}/{total} random scrambles not solved"


REAL_SCAN = "RLBRUBUULBUDDRFRLDFRULFFRDFUBDBDURDFBFLLLDURBLFDUBBLRF"


def test_trace_solves_real_scanned_cube():
    st = CubeState.from_facelets(REAL_SCAN)
    ok, c, e, end_c, end_e = _solves(st)
    assert ok, (
        f"trace did not solve the real scanned cube\n"
        f"corners={c} leftover={ {p: v for p, v in end_c.items() if p != v} }\n"
        f"edges={e} leftover={ {p: v for p, v in end_e.items() if p != v} }"
    )


# ---------------------------------------------------------------------------
# Orientation cost (cubing theory): flipped edge = 2 targets, twisted = 2.
# ---------------------------------------------------------------------------

def _place_corner(at, slot, piece, twist):
    ring_s = speffz.CORNER_COMPANIONS[slot]
    ring_p = speffz.CORNER_COMPANIONS[piece]
    for i, pos in enumerate(ring_s):
        at[pos] = ring_p[(i + twist) % 3]


def test_flipped_edge_in_place_needs_two_targets():
    """A single edge flipped in place costs exactly 2 targets, and the memo
    (together with its balancing flip) must re-solve the cube."""
    # Two edges flipped in place (flips must come in pairs on a real cube).
    at = {p: p for p in speffz.EDGE_COMPANION}
    at["F"], at["L"] = "L", "F"      # flip edge F
    at["W"], at["S"] = "S", "W"      # flip edge W

    # Trace one orbit directly via the solver's internal function.
    from solver.op import _trace_orbit
    edge_companions = {
        k: (k, speffz.EDGE_COMPANION[k]) for k in speffz.EDGE_COMPANION
    }
    targets = _trace_orbit(at, "B", speffz.EDGE_HOME, edge_companions)
    assert len(targets) == 4, f"two flipped edges should cost 4 targets: {targets}"
    end = execute(at, targets, EDGE_SWAP)
    assert _is_solved(end), f"flipped-edge memo must solve: {targets} -> {end}"


def test_twisted_corner_in_place_needs_two_targets():
    at = {p: p for p in speffz.CORNER_COMPANIONS}
    _place_corner(at, "D", "D", 1)   # twist corner D one way
    _place_corner(at, "C", "C", 2)   # twist corner C the other (balanced)
    from solver.op import _trace_orbit
    targets = _trace_orbit(at, "A", speffz.CORNER_HOME, speffz.CORNER_COMPANIONS)
    assert len(targets) == 4, f"two twisted corners should cost 4 targets: {targets}"
    end = execute(at, targets, CORNER_SWAP)
    assert _is_solved(end), f"twisted-corner memo must solve: {targets} -> {end}"
