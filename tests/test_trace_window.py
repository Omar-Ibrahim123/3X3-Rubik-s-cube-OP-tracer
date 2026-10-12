"""Tests for the on-screen OP result panel (vision.capture.render_trace_image).

The result window is drawn with OpenCV, but OpenCV is not available in CI, so
these tests inject a tiny cv2/numpy stub that records the draw calls. That lets
us assert the LAYOUT behaviour that actually matters:

  * the real corner/edge memo (from the real solver) is rendered, split into
    the two panels, with correct pair counts;
  * a parity banner appears only when the trace has parity;
  * every piece of text and every rectangle stays inside the canvas; and
  * a long (worst-case) memo is auto-shrunk to fit its card instead of
    overflowing -- the regression that a fixed font size would cause.

We drive it with a genuine trace from solver/op.py (not hand-written strings),
so the test breaks if the panel ever stops reflecting the real memo.
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from cube.state import CubeState
from solver.op import trace
from tests.cube_moves import apply_sequence


# Per-character advance (px) our fake getTextSize models for scale 1.0. The real
# HERSHEY_SIMPLEX is close to this; the exact value doesn't matter as long as
# render_trace_image and the test agree, since we only check the fit invariant.
_PX_PER_CHAR = 22


def _install_stubs(monkeypatch):
    """Install minimal numpy + cv2 stubs and return the recorded-calls dict."""
    np = types.ModuleType("numpy")

    class _Img:
        def __init__(self, shape):
            self.shape = shape

    np.full = lambda shape, fill, dtype=None: _Img(shape)
    np.uint8 = "uint8"

    cv2 = types.ModuleType("cv2")
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.LINE_AA = 16
    calls = {"rectangle": [], "putText": []}
    cv2.rectangle = lambda img, p1, p2, color, thick: calls["rectangle"].append(
        (p1, p2, color, thick)
    )

    def put_text(img, text, org, font, scale, color, thick, line):
        calls["putText"].append((text, org, scale, thick))

    cv2.putText = put_text
    cv2.getTextSize = lambda text, font, scale, thick: (
        (int(len(text) * _PX_PER_CHAR * scale), int(26 * scale)),
        6,
    )
    # Display + input stubs for the in-window confirm flow. `waitKey` returns a
    # steady key value the test sets via calls["key"], so the confirm loop
    # terminates deterministically instead of blocking.
    calls["key"] = 255
    calls["destroyed"] = []
    cv2.imshow = lambda *a, **k: None
    cv2.destroyWindow = lambda name: calls["destroyed"].append(name)
    cv2.waitKey = lambda delay=0: calls["key"]

    monkeypatch.setitem(sys.modules, "numpy", np)
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    import vision.capture as capture

    monkeypatch.setattr(capture, "_lazy_imports", lambda: (cv2, np))
    return capture, calls


class _FakeTrace:
    def __init__(self, corner_pairs, edge_pairs, parity):
        self._c, self._e, self.parity = corner_pairs, edge_pairs, parity

    def corner_pairs(self):
        return self._c

    def edge_pairs(self):
        return self._e


def test_renders_real_trace_split_with_parity(monkeypatch):
    capture, calls = _install_stubs(monkeypatch)

    # A real scramble that produces a parity case (verified: 5 corner pairs,
    # 6 edge pairs, parity True).
    facelets = apply_sequence("R U R2 U2 F2 D B2 L F2 R2 U D2 L2 B R U2")
    t = trace(CubeState.from_facelets(facelets))

    img = capture.render_trace_image(t)
    H, W, _ = img.shape
    texts = [c[0] for c in calls["putText"]]

    # The real memo appears, split into the two panels.
    assert "  ".join(t.corner_pairs()) in texts
    assert "  ".join(t.edge_pairs()) in texts
    assert "CORNERS" in texts and "EDGES" in texts
    assert f"{len(t.corner_pairs())} pairs" in texts
    assert f"{len(t.edge_pairs())} pairs" in texts

    # Parity banner present for this parity trace.
    assert t.parity
    assert any(txt.startswith("PARITY") for txt in texts)

    # Everything stays on-canvas.
    for txt, (x, y), _scale, _thick in calls["putText"]:
        assert 0 <= x < W and 0 <= y < H, f"text off-canvas: {txt!r} @ ({x},{y})"
    for p1, p2, _color, _thick in calls["rectangle"]:
        for (x, y) in (p1, p2):
            assert 0 <= x <= W and 0 <= y <= H


def test_no_parity_banner_when_solved_like_trace(monkeypatch):
    capture, calls = _install_stubs(monkeypatch)
    t = _FakeTrace(["AB", "CD"], ["EF", "GH"], parity=False)
    capture.render_trace_image(t)
    texts = [c[0] for c in calls["putText"]]
    assert not any(txt.startswith("PARITY") for txt in texts)


def test_long_memo_autofits_card(monkeypatch):
    capture, calls = _install_stubs(monkeypatch)
    # Worst-case-ish: many pairs in each orbit.
    t = _FakeTrace(
        ["AB", "CD", "EF", "GH", "IJ", "KL", "MN"],
        ["AB", "CD", "EF", "GH", "IJ", "KL", "MN", "OP", "QR", "ST", "UV"],
        parity=True,
    )
    img = capture.render_trace_image(t)
    W = img.shape[1]
    # The card runs from x=44 to (W-24); render leaves a 20px right pad.
    trace_max_w = (W - 24) - 44 - 20

    big = [c for c in calls["putText"] if c[3] == 3]  # the two memo strings
    assert len(big) == 2
    for text, _org, scale, _thick in big:
        width = len(text) * _PX_PER_CHAR * scale
        assert width <= trace_max_w, (
            f"memo overflows card: {text!r} width~{int(width)} > {trace_max_w}"
        )


# --------------------------------------------------------------------------
# In-window confirmation: the terminal must stay silent and the accept /
# re-scan decision must come from a keypress in the net window, not input().
# --------------------------------------------------------------------------

_FACELETS = "DUFBUUDRFRFRRRBBDDRBDDFULRUUFRDDFBLFBLBDLBLLFULLUBRLFU"


class _Report:
    def __init__(self, solvable=True, parity=False):
        self.solvable = solvable
        self.parity = parity
        self.low_confidence = []


@pytest.mark.parametrize(
    "key, expected",
    [
        (ord("y"), True),   # accept
        (ord("Y"), True),
        (13, True),         # Enter
        (ord("n"), False),  # re-scan
        (ord("N"), False),
        (ord("q"), False),  # quit -> re-scan
        (27, False),        # Esc -> re-scan
    ],
)
def test_confirm_decision_comes_from_window_key(monkeypatch, capsys, key, expected):
    capture, calls = _install_stubs(monkeypatch)
    calls["key"] = key
    result = capture.confirm_facelets(_FACELETS, None, _Report())
    assert result is expected
    # The terminal must be completely silent -- nothing printed either way.
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_confirm_accept_closes_net_window(monkeypatch):
    capture, calls = _install_stubs(monkeypatch)
    calls["key"] = ord("y")
    capture.confirm_facelets(_FACELETS, None, _Report())
    assert "detected cube" in calls["destroyed"]


def test_confirm_falls_back_to_accept_without_display(monkeypatch, capsys):
    capture, calls = _install_stubs(monkeypatch)

    # Simulate no display: imshow raises (as a headless OpenCV would).
    def _no_display(*a, **k):
        raise RuntimeError("cannot open display")

    import cv2  # the stub installed above
    cv2.imshow = _no_display

    result = capture.confirm_facelets(_FACELETS, None, _Report())
    assert result is True  # headless can't confirm visually -> accept
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
