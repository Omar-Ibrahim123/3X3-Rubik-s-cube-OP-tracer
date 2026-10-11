"""Turns-per-second counter.

TPS = total turns / elapsed time. The counter is deliberately decoupled from
*what* produces turns: callers invoke `record_turn()` whenever a turn happens.
Today that caller is solution playback; later the live camera-based turn
detector can drive the exact same interface. The clock uses time.monotonic()
so it never jumps backward if the system clock changes.
"""

from __future__ import annotations

import time
from typing import Callable, Optional


class TPSCounter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._start: Optional[float] = None
        self._end: Optional[float] = None
        self._turns = 0

    def start(self) -> None:
        self._start = self._clock()
        self._end = None
        self._turns = 0

    def record_turn(self, count: int = 1) -> None:
        if self._start is None:
            raise RuntimeError("call start() before record_turn()")
        if self._end is not None:
            raise RuntimeError("counter already stopped")
        if count < 0:
            raise ValueError("count must be non-negative")
        self._turns += count

    @property
    def turns(self) -> int:
        return self._turns

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else self._clock()
        return end - self._start

    @property
    def tps(self) -> float:
        e = self.elapsed
        return self._turns / e if e > 0 else 0.0

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("counter was never started")
        if self._end is None:
            self._end = self._clock()
        return self.tps

    def status(self) -> str:
        return f"{self.tps:5.2f} TPS | {self.elapsed:6.1f}s | {self._turns} turns"
