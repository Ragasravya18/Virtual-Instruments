"""Small, dependency-free helpers: music maths, filters and a tiny MIDI writer."""
from __future__ import annotations

import math
import struct
from typing import Iterable, Optional, Sequence, Tuple

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def midi_name(m: float) -> str:
    n = int(round(m))
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def snap_with_hysteresis(raw: float, prev: Optional[float], hysteresis: float) -> float:
    """Round to the nearest semitone but stay on the previous note until the
    pitch has clearly moved away from it (avoids flickering between notes)."""
    if prev is not None and abs(raw - prev) <= 0.5 + hysteresis:
        return prev
    return float(round(raw))


class OneEuroFilter:
    """One Euro filter: low lag when moving fast, strong smoothing when still."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 4.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._t = None
        self._x = None
        self._raw = None
        self._dx = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._t is None:
            self._t, self._x, self._raw = t, x, x
            return x
        dt = max(t - self._t, 1e-3)
        dx = (x - self._raw) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t, self._raw = t, x
        return self._x


def vlq(value: int) -> bytes:
    """MIDI variable-length quantity."""
    value = max(0, int(value))
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def write_midi(path, notes: Iterable[Tuple[float, float, int, int]],
               program: int = 40, bpm: int = 120, tpb: int = 480) -> None:
    """Write a single-track Standard MIDI File.

    notes: iterable of (start_seconds, end_seconds, midi_pitch, velocity)
    """
    def tick(s: float) -> int:
        return int(round(s * tpb * bpm / 60.0))

    events = []
    for start, end, pitch, vel in notes:
        pitch = int(max(0, min(127, pitch)))
        vel = int(max(1, min(127, vel)))
        t0 = tick(start)
        t1 = max(tick(end), t0 + 1)
        events.append((t0, 1, bytes([0x90, pitch, vel])))
        events.append((t1, 0, bytes([0x80, pitch, 0])))
    events.sort(key=lambda e: (e[0], e[1]))

    track = bytearray()
    track += vlq(0) + bytes([0xFF, 0x51, 0x03]) + int(60_000_000 / bpm).to_bytes(3, "big")
    track += vlq(0) + bytes([0xC0, int(program) & 0x7F])
    last = 0
    for t, _, data in events:
        track += vlq(t - last) + data
        last = t
    track += vlq(0) + bytes([0xFF, 0x2F, 0x00])
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpb))
        f.write(b"MTrk" + struct.pack(">I", len(track)) + bytes(track))
