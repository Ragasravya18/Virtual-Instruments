"""Record a performance, play it back and export it as a .mid file."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import config as cfg
from instrument import PlayState
from utils import write_midi

MAX_FRAMES = 6000   # ~3 minutes at 30 fps


@dataclass
class Frame:
    t: float
    midi: float
    volume: float
    instrument: str
    pluck: bool
    pluck_vol: float
    string_idx: int
    t_finger: float


class Recorder:
    def __init__(self) -> None:
        self.frames: List[Frame] = []
        self.recording = False
        self.playing = False
        self._t0 = 0.0
        self._pi = 0

    # ---- recording
    def start(self, now: float) -> None:
        self.frames, self.recording, self.playing, self._t0 = [], True, False, now

    def stop(self) -> None:
        self.recording = False

    def add(self, now: float, st: PlayState) -> None:
        if not self.recording:
            return
        if len(self.frames) >= MAX_FRAMES:
            self.recording = False
            return
        self.frames.append(Frame(now - self._t0, st.midi, st.audio_volume, st.instrument,
                                 st.pluck, st.pluck_vol, st.string_idx, st.t_finger))

    def duration(self, now: float) -> float:
        return now - self._t0 if self.recording else (self.frames[-1].t if self.frames else 0.0)

    # ---- playback
    def start_playback(self, now: float) -> bool:
        if not self.frames:
            return False
        self.recording, self.playing, self._t0, self._pi = False, True, now, 0
        return True

    def playback(self, now: float) -> Optional[PlayState]:
        """Return the state at `now`, or None when the recording has finished."""
        el = now - self._t0
        if el > self.frames[-1].t:
            self.playing = False
            return None
        plucked, pvol = False, 0.0
        while self._pi + 1 < len(self.frames) and self.frames[self._pi + 1].t <= el:
            self._pi += 1
            if self.frames[self._pi].pluck:
                plucked, pvol = True, self.frames[self._pi].pluck_vol
        fr = self.frames[self._pi]
        return PlayState(instrument=fr.instrument, midi=fr.midi, volume=fr.volume,
                         audio_volume=fr.volume, sounding=fr.volume > 0.02,
                         string_idx=fr.string_idx, t_finger=fr.t_finger, both_hands=True,
                         pluck=plucked, pluck_vol=pvol)

    # ---- export
    def to_notes(self):
        notes, cur = [], None   # cur = [start, pitch, vol_sum, count]

        def close(end):
            nonlocal cur
            if cur and end - cur[0] >= 0.05:
                notes.append((cur[0], end, cur[1], int(30 + 97 * cur[2] / cur[3])))
            cur = None

        for fr in self.frames:
            pitch = int(round(fr.midi))
            if fr.pluck:
                notes.append((fr.t, fr.t + 0.6, pitch, int(40 + 87 * fr.pluck_vol)))
                continue
            active = fr.volume > 0.08
            if cur and (not active or pitch != cur[1]):
                close(fr.t)
            if active and cur is None:
                cur = [fr.t, pitch, 0.0, 0]
            if cur:
                cur[2] += fr.volume
                cur[3] += 1
        if cur and self.frames:
            close(self.frames[-1].t)
        return notes

    def export(self, path: Optional[Path] = None) -> Optional[Path]:
        notes = self.to_notes()
        if not notes:
            return None
        if path is None:
            import time
            cfg.RECORDINGS_DIR.mkdir(exist_ok=True)
            path = cfg.RECORDINGS_DIR / time.strftime("performance_%Y%m%d_%H%M%S.mid")
        program = cfg.INSTRUMENTS[self.frames[0].instrument]["gm"]
        write_midi(path, notes, program=program)
        return Path(path)
