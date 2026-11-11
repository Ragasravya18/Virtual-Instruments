"""Play modes: Free Play, Rhythm Game and Learn Mode.

Songs live in songs.json:  {"Name": {"tempo": 64, "notes": [[midi, beats], ...]}}
(use null as the midi value for a rest)
"""
from __future__ import annotations

import colorsys
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config as cfg
from instrument import PlayState
from utils import clamp, midi_name

FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------- songs
@dataclass
class Note:
    midi: int
    start: float
    end: float
    hold: float = 0.0
    judged: bool = False
    result: str = ""


def load_songs(path=cfg.SONGS_FILE) -> Dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"Scale": {"tempo": 60, "notes": [[m, 1] for m in (60, 62, 64, 65, 67, 69, 71, 72)]}}


def build_notes(song: dict) -> List[Note]:
    beat = 60.0 / song.get("tempo", 60)
    t, notes = 0.0, []
    for midi, beats in song["notes"]:
        if midi is not None:
            notes.append(Note(int(midi), t, t + beats * beat))
        t += beats * beat
    return notes


def grade_for(accuracy: float) -> str:
    for limit, g in ((0.95, "S"), (0.85, "A"), (0.70, "B"), (0.50, "C")):
        if accuracy >= limit:
            return g
    return "D"


def _panel(frame, x0, y0, x1, y1, alpha=0.6) -> None:
    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x1, y1), (15, 15, 20), -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (90, 90, 100), 1)


def _text(frame, s, org, scale=0.6, color=(255, 255, 255), thick=1) -> None:
    cv2.putText(frame, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, s, org, FONT, scale, color, thick, cv2.LINE_AA)


# --------------------------------------------------------------------- modes
class Mode:
    name = "Free Play"
    uses_song = False

    def __init__(self, songs: Dict[str, dict]):
        self.songs = songs
        self.song_names = list(songs)
        self.song_idx = 0

    @property
    def song(self) -> dict:
        return self.songs[self.song_names[self.song_idx]]

    @property
    def song_name(self) -> str:
        return self.song_names[self.song_idx]

    def set_song(self, idx: int) -> None:
        self.song_idx = idx % len(self.song_names)
        self.reset()

    def wide(self) -> Optional[Tuple[float, float]]:
        """Pitch mapping override (base midi, span) so the whole song is reachable."""
        if not self.uses_song:
            return None
        ps = [n.midi for n in build_notes(self.song)]
        return min(ps) - 1, max(ps) - min(ps) + 2

    def reset(self) -> None: ...
    def start(self, now: float) -> None: ...
    def update(self, st: PlayState, now: float) -> None: ...
    def draw(self, frame, st: PlayState) -> None: ...


class FreePlay(Mode):
    name = "Free Play"


class RhythmGame(Mode):
    name = "Rhythm Game"
    uses_song = True
    POINTS = {"PERFECT": 100, "GOOD": 60, "OK": 25, "MISS": 0}

    def __init__(self, songs):
        super().__init__(songs)
        self.high = self._load_high()
        self.reset()

    # ---- persistence
    def _load_high(self) -> Dict[str, int]:
        try:
            with open(cfg.HIGHSCORE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_high(self) -> None:
        try:
            with open(cfg.HIGHSCORE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.high, f, indent=1)
        except OSError:
            pass

    # ---- state
    def reset(self) -> None:
        self.notes = build_notes(self.song)
        self.pitches = sorted({n.midi for n in self.notes})
        self.state = "ready"          # ready | playing | done
        self.score = self.combo = self.max_combo = 0
        self.base_points = 0
        self.t0: Optional[float] = None
        self.cur_t = -cfg.COUNT_IN_SECONDS
        self._last = 0.0
        self.flash = ("", 0.0)
        self.grade = ""
        self.new_high = False

    def start(self, now: float) -> None:
        self.reset()
        self.state, self.t0, self._last = "playing", now + cfg.COUNT_IN_SECONDS, now

    def accuracy(self) -> float:
        return self.base_points / (100.0 * len(self.notes)) if self.notes else 0.0

    # ---- logic
    @staticmethod
    def hits(st: PlayState, note: Note) -> bool:
        return st.sounding and abs(st.midi - note.midi) < 0.75

    def judge(self, note: Note) -> None:
        need = min(note.end - note.start, 0.6)
        frac = clamp(note.hold / need) if need > 0 else 0.0
        res = "PERFECT" if frac >= 0.75 else "GOOD" if frac >= 0.45 else "OK" if frac >= 0.15 else "MISS"
        note.judged, note.result = True, res
        self.base_points += self.POINTS[res]
        if res == "MISS":
            self.combo = 0
        else:
            if res != "OK":
                self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
        self.score += int(self.POINTS[res] * (1 + 0.05 * min(self.combo, 20)))

    def update(self, st: PlayState, now: float) -> None:
        if self.state != "playing":
            return
        dt = max(now - self._last, 0.0)
        self._last = now
        self.cur_t = now - self.t0
        t = self.cur_t
        for n in self.notes:
            if n.judged:
                continue
            if n.start - 0.2 <= t <= n.end + 0.15 and self.hits(st, n):
                n.hold += dt
            if t > n.end + 0.15:
                self.judge(n)
                self.flash = (n.result, now)
        if all(n.judged for n in self.notes):
            self.state = "done"
            self.grade = grade_for(self.accuracy())
            if self.score > self.high.get(self.song_name, 0):
                self.high[self.song_name] = self.score
                self.new_high = True
                self._save_high()

    # ---- drawing
    def _lane_pos(self, midi: float) -> float:
        ps = self.pitches
        if midi <= ps[0]:
            return 0.0
        if midi >= ps[-1]:
            return float(len(ps) - 1)
        for i in range(len(ps) - 1):
            if ps[i] <= midi <= ps[i + 1]:
                return i + (midi - ps[i]) / (ps[i + 1] - ps[i])
        return 0.0

    def draw(self, frame, st: PlayState) -> None:
        h, w = frame.shape[:2]
        x0, x1, top, hit_y = int(w * 0.60), w - 10, 70, int(h * 0.78)
        _panel(frame, x0, top - 30, x1, h - 40)
        _text(frame, self.song_name[:30], (x0 + 8, top - 10), 0.42, (200, 230, 255))
        n_l = len(self.pitches)
        lane_w = (x1 - x0) / n_l
        pps = cfg.FALL_PIXELS_PER_SECOND
        for i, p in enumerate(self.pitches):
            cx = int(x0 + lane_w * (i + 0.5))
            cv2.line(frame, (cx, top), (cx, hit_y), (60, 60, 70), 1)
            _text(frame, midi_name(p), (cx - 14, hit_y + 22), 0.4, (180, 180, 190))
        cv2.line(frame, (x0 + 4, hit_y), (x1 - 4, hit_y), (255, 255, 255), 2)

        t = self.cur_t
        for n in self.notes:
            i = self.pitches.index(n.midi)
            cx = int(x0 + lane_w * (i + 0.5))
            yb = int(hit_y - (n.start - t) * pps)
            yt = int(hit_y - (n.end - t) * pps)
            if yb < top or yt > hit_y + 18:
                continue
            r, g, b = (int(c * 255) for c in colorsys.hsv_to_rgb(i / max(n_l, 1) * 0.8, 0.7, 1.0))
            color = (b, g, r)
            if n.judged:
                color = (90, 200, 90) if n.result != "MISS" else (70, 70, 160)
            hw = int(lane_w * 0.32)
            y_lo = min(yb - 4, hit_y + 18)          # small gap between repeated notes
            if y_lo <= max(yt, top):
                continue
            cv2.rectangle(frame, (cx - hw, max(yt, top)), (cx + hw, y_lo), color, -1)
            cv2.rectangle(frame, (cx - hw, max(yt, top)), (cx + hw, y_lo), (255, 255, 255), 1)

        # where the player currently is
        if st.sounding or st.both_hands:
            px = int(x0 + lane_w * (self._lane_pos(st.midi) + 0.5))
            col = (0, 255, 255) if st.sounding else (150, 150, 150)
            cv2.drawMarker(frame, (px, hit_y + 6), col, cv2.MARKER_TRIANGLE_UP, 18, 3)

        _text(frame, f"Score {self.score}", (x0 + 8, h - 56), 0.55, (255, 255, 120))
        _text(frame, f"Combo {self.combo}", (x1 - 100, h - 56), 0.55, (120, 255, 255))
        best = self.high.get(self.song_name, 0)
        _text(frame, f"Best {best}", (x0 + 8, top + 8), 0.45, (190, 190, 190))
        if self.state == "ready":
            _text(frame, "SPACE to start", (x0 + 30, hit_y - 80), 0.7, (0, 255, 0), 2)
        elif self.state == "playing" and t < 0:
            _text(frame, str(int(-t) + 1), (x0 + int((x1 - x0) / 2) - 15, hit_y - 80), 1.6, (0, 255, 255), 3)
        elif self.state == "playing" and self.flash[0] and (self._last - self.flash[1]) < 0.6:
            _text(frame, self.flash[0], (x0 + 40, hit_y - 60), 0.8, (0, 255, 0) if self.flash[0] != "MISS" else (80, 80, 255), 2)
        elif self.state == "done":
            _text(frame, f"GRADE {self.grade}", (x0 + 40, top + 80), 1.2, (0, 255, 255), 3)
            _text(frame, f"{int(self.accuracy() * 100)}% accuracy  max combo {self.max_combo}",
                  (x0 + 10, top + 115), 0.5)
            if self.new_high:
                _text(frame, "NEW HIGH SCORE!", (x0 + 30, top + 145), 0.65, (0, 215, 255), 2)
            _text(frame, "SPACE to retry", (x0 + 40, top + 175), 0.55, (0, 255, 0))


class LearnMode(Mode):
    name = "Learn Mode"
    uses_song = True
    HOLD_SECONDS = 0.4

    def __init__(self, songs):
        super().__init__(songs)
        self.reset()

    def reset(self) -> None:
        self.seq = [n.midi for n in build_notes(self.song)]
        self.idx = 0
        self.ok_since: Optional[float] = None
        self.done = False
        self.message = "Find the highlighted note"

    def start(self, now: float) -> None:
        self.reset()

    def update(self, st: PlayState, now: float) -> None:
        if self.done:
            return
        target = self.seq[self.idx]
        diff = st.midi - target
        if not st.sounding:
            self.message, self.ok_since = "Bow to hear the note", None
        elif abs(diff) < 0.6:
            self.ok_since = self.ok_since or now
            self.message = "Perfect - hold it!"
            if now - self.ok_since >= self.HOLD_SECONDS:
                self.idx += 1
                self.ok_since = None
                if self.idx >= len(self.seq):
                    self.done, self.message = True, "Song complete!"
        else:
            self.ok_since = None
            self.message = "Too high - move down" if diff > 0 else "Too low - move up"

    def draw(self, frame, st: PlayState) -> None:
        h, w = frame.shape[:2]
        x0, x1, top = int(w * 0.62), w - 10, 40
        _panel(frame, x0, top, x1, int(h * 0.62))
        _text(frame, self.song_name[:26], (x0 + 8, top + 22), 0.5, (200, 230, 255))
        if self.done:
            _text(frame, "COMPLETE!", (x0 + 30, top + 100), 1.0, (0, 255, 255), 2)
            _text(frame, "SPACE to restart", (x0 + 30, top + 140), 0.55, (0, 255, 0))
            return
        target = self.seq[self.idx]
        _text(frame, midi_name(target), (x0 + 60, top + 100), 2.0, (0, 255, 0), 3)
        _text(frame, f"note {self.idx + 1}/{len(self.seq)}", (x0 + 10, top + 130), 0.5)
        # pitch meter: target in the middle, +-3 semitones
        mx0, mx1, my = x0 + 20, x1 - 20, top + 165
        cv2.line(frame, (mx0, my), (mx1, my), (120, 120, 130), 2)
        cx = (mx0 + mx1) // 2
        cv2.line(frame, (cx, my - 12), (cx, my + 12), (0, 255, 0), 3)
        if st.sounding:
            off = clamp((st.midi - target) / 3.0, -1, 1)
            cv2.circle(frame, (int(cx + off * (mx1 - mx0) / 2), my), 8, (0, 255, 255), -1)
        _text(frame, self.message, (x0 + 10, top + 205), 0.55, (255, 255, 120))


def create_modes(songs=None) -> Dict[str, Mode]:
    songs = songs or load_songs()
    return {"free": FreePlay(songs), "game": RhythmGame(songs), "learn": LearnMode(songs)}
