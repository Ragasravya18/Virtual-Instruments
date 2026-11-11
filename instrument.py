"""Gesture -> music logic (no camera / audio code in here, so it is easy to test).

Fingering hand : fingertip height  -> pitch (continuous or snapped to semitones)
Bow hand       : height            -> which string
                 horizontal speed  -> loudness (stopped bow = silence)
                 direction         -> up-bow / down-bow
                 pinch or flick    -> pluck (pizzicato / guitar)
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import config as cfg
from utils import OneEuroFilter, clamp, snap_with_hysteresis


@dataclass
class PlayState:
    instrument: str = "violin"
    sounding: bool = False
    midi: float = 60.0
    volume: float = 0.0          # 0..1 loudness (HUD + game logic)
    audio_volume: float = 0.0    # what the audio engine should play (bowed notes)
    vib_depth: float = 0.0       # semitones of synthetic vibrato
    string_idx: int = 0
    n_strings: int = 4
    t_finger: float = 0.0        # 0 = lowest position, 1 = highest
    bow_dir: int = 0             # +1 / -1 / 0
    bow_speed: float = 0.0
    bow_angle: float = 0.0
    vibrato: bool = False        # natural vibrato detected in the finger
    pluck: bool = False          # a pluck happened this frame
    pluck_vol: float = 0.0
    both_hands: bool = False
    finger_xy: Optional[Tuple[float, float]] = None   # normalised
    bow_xy: Optional[Tuple[float, float]] = None


class VirtualInstrument:
    def __init__(self) -> None:
        self.snap = cfg.SNAP_DEFAULT
        self.auto_vibrato = True
        self.pizz = False
        self.finger_range = cfg.DEFAULT_FINGER_RANGE
        self.bow_range = cfg.DEFAULT_BOW_RANGE
        self.wide: Optional[Tuple[float, float]] = None   # (base midi, span) override
        self._ffx, self._ffy = OneEuroFilter(), OneEuroFilter()
        self._fbx, self._fby = OneEuroFilter(), OneEuroFilter()
        self._hist: Deque[Tuple[float, float]] = deque(maxlen=40)
        self._last_t: Optional[float] = None
        self._t = 0.0
        self._u = 0.5
        self._vel = 0.0
        self._prev_bx: Optional[float] = None
        self._snapped: Optional[float] = None
        self._last_active = -1.0
        self._last_vol = 0.0
        self._was_pinch = self._was_flick = False
        self._last_pluck = -1.0
        self._pluck_env = 0.0
        self._note_id: Optional[int] = None
        self._note_since = 0.0
        self.set_instrument("violin")

    # ------------------------------------------------------------ settings
    def set_instrument(self, name: str) -> None:
        self.name = name
        self.profile = cfg.INSTRUMENTS[name]
        self._snapped = None

    def set_wide(self, base: float, span: float) -> None:
        self.wide = (base, span)

    def clear_wide(self) -> None:
        self.wide = None

    def set_calibration(self, finger=None, bow=None) -> None:
        if finger and abs(finger[1] - finger[0]) > 0.05:
            self.finger_range = (finger[0], finger[1])
        if bow and abs(bow[1] - bow[0]) > 0.05:
            self.bow_range = (bow[0], bow[1])

    # ------------------------------------------------------------ helpers
    def _detect_vibrato(self, now: float) -> bool:
        pts = [(t, v) for t, v in self._hist if now - t <= 0.6]
        if len(pts) < 10:
            return False
        vals = [v for _, v in pts]
        mean = sum(vals) / len(vals)
        dev = [v - mean for v in vals]
        amp = (max(dev) - min(dev)) / 2
        crossings = sum(1 for a, b in zip(dev, dev[1:]) if a * b < 0)
        dur = pts[-1][0] - pts[0][0]
        if dur <= 0:
            return False
        rate = crossings / 2 / dur
        return 0.06 <= amp <= 1.2 and 3.5 <= rate <= 9.5

    # ------------------------------------------------------------ main update
    def update(self, hands, now: float) -> PlayState:
        prof = self.profile
        dt = max((now - self._last_t) if self._last_t else 1 / 30, 1e-3)
        self._last_t = now
        n_str = len(prof["strings"])
        st = PlayState(instrument=self.name, n_strings=n_str)
        fh, bh = hands.fingering, hands.bow
        st.both_hands = fh is not None and bh is not None

        # ---- fingering hand -> position along the neck
        if fh is not None:
            fx = self._ffx(fh.lm[8][0], now)
            fy = self._ffy(fh.lm[8][1], now)
            st.finger_xy = (fx, fy)
            hi, lo = self.finger_range
            self._t = clamp((lo - fy) / (lo - hi)) if lo != hi else 0.0
        else:
            self._ffx.reset(); self._ffy.reset(); self._hist.clear()
        st.t_finger = self._t

        # ---- bow hand -> speed, direction, string
        if bh is not None:
            bx = self._fbx(bh.lm[9][0], now)
            by = self._fby(bh.lm[9][1], now)
            st.bow_xy = (bx, by)
            if self._prev_bx is not None:
                self._vel += cfg.BOW_SMOOTH * ((bx - self._prev_bx) / dt - self._vel)
            self._prev_bx = bx
            top, bot = self.bow_range
            self._u = clamp((by - top) / (bot - top)) if bot != top else 0.5
            wx, wy = bh.lm[0]
            mx, my = bh.lm[9]
            ang = math.atan2(my - wy, mx - wx) + math.pi / 2
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            st.bow_angle = clamp(ang, -0.5, 0.5)
        else:
            self._fbx.reset(); self._fby.reset()
            self._prev_bx, self._vel = None, 0.0
        st.string_idx = min(n_str - 1, int((1.0 - self._u) * n_str))
        speed = abs(self._vel)
        st.bow_speed = speed
        if bh is not None and speed > cfg.BOW_DEADZONE:
            st.bow_dir = 1 if self._vel > 0 else -1

        # ---- pitch
        if self.wide is not None:
            base, span = self.wide
        elif prof["wide"]:
            base, span = prof["strings"][0], prof["span"]
        else:
            base, span = prof["strings"][st.string_idx], prof["span"]
        raw = base + self._t * span
        vib = False
        if fh is not None:
            self._hist.append((now, raw))
            vib = self._detect_vibrato(now)
        if self.snap and not vib:
            midi = snap_with_hysteresis(raw, self._snapped, cfg.SNAP_HYSTERESIS)
            self._snapped = midi
        else:
            midi = raw
            if vib:
                self._snapped = float(round(raw))
        st.midi, st.vibrato = midi, vib

        # ---- loudness / plucking
        active = st.both_hands
        plucked = (not prof["bowed"]) or self.pizz
        if not plucked:
            if speed > cfg.BOW_DEADZONE:
                target = clamp((speed - cfg.BOW_DEADZONE) / (cfg.BOW_MAX_SPEED - cfg.BOW_DEADZONE)) ** 0.6
                self._last_active, self._last_vol = now, target
            elif now - self._last_active < 0.07:      # bridge bow changes
                target = self._last_vol * 0.6
            else:
                target = 0.0
            st.volume = target if active else 0.0
            st.audio_volume = st.volume
            st.sounding = active and st.volume > 0.02
        else:
            pinch = False
            if bh is not None:
                palm = math.dist(bh.lm[0], bh.lm[9]) + 1e-6
                pinch = math.dist(bh.lm[4], bh.lm[8]) / palm < cfg.PINCH_RATIO
            flick = speed > cfg.FLICK_SPEED
            trigger = (pinch and not self._was_pinch) or (flick and not self._was_flick)
            self._was_pinch, self._was_flick = pinch, flick
            self._pluck_env *= 0.5 ** (dt / 0.35)
            if trigger and active and now - self._last_pluck > 0.12:
                st.pluck = True
                st.pluck_vol = clamp(0.5 + 0.5 * speed / cfg.BOW_MAX_SPEED)
                self._last_pluck, self._pluck_env = now, st.pluck_vol
            st.volume = self._pluck_env if active else 0.0
            st.sounding = active and st.volume > 0.15

        # ---- synthetic vibrato once a note has been held
        if st.sounding and not plucked:
            nid = int(round(midi))
            if nid != self._note_id:
                self._note_id, self._note_since = nid, now
            held = now - self._note_since
            if self.auto_vibrato and not vib and prof["vibrato"] and held > 0.5:
                st.vib_depth = cfg.AUTO_VIBRATO_DEPTH * clamp((held - 0.5) / 0.6)
        else:
            self._note_id, self._note_since = None, now
        return st
