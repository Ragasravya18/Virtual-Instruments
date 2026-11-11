"""All drawing: the pseudo-3D violin, bow, vibrating strings, particles and HUD."""
from __future__ import annotations

import math
import random
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config as cfg
from instrument import PlayState
from utils import clamp, midi_name

FONT = cv2.FONT_HERSHEY_SIMPLEX
WOOD = (35, 80, 150)        # BGR
WOOD_DARK = (20, 45, 95)
WOOD_LIGHT = (70, 130, 205)
EBONY = (32, 30, 34)

HELP_LINES = [
    "LEFT hand: height = pitch      RIGHT hand: swipe = bow, height = string",
    "1-4 violin / cello / guitar / theremin     F free   G rhythm game   L learn",
    "[ ] change song     SPACE start/confirm     K calibrate     W swap hands",
    "S snap to notes     R reverb     V auto-vibrato     Z pizzicato (pinch / flick)",
    "P pose anchoring    E record     B playback     M export MIDI     H help     ESC quit",
]


def _text(frame, s, org, scale=0.55, color=(255, 255, 255), thick=1) -> None:
    cv2.putText(frame, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, s, org, FONT, scale, color, thick, cv2.LINE_AA)


class Renderer:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.k = h / 480.0
        self.particles: List[list] = []
        self.toasts: List[Tuple[str, float]] = []
        self._rng = random.Random(7)
        self._last_t = time.perf_counter()

    def toast(self, text: str, seconds: float = 2.2) -> None:
        self.toasts.append((text, time.perf_counter() + seconds))

    # ---------------------------------------------------------------- layout
    def _anchor(self, hands) -> Tuple[float, float]:
        pose = hands.pose
        if pose and hands.fingering is not None:
            wrist = hands.fingering.lm[0]
            sh = min(pose["shoulders"], key=lambda p: abs(p[0] - wrist[0]))
            nose = pose["nose"]
            return ((sh[0] + nose[0]) / 2 * self.w, (sh[1] + nose[1]) / 2 * self.h)
        return (0.42 * self.w, 0.62 * self.h)

    def _geometry(self, hands, st: PlayState) -> dict:
        bx, by = self._anchor(hands)
        k = self.k
        if st.finger_xy is not None:
            sx, sy = st.finger_xy[0] * self.w, st.finger_xy[1] * self.h
        else:
            sx, sy = bx - 220 * k, by - 80 * k
        dx, dy = sx - bx, sy - by
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        return dict(B=(bx, by), u=(ux, uy), p=(-uy, ux), L=clamp(dist, 170 * k, 360 * k),
                    ang=math.degrees(math.atan2(uy, ux)))

    @staticmethod
    def _pt(g, s: float, o: float) -> Tuple[int, int]:
        return (int(g["B"][0] + g["u"][0] * s + g["p"][0] * o),
                int(g["B"][1] + g["u"][1] * s + g["p"][1] * o))

    # ---------------------------------------------------------------- violin
    def _draw_body(self, img, g) -> None:
        k, ang = self.k, g["ang"]

        def shape(dx, dy, color):
            cl, cu = self._pt(g, -38 * k, 0), self._pt(g, 32 * k, 0)
            cl = (cl[0] + dx, cl[1] + dy)
            cu = (cu[0] + dx, cu[1] + dy)
            cv2.ellipse(img, cl, (int(48 * k), int(60 * k)), ang, 0, 360, color, -1, cv2.LINE_AA)
            cv2.ellipse(img, cu, (int(38 * k), int(47 * k)), ang, 0, 360, color, -1, cv2.LINE_AA)
            poly = np.array([self._pt(g, -38 * k, -44 * k), self._pt(g, 32 * k, -35 * k),
                             self._pt(g, 32 * k, 35 * k), self._pt(g, -38 * k, 44 * k)]) + [dx, dy]
            cv2.fillConvexPoly(img, poly, color, cv2.LINE_AA)

        shape(int(7 * k), int(10 * k), (25, 25, 25))        # drop shadow
        shape(0, 0, WOOD)                                     # body
        cv2.ellipse(img, self._pt(g, -48 * k, -20 * k), (int(20 * k), int(26 * k)), ang, 0, 360,
                    WOOD_LIGHT, -1, cv2.LINE_AA)              # highlight
        for side in (-1, 1):                                   # f-holes
            cv2.line(img, self._pt(g, -20 * k, side * 21 * k), self._pt(g, 4 * k, side * 25 * k),
                     EBONY, max(2, int(3 * k)), cv2.LINE_AA)
        cv2.ellipse(img, self._pt(g, -76 * k, 10 * k), (int(14 * k), int(22 * k)), ang, 0, 360, EBONY, -1, cv2.LINE_AA)
        tail = np.array([self._pt(g, -66 * k, -9 * k), self._pt(g, -66 * k, 9 * k),
                         self._pt(g, -20 * k, 5 * k), self._pt(g, -20 * k, -5 * k)])
        cv2.fillConvexPoly(img, tail, EBONY, cv2.LINE_AA)
        # neck, fingerboard and scroll
        L = g["L"]
        neck = np.array([self._pt(g, 0, -9 * k), self._pt(g, L, -6 * k), self._pt(g, L, 6 * k), self._pt(g, 0, 9 * k)])
        cv2.fillConvexPoly(img, neck, EBONY, cv2.LINE_AA)
        cv2.circle(img, self._pt(g, L + 8 * k, 0), int(9 * k), WOOD_DARK, -1, cv2.LINE_AA)
        cv2.line(img, self._pt(g, 6 * k, -15 * k), self._pt(g, 6 * k, 15 * k), (200, 225, 245), max(2, int(3 * k)), cv2.LINE_AA)

    def _string_offset(self, i: int, n: int, s: float, L: float) -> float:
        k = self.k
        t = clamp((s - 6 * k) / max(L - 6 * k, 1))
        spacing = min(7.0, 26.0 / n) * k * (1.0 - 0.35 * t)
        return (i - (n - 1) / 2.0) * spacing

    def _draw_strings(self, frame, g, st: PlayState, now: float):
        k, L, n = self.k, g["L"], st.n_strings
        contact = self._pt(g, 0.16 * L, 0)
        for i in range(n):
            active = i == st.string_idx
            vibrating = active and (st.sounding or st.volume > 0.05)
            amp = (1.0 + 5.0 * st.volume) * k if vibrating else 0.0
            pts = []
            for j in range(19):
                s = 6 * k + (L - 6 * k) * j / 18.0
                o = self._string_offset(i, n, s, L)
                o += amp * math.sin(math.pi * j / 18.0) * math.sin(2 * math.pi * 14 * now + i)
                pts.append(self._pt(g, s, o))
            color = (0, 230, 255) if active else (225, 225, 225)
            if vibrating:
                cv2.polylines(frame, [np.array(pts)], False, (0, 140, 255), 5, cv2.LINE_AA)
            cv2.polylines(frame, [np.array(pts)], False, color, 2 if active else 1, cv2.LINE_AA)
            if active:
                contact = self._pt(g, 0.16 * L, self._string_offset(i, n, 0.16 * L, L))
        return contact

    def _draw_finger(self, frame, g, st: PlayState) -> None:
        if st.finger_xy is None:
            return
        k, L = self.k, g["L"]
        s = (L - 8 * k) - st.t_finger * (L - 8 * k - 40 * k)
        c = self._pt(g, s, self._string_offset(st.string_idx, st.n_strings, s, L))
        if st.sounding:
            cv2.circle(frame, c, int(13 * k), (0, 200, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, c, int(6 * k), (0, 255, 120), -1, cv2.LINE_AA)
        cv2.circle(frame, c, int(6 * k), (255, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------- bow
    def _draw_bow(self, frame, g, st: PlayState, contact) -> None:
        if st.bow_xy is None:
            return
        k = self.k
        hx, hy = st.bow_xy[0] * self.w, st.bow_xy[1] * self.h
        side = -1.0 if hx > g["B"][0] else 1.0
        vx, vy = math.cos(st.bow_angle) * side, math.sin(st.bow_angle) * side
        frog = (hx - vx * 30 * k, hy - vy * 30 * k)
        tip = (frog[0] + vx * 340 * k, frog[1] + vy * 340 * k)
        nx, ny = -vy, vx
        P = lambda a, o=0.0: (int(a[0] + nx * o), int(a[1] + ny * o))
        if st.volume > 0.05:                       # glow where bow meets string
            ov = frame.copy()
            cv2.circle(ov, contact, int((10 + 14 * st.volume) * k), (0, 170, 255), -1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)
        cv2.line(frame, P(frog, -3), P(tip, -3), (25, 55, 95), max(3, int(4 * k)), cv2.LINE_AA)    # stick
        cv2.line(frame, P(frog, 4), P(tip, 4), (240, 240, 240), max(2, int(2 * k)), cv2.LINE_AA)   # hair
        fr = np.array([P(frog, -7), P(frog, 8), P((frog[0] + vx * 14 * k, frog[1] + vy * 14 * k), 8),
                       P((frog[0] + vx * 14 * k, frog[1] + vy * 14 * k), -7)])
        cv2.fillConvexPoly(frame, fr, (15, 15, 15), cv2.LINE_AA)

    def _update_particles(self, frame, contact, st: PlayState, dt: float) -> None:
        if st.volume > 0.08 and st.bow_xy is not None:
            for _ in range(int(1 + 3 * st.volume)):
                self.particles.append([contact[0], contact[1], self._rng.uniform(-40, 40),
                                       self._rng.uniform(-110, -30), 0.0, self._rng.uniform(0.3, 0.7)])
        alive = []
        for p in self.particles:
            p[0] += p[2] * dt; p[1] += p[3] * dt; p[3] += 120 * dt; p[4] += dt
            if p[4] < p[5]:
                f = 1 - p[4] / p[5]
                cv2.circle(frame, (int(p[0]), int(p[1])), 1 + int(2 * f),
                           (int(120 + 135 * f), int(200 + 55 * f), 255), -1, cv2.LINE_AA)
                alive.append(p)
        self.particles = alive[-250:]

    # ------------------------------------------------------------------- HUD
    def _hud(self, frame, st: PlayState, hud: dict, scope: Optional[np.ndarray]) -> None:
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (6, 6), (270, 160), (15, 15, 20), -1)
        cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
        _text(frame, f"{st.instrument.upper()}  |  {hud['mode']}", (14, 26), 0.5, (200, 230, 255))
        note_color = (0, 255, 0) if st.sounding else (140, 140, 140)
        _text(frame, midi_name(st.midi), (14, 72), 1.4, note_color, 3)
        names = {1: "single", 4: "GDAE", 6: "EADGBE"}
        _text(frame, f"string {st.string_idx + 1}/{st.n_strings}  {st.midi:.1f}", (130, 56), 0.45, (210, 210, 210))
        _text(frame, "VIBRATO" if (st.vibrato or st.vib_depth > 0) else "", (130, 76), 0.45, (0, 215, 255))
        cv2.rectangle(frame, (14, 88), (200, 100), (70, 70, 70), 1)
        cv2.rectangle(frame, (14, 88), (14 + int(186 * st.volume), 100), (0, 220, 120), -1)
        if st.bow_dir:
            a, b = ((208, 94), (258, 94)) if st.bow_dir > 0 else ((258, 94), (208, 94))
            cv2.arrowedLine(frame, a, b, (255, 255, 255), 2, tipLength=0.4)
            _text(frame, "down-bow" if st.bow_dir > 0 else "up-bow", (205, 116), 0.4)
        flags = [("SNAP", hud["snap"]), ("REV", hud["reverb"]), ("VIB", hud["vib"]), ("PIZZ", hud["pizz"])]
        x = 14
        for name, on in flags:
            _text(frame, name, (x, 124), 0.45, (0, 255, 0) if on else (110, 110, 110))
            x += 58
        _text(frame, f"engine: {hud['engine']}", (14, 146), 0.4, (170, 170, 170))
        _text(frame, f"{hud['fps']:.0f} FPS", (w - 80, 22), 0.5, (200, 200, 200))
        if hud.get("rec"):
            cv2.circle(frame, (w - 110, 44), 6, (0, 0, 255), -1)
            _text(frame, f"REC {hud['rec_time']:.0f}s", (w - 98, 49), 0.5, (80, 80, 255))
        if hud.get("playing"):
            _text(frame, "PLAYBACK", (w - 110, 49), 0.55, (0, 215, 255))
        if not st.both_hands and not hud.get("playing"):
            _text(frame, "Show BOTH hands:  LEFT = notes   RIGHT = bow", (int(w * 0.5) - 190, h - 50), 0.55, (0, 200, 255))
        if scope is not None and len(scope) and np.any(scope):
            x0, y0, sw, sh = w - 216, h - 60, 200, 44
            ov = frame.copy()
            cv2.rectangle(ov, (x0 - 4, y0 - 4), (x0 + sw + 4, y0 + sh + 4), (15, 15, 20), -1)
            cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
            ys = (y0 + sh / 2 - np.clip(scope, -1, 1) * sh / 2).astype(int)
            xs = np.linspace(x0, x0 + sw, len(scope)).astype(int)
            cv2.polylines(frame, [np.stack([xs, ys], 1)], False, (0, 255, 160), 1, cv2.LINE_AA)
        _text(frame, "H: help", (10, h - 12), 0.45, (200, 200, 200))

    def _help(self, frame) -> None:
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (20, 90), (w - 20, 90 + 34 * len(HELP_LINES) + 20), (10, 10, 15), -1)
        cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)
        for i, line in enumerate(HELP_LINES):
            _text(frame, line, (32, 125 + 34 * i), 0.5, (255, 255, 255))

    # ------------------------------------------------------------------ main
    def draw(self, frame, st: PlayState, hands, hud: dict, scope=None) -> None:
        now = time.perf_counter()
        dt = min(now - self._last_t, 0.1)
        self._last_t = now
        g = self._geometry(hands, st)
        ov = frame.copy()
        self._draw_body(ov, g)
        cv2.addWeighted(ov, 0.92, frame, 0.08, 0, frame)
        contact = self._draw_strings(frame, g, st, now)
        self._draw_finger(frame, g, st)
        self._draw_bow(frame, g, st, contact)
        self._update_particles(frame, contact, st, dt)
        self._hud(frame, st, hud, scope)
        self.toasts = [t for t in self.toasts if t[1] > now]
        for i, (text, _) in enumerate(self.toasts[-3:]):
            _text(frame, text, (int(self.w / 2) - 7 * len(text), 60 + 28 * i), 0.7, (0, 255, 255), 2)

    def draw_overlays(self, frame, hud: dict) -> None:
        """Help sheet and calibration banner - drawn last so nothing covers them."""
        if hud.get("help"):
            self._help(frame)
        if hud.get("banner"):
            for i, line in enumerate(hud["banner"]):
                _text(frame, line, (int(self.w / 2) - 7 * len(line), int(self.h / 2) + 30 * i), 0.7, (0, 255, 255), 2)
