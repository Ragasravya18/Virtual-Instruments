"""Quick 4-step calibration so the pitch / string zones fit your reach."""
from __future__ import annotations

import json
from typing import Optional

import config as cfg

STEPS = [
    ("finger_hi", "LEFT hand: raise it to your HIGHEST note, press SPACE"),
    ("finger_lo", "LEFT hand: lower it to your LOWEST note, press SPACE"),
    ("bow_hi", "RIGHT hand: hold it at the HIGH (E) string height, press SPACE"),
    ("bow_lo", "RIGHT hand: hold it at the LOW (G) string height, press SPACE"),
]


class Calibration:
    def __init__(self) -> None:
        self.active = False
        self.step = 0
        self.values: dict = {}

    def start(self) -> None:
        self.active, self.step, self.values = True, 0, {}

    def banner(self):
        if not self.active:
            return None
        return [f"CALIBRATION {self.step + 1}/{len(STEPS)}", STEPS[self.step][1], "(ESC-free: press K to cancel)"]

    def capture(self, hands) -> Optional[str]:
        """Record the current hand position. Returns an error text, 'done', or None."""
        key = STEPS[self.step][0]
        hand = hands.fingering if key.startswith("finger") else hands.bow
        if hand is None:
            return "Hand not visible"
        self.values[key] = hand.lm[8][1] if key.startswith("finger") else hand.lm[9][1]
        self.step += 1
        if self.step >= len(STEPS):
            self.active = False
            return "done"
        return None

    def result(self):
        v = self.values
        return (v["finger_hi"], v["finger_lo"]), (v["bow_hi"], v["bow_lo"])

    @staticmethod
    def save(finger, bow) -> None:
        try:
            with open(cfg.CALIBRATION_FILE, "w", encoding="utf-8") as f:
                json.dump({"finger": finger, "bow": bow}, f)
        except OSError:
            pass

    @staticmethod
    def load():
        try:
            with open(cfg.CALIBRATION_FILE, encoding="utf-8") as f:
                d = json.load(f)
            return tuple(d["finger"]), tuple(d["bow"])
        except (OSError, ValueError, KeyError):
            return None, None
