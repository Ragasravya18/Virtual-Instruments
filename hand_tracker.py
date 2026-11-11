"""Camera thread + MediaPipe hand (and optional pose) tracking."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2

import config as cfg

Point = Tuple[float, float]


@dataclass
class Hand:
    lm: List[Point]            # 21 normalised (x, y) landmarks
    label: str = "Left"        # as seen by the user (frame is mirrored)
    raw: object = None         # MediaPipe landmark object, for drawing


@dataclass
class Hands:
    fingering: Optional[Hand] = None
    bow: Optional[Hand] = None
    pose: Optional[dict] = None   # {"nose": (x,y), "shoulders": [(x,y), (x,y)]}


class CameraStream:
    """Grabs frames on a background thread so the main loop never waits on I/O."""

    def __init__(self, index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ok = self.cap.isOpened()
        self._frame = None
        self._cond = threading.Condition()
        self._run = self.ok
        if self.ok:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while self._run:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame = frame
                self._cond.notify_all()

    def read(self, timeout: float = 0.2):
        with self._cond:
            self._cond.wait(timeout)
            return None if self._frame is None else self._frame.copy()

    def release(self) -> None:
        self._run = False
        self.cap.release()


class HandTracker:
    def __init__(self, swap: bool = cfg.SWAP_HANDS, use_pose: bool = cfg.USE_POSE):
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This MediaPipe version has no legacy 'solutions' API.\n"
                "Install the pinned version:  pip install mediapipe==0.10.14")
        self.mp = mp
        self.swap = swap
        self.use_pose = use_pose
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2, model_complexity=0,
            min_detection_confidence=cfg.HAND_CONFIDENCE,
            min_tracking_confidence=cfg.HAND_CONFIDENCE)
        self._pose = None
        self._pose_pts: Optional[dict] = None
        self._n = 0
        self._draw = mp.solutions.drawing_utils
        self._conn = mp.solutions.hands.HAND_CONNECTIONS

    # ------------------------------------------------------------------ pose
    def _run_pose(self, rgb) -> None:
        if self._pose is None:
            try:   # the pose model is downloaded on first use, so it needs internet once
                self._pose = self.mp.solutions.pose.Pose(model_complexity=0, smooth_landmarks=True)
            except Exception as exc:
                print(f"[pose] could not start pose tracking ({exc}); anchoring disabled.")
                self.use_pose = False
                return
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            self._pose_pts = None
            return
        lm = res.pose_landmarks.landmark
        if min(lm[0].visibility, lm[11].visibility, lm[12].visibility) < 0.5:
            self._pose_pts = None
            return
        self._pose_pts = {"nose": (lm[0].x, lm[0].y),
                          "shoulders": [(lm[11].x, lm[11].y), (lm[12].x, lm[12].y)]}

    # --------------------------------------------------------------- process
    def process(self, bgr) -> Hands:
        """`bgr` must already be mirrored (selfie view)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        found: List[Hand] = []
        if result.multi_hand_landmarks:
            for lms, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                found.append(Hand([(p.x, p.y) for p in lms.landmark],
                                  handed.classification[0].label, lms))
        out = Hands()
        if len(found) == 2:
            if found[0].label == found[1].label:      # ambiguous -> use screen position
                found.sort(key=lambda h: h.lm[0][0])
                left, right = found
            else:
                left = next(h for h in found if h.label == "Left")
                right = next(h for h in found if h.label == "Right")
            out.fingering, out.bow = (right, left) if self.swap else (left, right)
        elif len(found) == 1:
            h = found[0]
            if (h.label == "Left") != self.swap:
                out.fingering = h
            else:
                out.bow = h
        if self.use_pose:
            if self._n % cfg.POSE_EVERY_N_FRAMES == 0:
                self._run_pose(rgb)
            out.pose = self._pose_pts
        self._n += 1
        return out

    def draw(self, frame, hands: Hands) -> None:
        for h in (hands.fingering, hands.bow):
            if h is not None and h.raw is not None:
                self._draw.draw_landmarks(
                    frame, h.raw, self._conn,
                    self._draw.DrawingSpec(color=(0, 255, 200), thickness=1, circle_radius=2),
                    self._draw.DrawingSpec(color=(255, 255, 255), thickness=1))

    def close(self) -> None:
        self.hands.close()
        if self._pose is not None:
            self._pose.close()
