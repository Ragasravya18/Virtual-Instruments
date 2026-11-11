"""3D Virtual Violin - play violin, cello, guitar or theremin with your hands.

Run:  python main.py        (press H in the window for the controls)
"""
from __future__ import annotations

import sys
import time

import cv2

import config as cfg
from audio_engine import create_engine
from calibration import Calibration
from game_modes import create_modes
from hand_tracker import CameraStream, HandTracker
from instrument import VirtualInstrument
from recorder import Recorder
from renderer import Renderer


def main() -> int:
    cam = CameraStream(cfg.CAMERA_INDEX, cfg.FRAME_W, cfg.FRAME_H)
    if not cam.ok:
        print(f"Could not open camera {cfg.CAMERA_INDEX}. Try another CAMERA_INDEX in config.py.")
        return 1
    try:
        tracker = HandTracker()
    except Exception as exc:
        print(f"Hand tracking failed to start:\n{exc}")
        cam.release()
        return 1
    engine = create_engine()
    inst = VirtualInstrument()
    recorder = Recorder()
    modes = create_modes()
    mode_key = "free"
    calib = Calibration()
    fingers, bow = Calibration.load()
    inst.set_calibration(fingers, bow)

    first = cam.read(timeout=3.0)
    if first is None:
        print("The camera did not deliver any frames.")
        return 1
    h, w = first.shape[:2]
    renderer = Renderer(w, h)
    renderer.toast(f"Audio: {engine.name}   -   press H for help", 4)

    state = dict(help=False, reverb=False, fps=30.0)
    last = time.perf_counter()
    cv2.namedWindow(cfg.WINDOW_NAME)

    def apply_mode(key: str) -> None:
        nonlocal mode_key
        mode_key = key
        m = modes[key]
        wide = m.wide()
        if wide:
            inst.set_wide(*wide)
        else:
            inst.clear_wide()
        m.reset()
        renderer.toast(m.name + (f": {m.song_name}" if m.uses_song else ""))

    def select_instrument(name: str) -> None:
        inst.set_instrument(name)
        engine.set_instrument(name)
        renderer.toast(name.capitalize())

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            state["fps"] += 0.1 * ((1.0 / max(now - last, 1e-3)) - state["fps"])
            last = now

            hands = tracker.process(frame)
            tracker.draw(frame, hands)

            if recorder.playing:
                st = recorder.playback(now)
                if st is None:
                    renderer.toast("Playback finished")
                    st = inst.update(hands, now)
                elif st.instrument != engine.instrument:
                    engine.set_instrument(st.instrument)
                    inst.set_instrument(st.instrument)
            else:
                st = inst.update(hands, now)

            mode = modes[mode_key]
            if not calib.active:
                mode.update(st, now)
            engine.set_params(st.midi, st.audio_volume, st.vib_depth)
            if st.pluck:
                engine.pluck(st.midi, st.pluck_vol)
            if recorder.recording:
                recorder.add(now, st)
                if not recorder.recording:
                    renderer.toast("Recording limit reached")

            hud = dict(mode=mode.name, fps=state["fps"], snap=inst.snap, reverb=state["reverb"],
                       vib=inst.auto_vibrato, pizz=inst.pizz or not inst.profile["bowed"],
                       engine=engine.name, help=state["help"], rec=recorder.recording,
                       rec_time=recorder.duration(now), playing=recorder.playing,
                       banner=calib.banner())
            renderer.draw(frame, st, hands, hud, engine.scope())
            if mode_key != "free" and not calib.active:
                mode.draw(frame, st)
            renderer.draw_overlays(frame, hud)
            cv2.imshow(cfg.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or cv2.getWindowProperty(cfg.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 255:
                continue
            ch = chr(key).lower()
            if ch in cfg.INSTRUMENT_KEYS:
                select_instrument(cfg.INSTRUMENT_KEYS[ch])
            elif ch == "f":
                apply_mode("free")
            elif ch == "g":
                apply_mode("game")
            elif ch == "l":
                apply_mode("learn")
            elif ch in "[]" and mode.uses_song:
                mode.set_song(mode.song_idx + (1 if ch == "]" else -1))
                wide = mode.wide()
                if wide:
                    inst.set_wide(*wide)
                renderer.toast(mode.song_name)
            elif ch == " ":
                if calib.active:
                    res = calib.capture(hands)
                    if res == "done":
                        f, b = calib.result()
                        inst.set_calibration(f, b)
                        Calibration.save(f, b)
                        renderer.toast("Calibration saved")
                    elif res:
                        renderer.toast(res)
                elif mode.uses_song:
                    mode.start(now)
            elif ch == "k":
                if calib.active:
                    calib.active = False
                else:
                    calib.start()
            elif ch == "s":
                inst.snap = not inst.snap
                renderer.toast("Snap to notes " + ("ON" if inst.snap else "OFF (continuous pitch)"))
            elif ch == "r":
                state["reverb"] = not state["reverb"]
                engine.set_reverb(state["reverb"])
                renderer.toast("Reverb " + ("ON" if state["reverb"] else "OFF"))
            elif ch == "v":
                inst.auto_vibrato = not inst.auto_vibrato
                renderer.toast("Auto-vibrato " + ("ON" if inst.auto_vibrato else "OFF"))
            elif ch == "z":
                inst.pizz = not inst.pizz
                renderer.toast("Pizzicato (pinch / flick to pluck)" if inst.pizz else "Arco (bowed)")
            elif ch == "p":
                tracker.use_pose = not tracker.use_pose
                renderer.toast("Pose anchoring " + ("ON" if tracker.use_pose else "OFF"))
            elif ch == "w":
                tracker.swap = not tracker.swap
                renderer.toast("Hands swapped" if tracker.swap else "LEFT = notes, RIGHT = bow")
            elif ch == "h":
                state["help"] = not state["help"]
            elif ch == "e":
                if recorder.recording:
                    recorder.stop()
                    renderer.toast(f"Recorded {recorder.duration(now):.1f}s  (B play, M export)")
                else:
                    recorder.start(now)
                    renderer.toast("Recording...")
            elif ch == "b":
                renderer.toast("Playing back" if recorder.start_playback(now) else "Nothing recorded yet")
            elif ch == "m":
                path = recorder.export()
                renderer.toast(f"Saved {path.name}" if path else "Nothing to export")
    finally:
        engine.close()
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
