"""Run with:  python -m unittest discover -s tests -v   (from the project folder)"""
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as cfg
from audio_engine import SynthVoice, karplus_strong
from game_modes import RhythmGame, build_notes, grade_for, load_songs
from hand_tracker import Hand, Hands
from instrument import PlayState, VirtualInstrument
from recorder import Recorder
from utils import OneEuroFilter, midi_name, midi_to_hz, snap_with_hysteresis, vlq, write_midi


def fake_hand(tip_y=0.5, palm_x=0.5, palm_y=0.5):
    lm = [(0.5, 0.8)] * 21
    lm[0] = (palm_x, palm_y + 0.15)       # wrist
    lm[9] = (palm_x, palm_y)              # palm
    lm[8] = (0.4, tip_y)                  # index tip
    lm[4] = (0.6, 0.9)                    # thumb tip (not pinching)
    return Hand(lm)


def run_instrument(inst, frames):
    """frames: list of (finger_y, bow_x, bow_y). Returns the PlayState list."""
    out, t = [], 0.0
    for fy, bx, by in frames:
        t += 1 / 30
        out.append(inst.update(Hands(fingering=fake_hand(tip_y=fy), bow=fake_hand(palm_x=bx, palm_y=by)), t))
    return out


class TestMusicUtils(unittest.TestCase):
    def test_hz_and_names(self):
        self.assertAlmostEqual(midi_to_hz(69), 440.0)
        self.assertAlmostEqual(midi_to_hz(57), 220.0)
        self.assertEqual(midi_name(60), "C4")
        self.assertEqual(midi_name(69.2), "A4")

    def test_snap_hysteresis(self):
        self.assertEqual(snap_with_hysteresis(60.4, None, 0.18), 60.0)
        self.assertEqual(snap_with_hysteresis(60.6, 60.0, 0.18), 60.0)   # stays
        self.assertEqual(snap_with_hysteresis(60.9, 60.0, 0.18), 61.0)   # moves

    def test_one_euro_reduces_noise(self):
        rng = random.Random(1)
        f = OneEuroFilter()
        raw = [0.5 + rng.uniform(-0.01, 0.01) for _ in range(200)]
        sm = [f(v, i / 30) for i, v in enumerate(raw)]
        self.assertLess(np.std(sm[30:]), np.std(raw[30:]) * 0.6)

    def test_one_euro_follows_fast_motion(self):
        f = OneEuroFilter()
        for i in range(60):
            y = f(i / 60.0, i / 30)       # steady ramp
        self.assertLess(abs(y - 59 / 60.0), 0.08)

    def test_vlq(self):
        self.assertEqual(vlq(0), b"\x00")
        self.assertEqual(vlq(127), b"\x7f")
        self.assertEqual(vlq(128), b"\x81\x00")
        self.assertEqual(vlq(16384), b"\x81\x80\x00")

    def test_midi_file_structure(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.mid")
            write_midi(p, [(0.0, 0.5, 60, 90), (0.5, 1.0, 64, 90)])
            with open(p, "rb") as fh:
                data = fh.read()
        self.assertTrue(data.startswith(b"MThd"))
        self.assertIn(b"MTrk", data)
        self.assertTrue(data.endswith(b"\xff\x2f\x00"))
        self.assertEqual(data.count(b"\x90"), 2)


class TestAudio(unittest.TestCase):
    def test_voice_pitch_and_level(self):
        v = SynthVoice(44100)
        blocks = [v.render(1024, 440.0, 0.8) for _ in range(40)]
        sig = np.concatenate(blocks[20:])
        self.assertLessEqual(float(np.abs(sig).max()), 1.0)
        self.assertGreater(float(np.abs(sig).max()), 0.05)
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        peak_hz = np.argmax(spec) * 44100 / len(sig)
        self.assertAlmostEqual(peak_hz, 440.0, delta=8.0)

    def test_voice_goes_silent_when_bow_stops(self):
        v = SynthVoice(44100)
        for _ in range(10):
            v.render(1024, 440.0, 1.0)
        for _ in range(60):
            last = v.render(1024, 440.0, 0.0)
        self.assertLess(float(np.abs(last).max()), 1e-3)

    def test_reverb_keeps_block_size(self):
        v = SynthVoice(44100)
        v.reverb_on = True
        self.assertEqual(len(v.render(1024, 330.0, 0.7)), 1024)

    def test_karplus_strong_pitch(self):
        w = karplus_strong(220.0, dur=0.8)[2000:12000]
        ac = np.fft.irfft(np.abs(np.fft.rfft(w, 2 * len(w))) ** 2)[:len(w)]
        lo, hi = int(44100 / 1000), int(44100 / 80)
        lag = lo + int(np.argmax(ac[lo:hi]))
        self.assertAlmostEqual(44100 / lag, 220.0, delta=4.0)
        self.assertLessEqual(float(np.abs(karplus_strong(220.0)).max()), 1.0)


class TestInstrument(unittest.TestCase):
    def test_higher_hand_gives_higher_note(self):
        a = run_instrument(VirtualInstrument(), [(0.8, 0.5, 0.5)] * 5)[-1]
        b = run_instrument(VirtualInstrument(), [(0.2, 0.5, 0.5)] * 5)[-1]
        self.assertGreater(b.midi, a.midi)

    def test_bow_motion_makes_sound_and_stillness_does_not(self):
        inst = VirtualInstrument()
        moving = run_instrument(inst, [(0.5, 0.3 + 0.02 * i, 0.5) for i in range(20)])
        self.assertTrue(moving[-1].sounding)
        self.assertGreater(moving[-1].volume, 0.2)
        self.assertEqual(moving[-1].bow_dir, 1)
        still = run_instrument(inst, [(0.5, 0.8, 0.5)] * 20)
        self.assertFalse(still[-1].sounding)
        self.assertEqual(still[-1].audio_volume, 0.0)

    def test_faster_bow_is_louder(self):
        slow = run_instrument(VirtualInstrument(), [(0.5, 0.2 + 0.012 * i, 0.5) for i in range(20)])[-1]
        fast = run_instrument(VirtualInstrument(), [(0.5, 0.1 + 0.03 * i, 0.5) for i in range(20)])[-1]
        self.assertGreater(fast.volume, slow.volume)

    def test_bow_height_selects_string(self):
        hi = run_instrument(VirtualInstrument(), [(0.5, 0.5, 0.1)] * 5)[-1]
        lo = run_instrument(VirtualInstrument(), [(0.5, 0.5, 0.9)] * 5)[-1]
        self.assertEqual(hi.string_idx, 3)      # E string
        self.assertEqual(lo.string_idx, 0)      # G string

    def test_open_string_pitch(self):
        st = run_instrument(VirtualInstrument(), [(0.85, 0.5, 0.5)] * 5)[-1]   # hand at the bottom
        self.assertEqual(st.midi, cfg.INSTRUMENTS["violin"]["strings"][st.string_idx])

    def test_requires_both_hands(self):
        inst = VirtualInstrument()
        st = inst.update(Hands(fingering=fake_hand(), bow=None), 0.1)
        self.assertFalse(st.sounding)

    def test_guitar_plucks_on_flick(self):
        inst = VirtualInstrument()
        inst.set_instrument("guitar")
        frames = [(0.5, 0.2, 0.5)] * 5 + [(0.5, 0.2 + 0.1 * i, 0.5) for i in range(1, 8)]
        states = run_instrument(inst, frames)
        self.assertTrue(any(s.pluck for s in states))
        self.assertTrue(all(s.audio_volume == 0 for s in states))

    def test_wide_override_covers_song_range(self):
        inst = VirtualInstrument()
        inst.set_wide(59, 14)
        lo = run_instrument(inst, [(0.85, 0.5, 0.5)] * 5)[-1]
        inst2 = VirtualInstrument()
        inst2.set_wide(59, 14)
        hi = run_instrument(inst2, [(0.15, 0.5, 0.5)] * 5)[-1]
        self.assertEqual(lo.midi, 59)
        self.assertEqual(hi.midi, 73)


class TestGame(unittest.TestCase):
    def setUp(self):
        self.songs = load_songs()
        self._old_hs = cfg.HIGHSCORE_FILE
        cfg.HIGHSCORE_FILE = Path(tempfile.mkdtemp()) / "hs.json"

    def tearDown(self):
        cfg.HIGHSCORE_FILE = self._old_hs

    def test_songs_load(self):
        self.assertGreaterEqual(len(self.songs), 3)
        for s in self.songs.values():
            self.assertTrue(build_notes(s))

    def test_grades(self):
        self.assertEqual(grade_for(1.0), "S")
        self.assertEqual(grade_for(0.86), "A")
        self.assertEqual(grade_for(0.1), "D")

    def _play(self, perfect: bool):
        g = RhythmGame(self.songs)
        g.start(0.0)
        t = 0.0
        while g.state != "done" and t < 200:
            t += 1 / 30
            st = PlayState()
            cur = t - g.t0
            if perfect:
                active = [n for n in g.notes if n.start <= cur <= n.end]
                if active:
                    st.sounding, st.midi = True, float(active[0].midi)
            g.update(st, t)
        return g

    def test_perfect_play_scores_s(self):
        g = self._play(True)
        self.assertEqual(g.state, "done")
        self.assertEqual(g.grade, "S")
        self.assertGreater(g.max_combo, 10)

    def test_silence_scores_d(self):
        g = self._play(False)
        self.assertEqual(g.state, "done")
        self.assertEqual(g.grade, "D")
        self.assertEqual(g.score, 0)


class TestRecorder(unittest.TestCase):
    def test_record_playback_export(self):
        r = Recorder()
        r.start(0.0)
        for i in range(60):
            st = PlayState(midi=60.0 + (i // 20), audio_volume=0.7, volume=0.7, sounding=True)
            r.add(i / 30, st)
        r.stop()
        notes = r.to_notes()
        self.assertEqual([n[2] for n in notes], [60, 61, 62])
        with tempfile.TemporaryDirectory() as d:
            p = r.export(Path(d) / "x.mid")
            self.assertTrue(p.exists() and p.stat().st_size > 20)
        self.assertTrue(r.start_playback(10.0))
        self.assertTrue(r.playback(10.5).sounding)
        self.assertIsNone(r.playback(20.0))


if __name__ == "__main__":
    unittest.main()
