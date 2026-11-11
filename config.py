"""Central configuration for the 3D Virtual Violin.

Everything you may want to tweak lives here: camera, audio engine, gesture
thresholds and the instrument definitions.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- camera
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
WINDOW_NAME = "3D Virtual Violin"

# ---------------------------------------------------------------- hands
SWAP_HANDS = False        # False: LEFT hand fingers the notes, RIGHT hand bows
HAND_CONFIDENCE = 0.6
USE_POSE = False          # anchor the violin to your shoulder/chin (toggle with P)
POSE_EVERY_N_FRAMES = 2

# ---------------------------------------------------------------- audio
ENGINE = "auto"           # "auto" | "synth" | "fluid"
SAMPLE_RATE = 44100
CHUNK = 1024              # synth block size (>= 1024 keeps the reverb valid)
MIXER_BUFFER = 512
MASTER_GAIN = 0.55
SOUNDFONT = ROOT / "soundfonts" / "violin.sf2"   # any General-MIDI .sf2 works
AUTO_VIBRATO_DEPTH = 0.22  # semitones, added after a note is sustained

# ---------------------------------------------------------------- bow
BOW_DEADZONE = 0.10       # frame-widths/second below which the bow is "stopped"
BOW_MAX_SPEED = 1.30      # frame-widths/second that gives full volume
BOW_SMOOTH = 0.55         # 0..1, higher = more responsive, lower = smoother
FLICK_SPEED = 1.10        # fast flick of the bow hand plucks (pizz / guitar)
PINCH_RATIO = 0.28        # thumb-index distance / palm size that counts as a pinch

# ---------------------------------------------------------------- fingering
SNAP_DEFAULT = True       # snap to the nearest semitone (toggle with S)
SNAP_HYSTERESIS = 0.18
# fraction of frame height: (y of the highest note, y of the lowest note)
DEFAULT_FINGER_RANGE = (0.15, 0.85)
# fraction of frame height: (y of the top string, y of the bottom string)
DEFAULT_BOW_RANGE = (0.20, 0.80)

# ---------------------------------------------------------------- rhythm game
COUNT_IN_SECONDS = 3.0
FALL_PIXELS_PER_SECOND = 170

# ---------------------------------------------------------------- instruments
# strings: MIDI note of each open string, low -> high
# span:    semitones reachable on a string by moving the fingering hand
# rolloff: harmonic roll-off exponent (soft, loud); larger = duller / purer
# body:    (centre Hz, gain) body-resonance peaks for the built-in synth
# gm:      General-MIDI program number (used by the FluidSynth engine)
INSTRUMENTS = {
    "violin": dict(strings=[55, 62, 69, 76], span=12, bowed=True, wide=False,
                   rolloff=(1.7, 0.7), noise=0.020, body=[(300, 1.2), (550, 1.0), (1200, 0.6), (2600, 0.5)],
                   gm=40, vibrato=True, pluck_decay=0.994),
    "cello":  dict(strings=[36, 43, 50, 57], span=12, bowed=True, wide=False,
                   rolloff=(2.0, 0.9), noise=0.030, body=[(120, 1.4), (230, 1.0), (500, 0.8)],
                   gm=42, vibrato=True, pluck_decay=0.996),
    "guitar": dict(strings=[40, 45, 50, 55, 59, 64], span=12, bowed=False, wide=False,
                   rolloff=(1.5, 0.8), noise=0.0, body=[(200, 1.0)],
                   gm=24, vibrato=False, pluck_decay=0.997),
    "theremin": dict(strings=[57], span=24, bowed=True, wide=True,
                     rolloff=(6.0, 5.0), noise=0.0, body=[],
                     gm=40, vibrato=False, pluck_decay=0.994),
}
INSTRUMENT_KEYS = {"1": "violin", "2": "cello", "3": "guitar", "4": "theremin"}

# ---------------------------------------------------------------- files
SONGS_FILE = ROOT / "songs.json"
HIGHSCORE_FILE = ROOT / "highscores.json"
CALIBRATION_FILE = ROOT / "calibration.json"
RECORDINGS_DIR = ROOT / "recordings"
