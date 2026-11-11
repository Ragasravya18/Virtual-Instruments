# 3D Virtual Violin

Play a violin, cello, guitar or theremin in the air with your hands and a webcam.
Built with OpenCV, MediaPipe and a built-in real-time synthesizer (no sound files needed).

## Quick start

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python main.py
```

Use Python 3.9 - 3.12. `mediapipe` is pinned to 0.10.14 because newer releases removed the
`mp.solutions` API this project uses.

## How to play

| Hand | Gesture | Effect |
|------|---------|--------|
| **LEFT** (fingering) | move index finger up / down | pitch (up = higher) |
| **RIGHT** (bow) | swipe sideways | bow - faster swipe = louder, stop = silence |
| **RIGHT** | height | which string (top = E, bottom = G) |
| **RIGHT** | pinch thumb + index, or fast flick | pluck (pizzicato / guitar) |
| **LEFT** | small up-down wobble | natural vibrato (auto-vibrato is added on held notes) |

Tips: stand about an arm's length from the camera in good light, show both hands, and press **K**
once to calibrate the pitch and string zones to your reach.

## Controls

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| 1 / 2 / 3 / 4 | violin / cello / guitar / theremin | F | free play |
| G | rhythm game | L | learn mode |
| `[` `]` | previous / next song | SPACE | start game / confirm calibration |
| S | snap to notes on/off (off = continuous pitch) | R | reverb |
| V | auto-vibrato | Z | pizzicato (pinch / flick) |
| K | calibrate | W | swap left/right hands |
| P | pose anchoring (violin follows your shoulder) | H | help overlay |
| E | record on/off | B | play back recording |
| M | export recording as `.mid` (saved in `recordings/`) | ESC | quit |

## Game modes

* **Free play** - sandbox with the pseudo-3D violin, vibrating strings, bow and rosin particles.
* **Rhythm game** - notes fall in lanes; hold the matching pitch (with the bow moving) while a note
  crosses the white line. Combos, grades (S-D) and high scores in `highscores.json`.
* **Learn mode** - shows the next target note and tells you "too high / too low / perfect".

In the game modes the pitch range is stretched over the song's notes so every note is reachable
without changing strings.

### Add your own songs
Edit `songs.json`:
```json
{ "My Song": { "tempo": 72, "notes": [[60, 1], [62, 1], [64, 2], [null, 1]] } }
```
Each entry is `[midi_note, beats]`; use `null` for a rest (middle C = 60).

## Realistic sound with a SoundFont (optional)

The built-in synth works out of the box. For sampled, studio-like instruments:

1. Install FluidSynth: Windows `choco install fluidsynth` (or the release zip from GitHub),
   macOS `brew install fluid-synth`, Linux `sudo apt install fluidsynth`.
2. `pip install -r requirements-optional.txt`
3. Put a General-MIDI SoundFont at `soundfonts/violin.sf2` (e.g. GeneralUser GS or FluidR3 GM).

The app picks it up automatically (bow speed -> MIDI expression, finger slides -> pitch bend).
If anything is missing it falls back to the built-in synth and tells you in the console.

## Project layout

```
main.py            app loop, key handling
config.py          every setting and instrument definition
hand_tracker.py    threaded camera + MediaPipe hands / pose
instrument.py      gesture -> note logic (bow speed, strings, vibrato, plucks)
audio_engine.py    built-in synth (additive + Karplus-Strong + reverb) and FluidSynth engine
renderer.py        3D-style violin, bow, strings, particles, HUD
game_modes.py      free play, rhythm game, learn mode
recorder.py        record, playback, MIDI export
calibration.py     4-step zone calibration (saved to calibration.json)
utils.py           music maths, One-Euro smoothing filter, MIDI writer
songs.json         song data
main_legacy.py     your original script (needs sounds/violin_*.mp3)
tests/             unit tests
```

## Tuning

Everything is in `config.py`:
* bow feels too sensitive / too stiff -> `BOW_DEADZONE`, `BOW_MAX_SPEED`
* notes flicker -> raise `SNAP_HYSTERESIS`; wrong left/right hand -> `SWAP_HANDS = True` (or press W)
* audio crackles -> raise `CHUNK` to 2048; lower latency -> `CHUNK = 512` (keep reverb delays larger than `CHUNK`)

## Troubleshooting

* **"Could not open camera"** - change `CAMERA_INDEX` (try 1), close other apps using the webcam,
  and on macOS allow camera access for your terminal.
* **MediaPipe error about `solutions`** - `pip install mediapipe==0.10.14`.
* **Pose anchoring does nothing** - the pose model is downloaded on first use, so it needs internet once.
* **No sound** - check the system volume / output device; run `python test_sound.py`-style checks with
  a short `pygame.mixer` test if needed.
* **Low FPS** - keep pose off (P), lower `FRAME_W/FRAME_H`, or close other camera apps.

## Tests

```bash
python -m unittest discover -s tests -v
```
