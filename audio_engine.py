"""Audio engines.

SynthEngine  - built-in real-time additive synth (numpy + pygame). No files needed.
FluidEngine  - optional, uses pyfluidsynth + a General-MIDI SoundFont (.sf2) for
               the most realistic sound.

Both expose the same small API:
    set_instrument(name)
    set_params(midi, volume, vib_depth)   # called every frame (bowed notes)
    pluck(midi, volume)                   # plucked / pizzicato notes
    set_reverb(on)
    scope()                               # recent waveform for the HUD
    close()
"""
from __future__ import annotations

import math
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

import config as cfg
from utils import clamp, midi_to_hz

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------- DSP
class Reverb:
    """Tiny feedback-comb reverb. Delay lengths must be >= the block size."""

    def __init__(self, delays=(1559, 1877, 2161, 2503), gains=(0.80, 0.78, 0.76, 0.74)):
        self.delays, self.gains = delays, gains
        self.bufs = [np.zeros(d, np.float32) for d in delays]
        self.idx = [0] * len(delays)

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        wet = np.zeros(n, np.float32)
        for i, (buf, d, g) in enumerate(zip(self.bufs, self.delays, self.gains)):
            ix = (self.idx[i] + np.arange(n)) % d
            y = x + g * buf[ix]
            buf[ix] = y
            self.idx[i] = (self.idx[i] + n) % d
            wet += y
        return 0.8 * x + 0.12 * wet


def karplus_strong(hz: float, dur: float = 1.4, sr: int = 44100,
                   decay: float = 0.995, brightness: float = 0.5) -> np.ndarray:
    """Plucked-string sound (vectorised Karplus-Strong)."""
    n = max(2, int(round(sr / hz)))
    total = int(dur * sr)
    rng = np.random.default_rng(1)
    prev = rng.uniform(-1, 1, n).astype(np.float32)
    out = np.empty(total + n, np.float32)
    out[:n] = prev
    pos = n
    while pos < total:
        new = decay * (brightness * prev + (1 - brightness) * np.roll(prev, 1))
        out[pos:pos + n] = new
        prev, pos = new, pos + n
    out = out[:total]
    fade = int(0.08 * sr)
    out[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    peak = float(np.max(np.abs(out))) or 1.0
    return out * (0.9 / peak)


class SynthVoice:
    """One monophonic bowed voice, rendered block by block (phase-continuous)."""

    MAX_HARMONICS = 24

    def __init__(self, sr: int = cfg.SAMPLE_RATE):
        self.sr = sr
        self.profile = cfg.INSTRUMENTS["violin"]
        self.phase = self.phase2 = self.lfo = 0.0
        self.f = 220.0
        self.amp = 0.0
        self.reverb = Reverb()
        self.reverb_on = False
        self._noise_rng = np.random.default_rng(7)

    def render(self, n: int, target_hz: float, target_amp: float, vib_depth: float = 0.0,
               vib_hz: float = 5.5) -> np.ndarray:
        prof, sr = self.profile, self.sr
        target_amp = clamp(target_amp)

        # envelope (fast attack, slower release), interpolated across the block
        coef = 0.45 if target_amp > self.amp else 0.16
        new_amp = self.amp + (target_amp - self.amp) * coef
        if target_amp == 0.0 and new_amp < 1e-3:
            new_amp = 0.0
        amp = np.linspace(self.amp, new_amp, n, endpoint=False, dtype=np.float32)
        was_silent = self.amp == 0.0
        self.amp = new_amp

        # glide towards the target pitch (legato); jump if starting from silence
        if was_silent:
            self.f = target_hz
        new_f = self.f + (target_hz - self.f) * 0.5
        f = np.linspace(self.f, new_f, n, endpoint=False)
        self.f = new_f

        # vibrato
        lfo = self.lfo + TWO_PI * vib_hz / sr * np.arange(1, n + 1)
        self.lfo = float(lfo[-1] % TWO_PI)
        if vib_depth:
            f = f * 2.0 ** (vib_depth * np.sin(lfo) / 12.0)

        ph = self.phase + TWO_PI * np.cumsum(f) / sr
        ph2 = self.phase2 + TWO_PI * np.cumsum(f * 1.0017) / sr   # slightly detuned twin
        self.phase, self.phase2 = float(ph[-1] % TWO_PI), float(ph2[-1] % TWO_PI)

        # harmonic spectrum: brighter when played louder
        f0 = max(new_f, 20.0)
        k = np.arange(1, int(min(self.MAX_HARMONICS, 0.45 * sr // f0)) + 1)
        soft, loud = prof["rolloff"]
        p = soft + (loud - soft) * target_amp
        a = k.astype(np.float64) ** -p
        for centre, gain in prof["body"]:
            a *= 1.0 + gain * np.exp(-(((k * f0) - centre) / (centre * 0.35)) ** 2)
        a = (a / a.sum()).astype(np.float32)

        sig = a @ np.sin(np.outer(k, ph)).astype(np.float32)
        sig2 = a @ np.sin(np.outer(k, ph2)).astype(np.float32)
        out = 0.72 * sig + 0.28 * sig2   # gentle chorus, never cancels the fundamental

        if prof["noise"]:  # bow hiss
            noise = self._noise_rng.standard_normal(n).astype(np.float32)
            out += prof["noise"] * np.convolve(noise, np.ones(6, np.float32) / 6, "same")

        out = out * amp * cfg.MASTER_GAIN
        if self.reverb_on:
            out = self.reverb.process(out.astype(np.float32))
        return np.tanh(out).astype(np.float32)


# ------------------------------------------------------------ pygame engine
class SynthEngine:
    name = "Built-in synth"

    def __init__(self):
        import pygame
        self.pg = pygame
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(cfg.SAMPLE_RATE, -16, 1, cfg.MIXER_BUFFER)
            pygame.mixer.init()
        pygame.mixer.set_num_channels(8)
        self._channels = pygame.mixer.get_init()[2]
        self._stream = pygame.mixer.Channel(0)
        self._pluck_i = 0
        self._pluck_cache: Dict[Tuple[str, int], "pygame.mixer.Sound"] = {}
        self.voice = SynthVoice()
        self.instrument = "violin"
        self._params = (220.0, 0.0, 0.0)
        self._scope = np.zeros(256, np.float32)
        self._lock = threading.Lock()
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---- control API
    def set_instrument(self, name: str) -> None:
        self.instrument = name
        self.voice.profile = cfg.INSTRUMENTS[name]
        self._pluck_cache.clear()

    def set_params(self, midi: float, volume: float, vib_depth: float = 0.0) -> None:
        with self._lock:
            self._params = (midi_to_hz(midi), volume, vib_depth)

    def pluck(self, midi: float, volume: float = 0.8) -> None:
        key = (self.instrument, int(round(midi * 4)))
        snd = self._pluck_cache.get(key)
        if snd is None:
            decay = cfg.INSTRUMENTS[self.instrument]["pluck_decay"]
            wave = karplus_strong(midi_to_hz(round(midi * 4) / 4), 1.6, cfg.SAMPLE_RATE, decay)
            snd = self.pg.mixer.Sound(buffer=self._to_pcm(wave).tobytes())
            self._pluck_cache[key] = snd
        snd.set_volume(clamp(volume) * 0.9)
        self._pluck_i = self._pluck_i % 6 + 1
        self.pg.mixer.Channel(self._pluck_i).play(snd)

    def set_reverb(self, on: bool) -> None:
        self.voice.reverb_on = on

    def scope(self) -> np.ndarray:
        return self._scope

    def close(self) -> None:
        self._run = False
        self._thread.join(timeout=0.5)
        self.pg.mixer.quit()

    # ---- internals
    def _to_pcm(self, mono: np.ndarray) -> np.ndarray:
        pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
        if self._channels == 2:
            pcm = np.column_stack([pcm, pcm])
        return np.ascontiguousarray(pcm)

    def _loop(self) -> None:
        ch = self._stream
        while self._run:
            busy = ch.get_busy()
            if busy and ch.get_queue() is not None:
                time.sleep(0.002)
                continue
            with self._lock:
                hz, amp, vib = self._params
            block = self.voice.render(cfg.CHUNK, hz, amp, vib)
            self._scope = block[::cfg.CHUNK // 256][:256].copy()
            snd = self.pg.sndarray.make_sound(self._to_pcm(block))
            if not busy:
                ch.play(snd)
            else:
                ch.queue(snd)


# ----------------------------------------------------------- FluidSynth engine
class FluidEngine:
    name = "FluidSynth + SoundFont"

    def __init__(self, soundfont):
        import fluidsynth  # pip install pyfluidsynth (needs the FluidSynth library)
        self.fs = fluidsynth.Synth(gain=0.9)
        self.fs.start()
        self.sfid = self.fs.sfload(str(soundfont))
        if self.sfid < 0:
            raise RuntimeError(f"could not load SoundFont {soundfont}")
        self.instrument = "violin"
        self.cur: Optional[int] = None
        self._t0 = time.perf_counter()
        self._select("violin")

    def _select(self, name: str) -> None:
        self.fs.program_select(0, self.sfid, 0, cfg.INSTRUMENTS[name]["gm"])

    def set_instrument(self, name: str) -> None:
        self._release()
        self.instrument = name
        self._select(name)

    def _release(self) -> None:
        if self.cur is not None:
            self.fs.noteoff(0, self.cur)
            self.cur = None

    def set_params(self, midi: float, volume: float, vib_depth: float = 0.0) -> None:
        if volume < 0.03:
            self._release()
            self.fs.cc(0, 11, 0)
            return
        if self.cur is None or abs(midi - self.cur) > 1.6:
            self._release()
            self.cur = int(round(midi))
            self.fs.cc(0, 11, int(clamp(volume) * 127))
            self.fs.noteon(0, self.cur, 100)
        bend = midi - self.cur
        if vib_depth:
            bend += vib_depth * math.sin(TWO_PI * 5.5 * (time.perf_counter() - self._t0))
        self.fs.pitch_bend(0, int(clamp(bend / 2.0, -1.0, 1.0) * 8191))
        self.fs.cc(0, 11, int(clamp(volume) * 127))

    def pluck(self, midi: float, volume: float = 0.8) -> None:
        self.fs.noteon(0, int(round(midi)), int(40 + 87 * clamp(volume)))

    def set_reverb(self, on: bool) -> None:
        try:
            self.fs.set_reverb(0.7, 0.4, 0.8, 0.6 if on else 0.0)
        except Exception:
            pass

    def scope(self) -> np.ndarray:
        return np.zeros(256, np.float32)

    def close(self) -> None:
        self.fs.delete()


def create_engine(mode: str = cfg.ENGINE):
    """Pick the best available engine, falling back to the built-in synth."""
    if mode in ("auto", "fluid") and cfg.SOUNDFONT.exists():
        try:
            return FluidEngine(cfg.SOUNDFONT)
        except Exception as exc:  # missing library, bad SoundFont, no audio driver...
            print(f"[audio] FluidSynth unavailable ({exc}); using the built-in synth.")
    elif mode == "fluid":
        print(f"[audio] SoundFont not found at {cfg.SOUNDFONT}; using the built-in synth.")
    return SynthEngine()
