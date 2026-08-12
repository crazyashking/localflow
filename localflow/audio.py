"""Microphone capture.

Captures at 16kHz mono float32, which is Whisper's native input format. That
means no resampling and no ffmpeg dependency: the numpy array goes straight
into faster-whisper.

The stream is opened once at startup and left open. Recording is just a flag
that starts filling a ring buffer, so press-to-first-sample latency is
effectively zero rather than paying stream-open cost on every utterance.
"""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import sounddevice as sd

from .models import SAMPLE_RATE

CHANNELS = 1
BLOCKSIZE = 1024  # ~64ms at 16kHz

# Audio kept on hand while idle, so a recording can start slightly in the past.
# The wake word detector fires a few hundred ms after the phrase ends, and
# people start their sentence immediately, so without this the first word is
# already gone by the time recording begins.
#
# This is only the size of the buffer. How far back a recording actually
# reaches is chosen per call, because reaching back the full second puts the
# wake phrase itself into the recording and Whisper duly types "hey flow" at
# the start of every sentence. See begin().
PREROLL_SECONDS = 1.0

# RMS that reads as a full-height meter. Speech sits roughly between 0.02 and
# 0.15 depending on the mic, the gain and how close you sit, so this is the one
# number worth exposing: too high and the meter barely twitches on a quiet mic.
DEFAULT_LEVEL_CEILING = 0.14

# Curve applied after scaling. Below 1.0 it lifts quiet sounds, which is what
# gives small speech visible travel instead of leaving it pinned near the floor.
LEVEL_CURVE = 0.55


class MicError(RuntimeError):
    pass


class Recorder:
    def __init__(
        self,
        device: str | int | None = None,
        max_seconds: float = 120.0,
        min_seconds: float = 0.3,
        level_ceiling: float = DEFAULT_LEVEL_CEILING,
        on_block: Callable[[np.ndarray], None] | None = None,
    ):
        self.device = _resolve_device(device)
        # Called with every captured block, recording or not, so the wake word
        # detector can listen without opening a second stream on the same
        # device. It runs on the PortAudio thread, so whatever is passed here
        # must only enqueue and return. See the note in _callback.
        self.on_block = on_block
        self.max_seconds = max_seconds
        self.min_seconds = min_seconds
        # Guarded: a zero or negative ceiling from settings.json would divide by
        # zero on every audio block, inside the PortAudio callback.
        self.level_ceiling = max(float(level_ceiling), 1e-3)

        self._lock = threading.Lock()
        self._frames: deque[np.ndarray] = deque()
        # Rolling window of the most recent idle audio, for begin(preroll=True).
        # maxlen means old blocks fall off on their own, with no bookkeeping.
        self._preroll: deque[np.ndarray] = deque(
            maxlen=max(1, int(PREROLL_SECONDS * SAMPLE_RATE / BLOCKSIZE))
        )
        self._recording = False
        self._n_samples = 0
        self._peak = 0.0
        self._stream: sd.InputStream | None = None
        self._overflowed = False
        self._hit_cap = False
        # Smoothed instantaneous loudness, updated on every block whether or
        # not we are recording, so the overlay can show a live meter.
        self._live_level = 0.0

    # --- lifecycle -----------------------------------------------------

    def open(self) -> None:
        if self._stream is not None:
            return
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=BLOCKSIZE,
                device=self.device,
                callback=self._callback,
                # The callback is Python, so it needs the GIL to run. Anything
                # else holding the GIL, notably the overlay redrawing, can
                # delay it long enough for PortAudio to drop incoming samples,
                # which punches holes in the recording and loses whole words.
                # High latency buys a much deeper buffer so a late callback is
                # harmless instead of lossy.
                latency="high",
            )
            self._stream.start()
        except Exception as exc:  # sounddevice raises a variety of types
            raise MicError(f"could not open microphone: {exc}") from exc

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    # --- capture -------------------------------------------------------

    def _callback(self, indata, frames, time_info, status) -> None:
        # This runs on the PortAudio thread. Keep it cheap: copy and return.
        # Only input_overflow matters: it means PortAudio dropped incoming
        # samples, so the recording has a hole in it and the transcription will
        # be wrong for no visible reason. Other status flags are not about lost
        # input and would only produce noise.
        if status.input_overflow:
            self._overflowed = True

        block = indata[:, 0].copy()

        # Level tracking runs even when not recording, so the overlay can show
        # live input. Fast attack and slow release is what makes a meter look
        # responsive to speech but not jittery: it snaps up on a syllable and
        # eases back down instead of flickering.
        block_rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        alpha = 0.55 if block_rms > self._live_level else 0.12
        self._live_level += (block_rms - self._live_level) * alpha

        # The wake word detector's tap. Contained, because an exception raised
        # here would propagate into PortAudio's callback and kill the stream,
        # taking dictation down with it over a fault in an optional feature.
        if self.on_block is not None:
            with contextlib.suppress(Exception):
                self.on_block(block)

        if not self._recording:
            # Idle audio is kept so begin(preroll=True) can reach backwards.
            # Under the lock because begin() drains this from another thread,
            # and iterating a deque while it is appended to raises. An
            # uncontended lock costs tens of nanoseconds against a 64ms budget.
            with self._lock:
                self._preroll.append(block)
            return
        with self._lock:
            if self._n_samples >= self.max_seconds * SAMPLE_RATE:
                # Past the cap, keep discarding but record that we did. Losing
                # audio without saying so is the worst possible behaviour: the
                # user gets a transcription that just stops, with no clue why.
                self._hit_cap = True
                return
            self._frames.append(block)
            self._n_samples += len(block)
            peak = float(np.abs(block).max())
            if peak > self._peak:
                self._peak = peak

    def _recent(self, seconds: float | None) -> list[np.ndarray]:
        """The tail of the idle buffer, at most `seconds` long.

        Whole blocks only. Cutting mid-block to hit an exact duration would
        save at most 64ms and cost a partial frame at the join, and the caller's
        number is an estimate of detector latency rather than a precise edge.
        Caller holds the lock.
        """
        if seconds is None:
            return list(self._preroll)
        wanted = max(0, int(seconds * SAMPLE_RATE))
        kept: list[np.ndarray] = []
        total = 0
        for block in reversed(self._preroll):
            if total >= wanted:
                break
            kept.append(block)
            total += len(block)
        kept.reverse()
        return kept

    def begin(self, preroll: bool = False,
              preroll_seconds: float | None = None) -> None:
        """Start accumulating audio. Safe to call when already recording.

        With preroll=True the recording opens with recent idle audio already in
        it. That is for the wake word, which only knows the phrase was spoken
        after it has finished, by which point the user is usually a word into
        their sentence.

        How far back to reach matters more than it looks. The window has to
        cover the detector's latency, so a word begun the instant the phrase
        ended is not clipped, and it has to stop short of the phrase itself,
        because audio containing "hey flow" gets transcribed and typed like any
        other speech. preroll_seconds is that dividing line; None takes the
        whole buffer, which is almost always too much.

        Push-to-talk passes False on purpose. The key goes down before you
        speak, so there is nothing to recover, and reaching backwards would
        only fold in room noise or the tail of whatever was said last.
        """
        if self._stream is None:
            self.open()
        with self._lock:
            self._frames.clear()
            if preroll:
                self._frames.extend(self._recent(preroll_seconds))
            self._preroll.clear()
            self._n_samples = sum(len(f) for f in self._frames)
            self._peak = 0.0
            self._overflowed = False
            self._hit_cap = False
            self._recording = True

    def end(self) -> np.ndarray | None:
        """Stop and return the captured audio.

        Returns None for anything shorter than min_seconds, which filters out
        accidental taps of the hotkey.
        """
        with self._lock:
            self._recording = False
            frames = list(self._frames)
            self._frames.clear()
            n = self._n_samples
            self._n_samples = 0

        if not frames or n < self.min_seconds * SAMPLE_RATE:
            return None
        return np.concatenate(frames).astype(np.float32, copy=False)

    # --- introspection -------------------------------------------------

    @property
    def peak(self) -> float:
        """Loudest sample seen this utterance, for a level meter."""
        return self._peak

    @property
    def live_level(self) -> float:
        """Smoothed current loudness as 0..1, for the overlay animation.

        Scaled against level_ceiling and curved, so quiet speech still produces
        visible travel rather than sitting flat against the floor. The ceiling
        was 0.28 with a 0.65 curve, which put ordinary speech at roughly a
        third of the meter's height: the display technically worked and looked
        broken, which for a level meter is the same thing.
        """
        scaled = min(self._live_level / self.level_ceiling, 1.0)
        return scaled**LEVEL_CURVE if scaled > 0 else 0.0

    @property
    def recording(self) -> bool:
        """True while audio is being accumulated."""
        return self._recording

    @property
    def overflowed(self) -> bool:
        """True if PortAudio reported dropped input this utterance."""
        return self._overflowed

    @property
    def hit_cap(self) -> bool:
        """True if this utterance ran past max_seconds and lost the remainder."""
        return self._hit_cap


class ClipStats(NamedTuple):
    peak: float
    rms: float
    clipped_fraction: float

    @property
    def is_clipping(self) -> bool:
        """Clipping squares off the waveform and destroys information.

        Whisper copes with quiet audio far better than with clipped audio, so
        this is worth surfacing rather than transcribing damaged input and
        blaming the model for the result.
        """
        return self.clipped_fraction > 0.001

    @property
    def is_quiet(self) -> bool:
        return self.peak < 0.03

    def summary(self) -> str:
        if self.is_clipping:
            return f"CLIPPING ({self.clipped_fraction:.1%} of samples maxed out)"
        if self.is_quiet:
            return f"very quiet (peak {self.peak:.3f})"
        return f"level ok (peak {self.peak:.2f})"


def analyse(clip: np.ndarray) -> ClipStats:
    """Measure level and clipping on a captured utterance."""
    if clip.size == 0:
        return ClipStats(0.0, 0.0, 0.0)
    absolute = np.abs(clip)
    return ClipStats(
        peak=float(absolute.max()),
        rms=float(np.sqrt(np.mean(clip.astype(np.float64) ** 2))),
        clipped_fraction=float(np.count_nonzero(absolute >= 0.999) / absolute.size),
    )


def _resolve_device(device: str | int | None) -> int | None:
    """Turn a device-name substring into an index, or pass through."""
    if device is None or isinstance(device, int):
        return device
    needle = str(device).lower()
    for idx, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0 and needle in info["name"].lower():
            return idx
    raise MicError(
        f"no input device matching {device!r}. Available inputs:\n"
        + "\n".join(
            f"  [{i}] {d['name']}"
            for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0
        )
    )


def list_input_devices() -> list[tuple[int, str]]:
    return [
        (i, d["name"])
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]
