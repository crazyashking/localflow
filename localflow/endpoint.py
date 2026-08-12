"""Decide when a voice-started utterance has finished.

Push-to-talk gets this for free: you let go of the key. A recording started by
the wake word has no such signal, so something has to watch the microphone and
close the utterance when you stop talking.

The whole design follows from the costs being lopsided. Closing too early
splits a sentence in half, injects the fragment, and makes the user re-trigger
and re-speak, which costs five to ten seconds and a broken line of text.
Closing too late costs exactly the extra wait, once, and is always recoverable.
So this errs toward waiting, and the default is deliberately longer than feels
snappy in a demo.

Hysteresis rather than one threshold, for the same reason overlay.py uses it:
a voice hovering near a single cut-off flickers across it between syllables,
and every flicker would reset the silence timer or start it early. Speech has
to clear the higher bar to register, and drop below the lower one to count as
silence.
"""

from __future__ import annotations

# Both are read on the same 0..1 curved scale the overlay meter uses, since
# they consume the same Recorder.live_level. Kept equal to the overlay's own
# band on purpose: the moment the capsule turns pink is the moment this counts
# you as speaking, so what you see matches what the endpointer believes.
SPEECH_ON = 0.20
SPEECH_OFF = 0.09

# Seconds of continuous silence that closes an utterance. Two seconds rather
# than the more common 1.5, because LocalFlow is used to compose text rather
# than to read it aloud, and composition pauses (hunting for a word, deciding
# how to phrase something) sit squarely in the 1.5 to 2.5 second range. At 22x
# real time decode the extra half second is roughly 27% more perceived wait,
# which buys back its cost if it prevents one split utterance in twenty.
DEFAULT_SILENCE_SECONDS = 2.0

# Never close before this much has elapsed, no matter how quiet it is. The wake
# word fires as the phrase ends, and there is a real gap before the first word
# arrives while the user draws breath. Without a floor, a slow start would be
# read as silence and close the recording before it began.
MIN_UTTERANCE_SECONDS = 1.0


class SilenceEndpointer:
    """Tracks talking against silence and says when the utterance is over.

    Feed it the live microphone level. It is a plain state machine with no
    thread and no timer of its own, so it can be driven from the app's watcher
    thread in production and from a list of numbers in a test.
    """

    def __init__(
        self,
        silence_seconds: float = DEFAULT_SILENCE_SECONDS,
        min_utterance_seconds: float = MIN_UTTERANCE_SECONDS,
    ):
        self.silence_seconds = silence_seconds
        self.min_utterance_seconds = min_utterance_seconds

        self._started = 0.0
        self._last_speech = 0.0
        self._speaking = False
        self._heard_speech = False

    def reset(self, now: float) -> None:
        """Arm the endpointer at the moment recording starts."""
        self._started = now
        self._last_speech = now
        self._speaking = False
        self._heard_speech = False

    def update(self, level: float, now: float) -> bool:
        """Feed one level sample. Returns True when the utterance should close."""
        if level >= SPEECH_ON:
            self._speaking = True
            self._heard_speech = True
            self._last_speech = now
        elif level > SPEECH_OFF and self._speaking:
            # Inside the band while already speaking, so this still counts as
            # talking. This is the whole point of the hysteresis: the quiet
            # part of a word cannot start the silence clock.
            self._last_speech = now
        else:
            self._speaking = False

        if now - self._started < self.min_utterance_seconds:
            return False
        return now - self._last_speech >= self.silence_seconds

    @property
    def heard_speech(self) -> bool:
        """True if anything above the speech threshold was ever seen.

        Lets the caller tell "you spoke and finished" apart from "the wake word
        fired and nobody said anything", which are worth reporting differently.
        """
        return self._heard_speech
