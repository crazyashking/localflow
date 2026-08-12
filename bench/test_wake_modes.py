"""Drive all three wake-word end modes without a microphone or a voice.

The wake word itself is measured in test_wake.py. What this file checks is the
state machine around it, which is where the bugs actually live: one key that
means "start" while idle and "finish" while recording, a release event that
must not be processed twice, and a detector that has to stop listening while
its own output is being dictated.

Injection, transcription and the detector are all stubbed. Audio is pushed
through the real recorder callback, so the pre-roll and the level tracking run
for real.

Run:  .venv\\Scripts\\python.exe bench\\test_wake_modes.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import wake  # noqa: E402
from localflow.app import DictationApp  # noqa: E402
from localflow.config import DEFAULTS, Settings  # noqa: E402
from localflow.endpoint import SilenceEndpointer  # noqa: E402
from localflow.models import SAMPLE_RATE  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures += 1


class _Status:
    """Stand-in for PortAudio's CallbackFlags."""

    input_overflow = False


class _StubModel:
    """Stand-in for openwakeword.model.Model, scoring nothing."""

    def reset(self) -> None:
        pass


def speech(n: int = 1024, amplitude: float = 0.25) -> np.ndarray:
    """A block loud enough to clear the endpointer's speech threshold."""
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 180.0 * t)).reshape(-1, 1)


def quiet(n: int = 1024) -> np.ndarray:
    return np.zeros((n, 1), dtype=np.float32)


def build(**overrides) -> tuple[DictationApp, list]:
    """An app with the wake word on, but no model and no GPU."""
    settings = Settings({
        **DEFAULTS,
        "wake_word": True,
        "save_transcripts": False,
        "overlay": False,
        **overrides,
    })
    queued: list = []
    application = DictationApp(settings, on_status=lambda s, d: None)
    # The detector needs a loaded-looking model for mute/unmute bookkeeping,
    # but must never load one or score anything here. reset() exists because
    # unmute() calls it to clear openWakeWord's internal audio history.
    application.wake._model = _StubModel()
    application.wake._model_name = "stub"
    # Capture what would have been transcribed instead of touching the GPU.
    application._jobs = type("Q", (), {"put": lambda _self, clip: queued.append(clip)})()
    return application, queued


# --- the endpointer on its own ---------------------------------------------

print("endpointer")

ep = SilenceEndpointer(silence_seconds=2.0, min_utterance_seconds=1.0)
ep.reset(0.0)
check("silence alone does not close before the floor", not ep.update(0.0, 0.9))
check("silence closes once past the floor and the timeout", ep.update(0.0, 2.5))

ep.reset(0.0)
ep.update(0.5, 0.5)          # speaking
check("speech holds it open", not ep.update(0.5, 3.0))
check("and it closes 2s after the last word", ep.update(0.0, 5.01),
      "(spoke at t=3.0, closed at t=5.01)")

# The whole reason for hysteresis: a level inside the band, between syllables,
# must not start the silence clock while the user is clearly still talking.
ep.reset(0.0)
ep.update(0.5, 0.5)
for i in range(1, 40):
    ep.update(0.15, 0.5 + i * 0.05)   # in the band, below SPEECH_ON
check("a level inside the hysteresis band counts as still talking",
      not ep.update(0.15, 2.4))

ep.reset(0.0)
ep.update(0.0, 3.0)
check("heard_speech is False when nobody spoke", not ep.heard_speech)
ep.reset(0.0)
ep.update(0.5, 0.5)
check("heard_speech is True after real speech", ep.heard_speech)

# --- mode: hotkey ------------------------------------------------------------

print("\nwake_end_mode = hotkey")

app, queued = build(wake_end_mode="hotkey")
app.recorder._stream = object()          # pretend the stream is open
app._on_wake()
check("wake starts a recording", app.recorder.recording)
check("detector muted while recording", app.wake.muted)

for _ in range(30):
    app.recorder._callback(speech(), 1024, None, _Status())
app._on_press()                           # the tap that ends it
check("hotkey tap ended the recording", not app.recorder.recording)
check("clip was queued", len(queued) == 1, f"({len(queued)})")
check("detector listening again", not app.wake.muted)

# The key-up after that tap must not be processed as a push-to-talk release.
app._on_release()
check("the following key-up is swallowed", len(queued) == 1, f"({len(queued)})")
check("and it did not start anything", not app.recorder.recording)

# --- mode: silence -----------------------------------------------------------

print("\nwake_end_mode = silence")

app, queued = build(wake_end_mode="silence", wake_endpoint_silence=0.4)
app.recorder._stream = object()
app._on_wake()
for _ in range(20):
    app.recorder._callback(speech(), 1024, None, _Status())

# Drive the watcher's decision directly rather than sleeping on a thread.
now = time.monotonic()
app._endpointer.reset(now - 5.0)
app._endpointer.update(0.5, now - 3.0)
closed = app._endpointer.update(0.0, now)
check("endpointer says close after the silence window", closed)
app._finish_wake_utterance("closed after silence")
check("recording stopped", not app.recorder.recording)
check("clip was queued", len(queued) == 1, f"({len(queued)})")

# In silence mode the hotkey must still work as ordinary push-to-talk.
app._on_press()
check("hotkey still starts push-to-talk when idle", app.recorder.recording)
for _ in range(20):
    app.recorder._callback(speech(), 1024, None, _Status())
app._on_release()
check("and its release is processed normally", len(queued) == 2, f"({len(queued)})")

# An empty hold is still dropped rather than sent to the GPU.
app._on_press()
app._on_release()
check("an empty push-to-talk hold is still ignored", len(queued) == 2, f"({len(queued)})")

# --- pre-roll ----------------------------------------------------------------

print("\npre-roll")

app, queued = build()
app.recorder._stream = object()
# Half a second of audio while idle, which must survive into the recording.
idle_blocks = int(0.5 * SAMPLE_RATE / 1024)
for _ in range(idle_blocks):
    app.recorder._callback(speech(amplitude=0.4), 1024, None, _Status())
check("idle audio is buffered", len(app.recorder._preroll) == idle_blocks,
      f"({len(app.recorder._preroll)} blocks)")

app._on_wake()
for _ in range(10):
    app.recorder._callback(speech(), 1024, None, _Status())
app._finish_wake_utterance("")
clip = queued[0]
expected = (idle_blocks + 10) * 1024
check("wake recording opens with the pre-roll already in it",
      len(clip) == expected, f"({len(clip)} samples, expected {expected})")

# Push-to-talk must NOT reach backwards: the key goes down before you speak.
app2, queued2 = build()
app2.recorder._stream = object()
for _ in range(idle_blocks):
    app2.recorder._callback(speech(), 1024, None, _Status())
app2._on_press()
for _ in range(10):
    app2.recorder._callback(speech(), 1024, None, _Status())
app2._on_release()
check("push-to-talk takes no pre-roll", len(queued2[0]) == 10 * 1024,
      f"({len(queued2[0])} samples, expected {10 * 1024})")

# --- re-entrancy -------------------------------------------------------------

print("\nre-entrancy")

app, queued = build()
app.recorder._stream = object()
app._on_wake()
started = app.recorder._n_samples
app._on_wake()          # detector fires again mid-utterance
check("a second detection cannot restart the recording",
      app.recorder._n_samples == started and app.recorder.recording)
check("still exactly one utterance in flight", len(queued) == 0)

# --- naming a phrase ---------------------------------------------------------
#
# The ready line and the demo both label the active phrase. Reading REGISTRY
# directly worked for the three pretrained names and raised KeyError for every
# model trained in training/, which is the only kind this project produces.

print("\nphrase labels")

check("a pretrained phrase keeps its label",
      wake.phrase_label("hey_jarvis") == "hey jarvis")

trained = wake.CUSTOM_DIR / "regression_test_phrase.onnx"
trained.parent.mkdir(parents=True, exist_ok=True)
trained.write_bytes(b"")   # resolve_phrase only checks that the file is there
try:
    label = wake.phrase_label("regression_test_phrase")
    check("a locally trained model is named from its filename",
          label == "regression test phrase", f"(got {label!r})")
finally:
    trained.unlink()

try:
    wake.phrase_label("no_such_phrase_anywhere")
    check("an unknown phrase still raises", False)
except wake.WakeError:
    check("an unknown phrase still raises", True)

print(f"\n{'wake state machine is sound' if not failures else f'{failures} check(s) failed'}")
raise SystemExit(1 if failures else 0)
