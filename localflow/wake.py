"""Wake word detection, so dictation can start without touching the keyboard.

openWakeWord on the ONNX runtime. It was chosen over Porcupine, which scores
better on false accepts and costs five fewer packages, because Porcupine needs
an account, an access key, and a periodic online licence check. Validating a
licence over the network in order to use an offline dictation tool contradicts
the one claim this project is built on, so the accuracy was not worth it.

Whisper cannot do this job. It is far too heavy to run continuously and would
hold the GPU busy for nothing, which is why a second, much smaller always-on
model exists at all. This one runs on the CPU in a few milliseconds per frame.

Three constraints shape the code below:

  * The detector never runs on the PortAudio callback. audio.py explains why:
    anything holding the GIL in that callback makes PortAudio drop incoming
    samples, which punches holes in the recording. Blocks are handed to this
    module through a bounded queue and processed on its own thread.

  * The models are hash-pinned. openWakeWord downloads them from GitHub release
    assets at runtime with no verification at all. models.py already pins the
    Whisper weights to an exact commit for the same reason, so anything that
    executes or infers here gets the same treatment. A mismatch is fatal.

  * tflite is not an option. openWakeWord defaults to `inference_framework=
    "tflite"`, and tflite-runtime publishes no wheel for CPython 3.14 on
    Windows, so every path here forces ONNX.
"""

from __future__ import annotations

import contextlib
import hashlib
import queue
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT

# Release the pinned assets come from. openWakeWord's own downloader tracks
# whatever this tag currently points at; we name it explicitly so a re-tagged
# release cannot change the bytes underneath us.
_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"

# Models live outside site-packages, so rebuilding the venv does not re-download
# them. `models/` is already in .gitignore.
MODEL_DIR = PROJECT_ROOT / "models" / "wake"

# Models trained here rather than downloaded. Kept apart from the upstream
# assets so it is always obvious which files carry a pinned digest and which
# ones we produced and are responsible for ourselves.
CUSTOM_DIR = MODEL_DIR / "custom"


@dataclass(frozen=True)
class WakeSpec:
    filename: str
    sha256: str
    label: str

    @property
    def url(self) -> str:
        return f"{_RELEASE}/{self.filename}"

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.filename


# The two feature extractors every wake word model sits on top of. They are not
# wake words themselves and are always required.
FEATURE_MODELS: dict[str, WakeSpec] = {
    "melspectrogram": WakeSpec(
        filename="melspectrogram.onnx",
        sha256="ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
        label="mel spectrogram front end",
    ),
    "embedding": WakeSpec(
        filename="embedding_model.onnx",
        sha256="70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
        label="speech embedding model",
    ),
}

# Only the phrases openWakeWord ships pretrained models for, which is why every
# entry here carries a digest: these are release assets that could change under
# us. Any other phrase is a model somebody trained themselves with training/,
# and resolve_phrase below looks those up in CUSTOM_DIR instead.
REGISTRY: dict[str, WakeSpec] = {
    "hey_jarvis": WakeSpec(
        filename="hey_jarvis_v0.1.onnx",
        sha256="94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
        label="hey jarvis",
    ),
    "alexa": WakeSpec(
        filename="alexa_v0.1.onnx",
        sha256="6ff566a01d12670e8d9e3c59da32651db1575d17272a601b7f8a39283dfbae3e",
        label="alexa",
    ),
    "hey_mycroft": WakeSpec(
        filename="hey_mycroft_v0.1.onnx",
        sha256="c2a311e8fa1338de89c31b3b46dc4dffd4af2f9a8d6ddead48893c2d301b1f18",
        label="hey mycroft",
    ),
}

DEFAULT_PHRASE = "hey_jarvis"

# openWakeWord works in 80ms frames. Feeding it exact multiples keeps the
# internal accumulator empty and avoids adding up to 80ms of detection delay.
FRAME_SAMPLES = 1280

# Bounded on purpose. If the detector thread ever falls behind, the correct
# thing is to drop frames it has not reached yet: a missed wake word costs the
# user one repeat, while letting this queue grow without limit costs memory,
# and blocking the producer would cost recorded audio.
QUEUE_FRAMES = 32


class WakeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_phrase(phrase: str) -> WakeSpec | Path:
    """Turn a wake_phrase setting into either a pinned spec or a local file.

    A locally trained model is a path to an .onnx we produced ourselves, so
    there is nothing to download and no upstream digest to pin it against. The
    pins exist to detect a release asset changing under us, which cannot happen
    to a file that never came from a release.
    """
    if phrase in REGISTRY:
        return REGISTRY[phrase]

    candidate = Path(phrase)
    if not candidate.is_absolute():
        candidate = CUSTOM_DIR / phrase
    if candidate.suffix != ".onnx":
        candidate = candidate.with_suffix(".onnx")
    if candidate.is_file():
        return candidate

    known = ", ".join(sorted(REGISTRY))
    raise WakeError(
        f"Unknown wake phrase {phrase!r}.\n"
        f"  pretrained: {known}\n"
        f"  or the name of a trained model in {CUSTOM_DIR}"
    )


def phrase_label(phrase: str) -> str:
    """What to call this phrase on screen.

    Indexing REGISTRY directly raises KeyError for a model trained in
    training/, which is every model this project produces itself, so the
    lookup has to go through resolve_phrase.
    """
    target = resolve_phrase(phrase)
    if isinstance(target, Path):
        return target.stem.replace("_", " ")
    return target.label


def ensure_models(phrase: str = DEFAULT_PHRASE) -> dict[str, Path]:
    """Download the models this phrase needs, and verify every byte.

    Returns a mapping of role to path. Raises WakeError if a file cannot be
    fetched or does not match its pinned digest.

    A digest mismatch is fatal rather than a warning. These files are fed
    straight to an inference runtime, and "the bytes changed since we pinned
    them" is exactly the event the pin exists to catch.
    """
    target = resolve_phrase(phrase)
    if isinstance(target, Path):
        # A model we trained. The feature extractors underneath it are still
        # upstream downloads, so those are still fetched and verified.
        paths = ensure_feature_models()
        paths["wakeword"] = target
        return paths

    needed = {**FEATURE_MODELS, "wakeword": target}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _fetch_and_verify(needed.values())
    return {role: spec.path for role, spec in needed.items()}


def ensure_feature_models() -> dict[str, Path]:
    """Fetch and verify only the shared front end, for a locally trained model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _fetch_and_verify(FEATURE_MODELS.values())
    return {role: spec.path for role, spec in FEATURE_MODELS.items()}


def _fetch_and_verify(specs: Iterable[WakeSpec]) -> None:
    for spec in specs:
        if spec.path.exists():
            if _sha256(spec.path) == spec.sha256:
                continue
            # A local file that no longer matches is more likely a truncated
            # download than an attack, so replace it rather than dying. The
            # verification after the re-download is what actually guards this.
            spec.path.unlink()

        _download(spec)
        actual = _sha256(spec.path)
        if actual != spec.sha256:
            spec.path.unlink(missing_ok=True)
            raise WakeError(
                f"{spec.filename} does not match its pinned digest.\n"
                f"  expected sha256:{spec.sha256}\n"
                f"  got      sha256:{actual}\n"
                "Refusing to load it. The release asset changed, or the "
                "download was tampered with in transit."
            )


def _download(spec: WakeSpec) -> None:
    import requests  # imported here so the module stays importable offline

    try:
        response = requests.get(spec.url, timeout=60, stream=True)
        response.raise_for_status()
        with spec.path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                handle.write(chunk)
    except Exception as exc:
        spec.path.unlink(missing_ok=True)
        raise WakeError(
            f"could not download {spec.filename} from {spec.url}: {exc}\n"
            "The wake word needs a one-off download. Everything after it runs "
            "offline."
        ) from exc


class WakeListener:
    """Watches the microphone for one phrase and calls on_wake when it hears it.

    Audio arrives through feed(), which is safe to call from the PortAudio
    callback because it only converts and enqueues. All inference happens on
    the thread started by run().
    """

    def __init__(
        self,
        on_wake: Callable[[], None],
        phrase: str = DEFAULT_PHRASE,
        threshold: float = 0.5,
        patience: int = 2,
        debounce_seconds: float = 3.0,
    ):
        self.on_wake = on_wake
        self.phrase = phrase
        self.threshold = threshold
        # Consecutive 80ms frames that must clear the threshold before this
        # counts as a detection. openWakeWord's own lever for trading true
        # positives against false ones, and the cheapest defence against a
        # single noisy frame starting a recording during a meeting.
        self.patience = patience
        # Cooldown after a detection, counted here rather than passed to
        # openWakeWord. Its predict() raises if `patience` and `debounce_time`
        # are given together, and patience is the more valuable of the two
        # since it is the only lever that reduces false accepts. Owning the
        # cooldown also means it holds even when the app is slow to call
        # mute(), which is the window a second detection would slip through.
        self.debounce_seconds = debounce_seconds
        self._last_fire = 0.0
        self._above = 0

        self.detections = 0
        self.frames_seen = 0
        # Most recent score, for the live meter in bench/demo_wake.py. Tuning
        # wake_threshold by eye needs to show what the model actually returns,
        # since "it did not fire" and "it scored 0.48" call for different fixes.
        self.last_score = 0.0

        self._model = None
        self._model_name: str | None = None
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=QUEUE_FRAMES)
        self._leftover = np.empty(0, dtype=np.int16)
        self._running = threading.Event()
        self._stopping = threading.Event()
        # Set while the app is recording. Detection pauses then, so saying the
        # wake word inside your own dictation cannot start a second recording.
        self._muted = threading.Event()
        self.dropped_frames = 0

    # --- setup ------------------------------------------------------------

    def load(self) -> None:
        """Fetch, verify, and load the models. Safe to call more than once."""
        if self._model is not None:
            return

        paths = ensure_models(self.phrase)

        from openwakeword.model import Model

        self._model = Model(
            wakeword_models=[str(paths["wakeword"])],
            melspec_model_path=str(paths["melspectrogram"]),
            embedding_model_path=str(paths["embedding"]),
            # Forced. tflite-runtime has no wheel for CPython 3.14 on Windows,
            # and openWakeWord defaults to tflite, so leaving this off would
            # fail at load time on the only platform this project supports.
            inference_framework="onnx",
        )
        # openWakeWord keys its scores by the model's stem, so read the name it
        # actually assigned rather than assuming it matches our registry key.
        self._model_name = next(iter(self._model.models))

    # --- audio in (called from the PortAudio callback) --------------------

    def feed(self, block: np.ndarray) -> None:
        """Hand one block of float32 mono audio to the detector.

        Runs on the PortAudio thread, so it converts and enqueues and does
        nothing else. On a full queue the frame is dropped and counted, since
        blocking here would cost recorded samples.
        """
        if self._model is None or self._muted.is_set():
            return
        # openWakeWord expects signed 16-bit samples. Clipping first, because a
        # block above 1.0 would wrap around to negative and read as a loud
        # burst of noise to the detector.
        pcm = (np.clip(block, -1.0, 1.0) * 32767.0).astype(np.int16)
        try:
            self._queue.put_nowait(pcm)
        except queue.Full:
            self.dropped_frames += 1

    # --- detector thread ---------------------------------------------------

    def run(self) -> None:
        """Consume audio and score it. Blocks until stop() is called."""
        if self._model is None:
            self.load()
        self._running.set()
        try:
            while not self._stopping.is_set():
                try:
                    block = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if block is None:
                    break
                self._process(block)
        finally:
            self._running.clear()

    def _process(self, block: np.ndarray) -> None:
        # The mic delivers 1024-sample blocks and the detector wants 1280, so
        # carry the remainder rather than zero-padding, which would inject a
        # silent gap into the middle of the phrase being scored.
        buffer = np.concatenate((self._leftover, block))
        n_frames = len(buffer) // FRAME_SAMPLES
        self._leftover = buffer[n_frames * FRAME_SAMPLES :]
        if n_frames == 0:
            return

        assert self._model is not None and self._model_name is not None
        for i in range(n_frames):
            frame = buffer[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            # Scored one 80ms frame at a time, with no `patience` argument.
            # openWakeWord's own patience implementation returns zero for every
            # frame that has not yet met the condition, which makes the score
            # unreadable: a live meter pinned at 0.000 cannot tell "it did not
            # hear you" apart from "it scored 0.46". Since the score is the
            # only tuning signal there is, patience is counted here instead and
            # predict() is left to report what it actually saw. At 1.57ms per
            # frame against an 80ms budget, the per-frame call is free.
            scores = self._model.predict(frame)
            self.last_score = float(scores.get(self._model_name, 0.0))
            self.frames_seen += 1

            if self.last_score < self.threshold:
                self._above = 0
                continue

            # Consecutive frames over the line, so one noisy frame cannot start
            # a recording while a real phrase clears several in a row.
            self._above += 1
            if self._above < self.patience:
                continue

            now = time.monotonic()
            if now - self._last_fire < self.debounce_seconds:
                continue
            self._last_fire = now
            self._above = 0
            self.detections += 1
            self._fire()

    def _fire(self) -> None:
        # An exception here would kill the detector thread and silently end all
        # wake word support for the rest of the session, so it is contained.
        try:
            self.on_wake()
        except Exception as exc:
            print(f"\n[wake] callback error: {exc!r}")

    # --- control -----------------------------------------------------------

    def mute(self) -> None:
        """Stop detecting, and discard anything already queued.

        Called when recording starts. Draining matters: frames captured just
        before the mute would otherwise be scored afterwards and could fire a
        detection for audio the user has already finished speaking.
        """
        self._muted.set()
        self._leftover = np.empty(0, dtype=np.int16)
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def unmute(self) -> None:
        if self._model is not None:
            self._model.reset()
        self._muted.clear()

    def stop(self) -> None:
        self._stopping.set()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)

    @property
    def running(self) -> bool:
        return self._running.is_set()

    @property
    def muted(self) -> bool:
        return self._muted.is_set()
