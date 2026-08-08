"""Whisper transcription via faster-whisper / CTranslate2.

The model is loaded once and kept warm in VRAM for the life of the process.
That is what removes the 8-10 second cold start that cloud dictation tools pay
on every launch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import cuda

cuda.init()  # must precede the ctranslate2 import chain; see cuda.py

from faster_whisper import WhisperModel  # noqa: E402

from . import models  # noqa: E402

# Whisper reliably emits these when handed silence or non-speech noise. They
# are training-set artifacts from captioned video rather than transcription.
# They are only dropped on clips shorter than ARTIFACT_MAX_SECONDS, so a
# genuine "thank you" in the middle of real dictation is never eaten.
HALLUCINATION_ARTIFACTS = {
    "thank you.",
    "thanks for watching!",
    "thanks for watching.",
    "thank you for watching.",
    "thank you for watching!",
    "you",
    "bye.",
    "please subscribe.",
    "subtitles by the amara.org community",
    ".",
    "...",
}

ARTIFACT_MAX_SECONDS = 4.0


@dataclass
class Transcription:
    text: str
    raw_text: str
    audio_seconds: float
    decode_seconds: float
    # True when raw_text held only a known silence artifact that we discarded,
    # which is a different situation from the model hearing nothing at all.
    dropped_artifact: bool = False

    @property
    def speed(self) -> float:
        """Real-time factor. 20 means 20x faster than the audio's duration."""
        return self.audio_seconds / max(self.decode_seconds, 1e-6)


class Transcriber:
    def __init__(self, model_key: str = models.DEFAULT_MODEL, language: str = "en"):
        self.language = language
        self._model_key = model_key
        self._model: WhisperModel | None = None

    # --- model management ----------------------------------------------

    def load(self) -> None:
        """Load the model into VRAM, where it stays for the life of the process."""
        if self._model is not None:
            return
        spec = models.get(self._model_key)
        self._model = WhisperModel(
            spec.repo,
            revision=spec.revision,
            device=models.DEVICE,
            compute_type=models.COMPUTE_TYPE,
        )
        self._burn_in()

    def _burn_in(self) -> None:
        """Force the first GPU encode at startup rather than mid-dictation.

        Two jobs. It pays the one-off cost of CUDA kernel compilation so the
        user's first utterance is as fast as every later one, and it surfaces a
        broken GPU stack now instead of on the first thing they say, since
        CTranslate2 does not touch cuBLAS until a real encode runs.

        A failure here is fatal on purpose. Reporting "model warm" and then
        dying on the user's first sentence is the exact silent failure the
        whole cuda module exists to prevent.

        vad_filter is off deliberately: with VAD on, this synthetic clip would
        be stripped to nothing and the encode would never run.
        """
        assert self._model is not None
        rng = np.random.default_rng(0)
        clip = rng.normal(0, 0.01, models.SAMPLE_RATE).astype(np.float32)
        try:
            segments, _ = self._model.transcribe(
                clip, language=self.language, beam_size=1, vad_filter=False
            )
            for _ in segments:  # generator is lazy; must drain to force compute
                pass
        except Exception as exc:
            self._model = None
            raise RuntimeError(
                f"The model loaded but the first GPU encode failed: {exc}\n"
                "This is almost always a CUDA library that is present at import "
                "time but not on the loader path at compute time.\n"
                "Run the environment gate for a layer-by-layer diagnosis:\n"
                "  .venv\\Scripts\\python.exe gate_check.py"
            ) from exc

    @property
    def ready(self) -> bool:
        return self._model is not None

    # --- transcription ---------------------------------------------------

    def transcribe(self, audio: np.ndarray, *, beam_size: int = 5) -> Transcription:
        if self._model is None:
            self.load()
        assert self._model is not None

        audio_seconds = len(audio) / models.SAMPLE_RATE
        t0 = time.perf_counter()

        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=beam_size,
            # Silero VAD trims leading and trailing silence, which is the
            # single biggest reduction in hallucinated output.
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            # Prevents the decoder from looping on its own previous output,
            # which is how Whisper produces repeated-phrase degeneration.
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )

        parts = [s.text.strip() for s in segments]
        parts = [p for p in parts if p]
        decode_seconds = time.perf_counter() - t0

        raw = " ".join(parts).strip()
        text, dropped = _strip_artifacts(raw, audio_seconds)

        return Transcription(
            text=text,
            raw_text=raw,
            audio_seconds=audio_seconds,
            decode_seconds=decode_seconds,
            dropped_artifact=dropped,
        )


def _strip_artifacts(text: str, audio_seconds: float) -> tuple[str, bool]:
    """Drop known silence artifacts on short clips only."""
    if not text:
        return "", False
    if audio_seconds > ARTIFACT_MAX_SECONDS:
        return text, False
    if text.strip().lower() in HALLUCINATION_ARTIFACTS:
        return "", True
    return text, False
