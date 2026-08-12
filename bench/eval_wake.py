r"""Measure a wake model the way LocalFlow will actually run it.

    .\.venv\Scripts\python.exe bench\eval_wake.py hey_flow

openWakeWord's trainer prints one accuracy, one recall and one
false-positives-per-hour at a single threshold. Three things make those
figures the wrong ones to set `wake_threshold` from:

  * They score a single frame crossing the threshold. LocalFlow requires
    `wake_patience` consecutive frames and then holds off for
    `wake_debounce_seconds`, which rejects most isolated spikes.
  * The per-hour figure divides by a hardcoded 11.3 hours. The validation
    features are 10.70 hours at 80 ms per frame.
  * One threshold hides the trade-off, which is the only thing worth knowing.

So this replays the held-out clips and the validation audio through the real
rule, across thresholds, from the app's own venv and runtime.

Both halves need the training data, so run it after training. Either half is
skipped with a note if its inputs are missing.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from localflow.config import DEFAULTS  # noqa: E402
from localflow.wake import ensure_models  # noqa: E402

FRAME_SAMPLES = 1280
FRAME_SECONDS = 0.08   # one embedding frame, 1280 samples at 16 kHz
WINDOW_FRAMES = 16     # what the classifier sees at once
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

CLIP_ROOT = PROJECT_ROOT / "training" / "output"
VALIDATION_FEATURES = PROJECT_ROOT / "training" / "data" / "validation_set_features.npy"


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def clip_peaks(model, key: str, directory: Path, limit: int) -> np.ndarray:
    """Highest score the model reaches on each clip in a directory."""
    files = sorted(directory.glob("*.wav"))[:limit]
    peaks = np.empty(len(files), dtype=np.float32)
    for n, path in enumerate(files):
        model.reset()
        # Lead in with silence so the phrase finishes at the end of the
        # model's window, which is how it arrives in a live stream. Scoring
        # from a cold buffer instead would understate every clip.
        padded = np.concatenate(
            (np.zeros(FRAME_SAMPLES * 4, dtype=np.int16), _read_wav(path))
        )
        best = 0.0
        for i in range(0, len(padded) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            best = max(best, float(model.predict(padded[i:i + FRAME_SAMPLES])[key]))
        peaks[n] = best
    return peaks


def count_firings(scores: np.ndarray, threshold: float, patience: int,
                  debounce: float) -> int:
    """Apply LocalFlow's rule: N consecutive frames over, then a hold-off."""
    firings = 0
    consecutive = 0
    blocked_until = -1.0
    for i, hot in enumerate(scores >= threshold):
        if not hot:
            consecutive = 0
            continue
        consecutive += 1
        if consecutive < patience:
            continue
        now = i * FRAME_SECONDS
        if now < blocked_until:
            continue
        blocked_until = now + debounce
        consecutive = 0
        firings += 1
    return firings


def detection_table(phrase: str, paths: dict[str, Path]) -> None:
    clips = CLIP_ROOT / phrase
    if not (clips / "positive_test").is_dir():
        print(f"no held-out clips under {clips}, skipping detection rates\n")
        return

    from openwakeword.model import Model

    model = Model(
        wakeword_models=[str(paths["wakeword"])],
        melspec_model_path=str(paths["melspectrogram"]),
        embedding_model_path=str(paths["embedding"]),
        inference_framework="onnx",
    )
    key = next(iter(model.models))

    positive = clip_peaks(model, key, clips / "positive_test", 2000)
    negative = clip_peaks(model, key, clips / "negative_test", 2000)
    print(f"{len(positive)} held-out positives, {len(negative)} adversarial negatives")
    for name, scores in (("positive", positive), ("negative", negative)):
        p5, p50, p95 = np.percentile(scores, [5, 50, 95])
        print(f"  {name:9s} median={p50:.3f}  p5={p5:.3f}  p95={p95:.3f}")

    print("\n  threshold   detected   adversarial accepted")
    for t in THRESHOLDS:
        print(f"    {t:.2f}      {100 * (positive >= t).mean():6.1f}%       "
              f"{100 * (negative >= t).mean():6.1f}%")
    print()


def false_accept_table(model_path: Path, patience: int, debounce: float) -> None:
    if not VALIDATION_FEATURES.is_file():
        print(f"no {VALIDATION_FEATURES.name}, skipping false accepts per hour")
        return

    features = np.load(VALIDATION_FEATURES).astype(np.float32)
    n_windows = features.shape[0] - WINDOW_FRAMES + 1
    hours = n_windows * FRAME_SECONDS / 3600

    # The exported model fixes its batch dimension at 1, which is how the app
    # drives it, so each window is its own call.
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    run = session.run
    scores = np.empty(n_windows, dtype=np.float32)
    for i in range(n_windows):
        scores[i] = run(None, {name: features[i:i + WINDOW_FRAMES][None]})[0][0, 0]

    print(f"{hours:.2f} hours of validation audio, "
          f"patience={patience} frames, debounce={debounce}s")
    print("\n  threshold   false accepts/hour")
    for t in THRESHOLDS:
        firings = count_firings(scores, t, patience, debounce)
        print(f"    {t:.2f}         {firings / hours:8.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phrase", help="a pretrained name, or a model in models/wake/custom")
    parser.add_argument("--patience", type=int, default=DEFAULTS["wake_patience"])
    parser.add_argument("--debounce", type=float, default=DEFAULTS["wake_debounce_seconds"])
    args = parser.parse_args()

    paths = ensure_models(args.phrase)
    print(f"\n=== {args.phrase} ({paths['wakeword'].name}) ===\n")

    detection_table(args.phrase, paths)
    false_accept_table(paths["wakeword"], args.patience, args.debounce)


if __name__ == "__main__":
    main()
