"""Adapter so openWakeWord's trainer can drive the current sample generator.

openWakeWord's train.py does this, unconditionally:

    sys.path.insert(0, os.path.abspath(config["piper_sample_generator_path"]))
    from generate_samples import generate_samples

That import targets the layout of dscripka's fork, which keeps a top-level
`generate_samples.py`. Upstream rhasspy has since restructured into a
`piper_sample_generator` package, so the import fails against a fresh clone.

Using the fork instead is not an option here: its generate_samples.py imports
`espeak_phonemizer` and `webrtcvad`, and neither builds on CPython 3.14 for
Windows (webrtcvad ships no wheel and needs MSVC). The current package works,
and generates at roughly 65 clips/sec on a mid-range CUDA GPU.

The two signatures are otherwise the same call, so this file is three
adjustments: supply the `model` argument that openWakeWord never passes, let
`auto_reduce_batch_size` fall into **kwargs since the current generator has no
such parameter and handles batching itself, and resample the output to 16 kHz.

That last one is not cosmetic. The generator writes at the voice model's native
22050 Hz, and openWakeWord's augment_clips does no resampling at all:

    if clip_sr != sr:                       # sr defaults to 16000
        raise ValueError("Error! Clip does not have the correct sample rate!")

train.py never passes `sr`, so every clip would be rejected on the first batch.
The fork the trainer was written against emitted 16 kHz directly, which is why
the upstream notebook never had to think about it. Truncation is the second
reason: total_length is 16000 samples, so a 0.96 s clip left at 22050 Hz would
be cut to 0.73 s and lose the end of the phrase.
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from piper_sample_generator.__main__ import generate_samples as _generate_samples
from scipy.signal import resample_poly

# The voice openWakeWord's own notebook uses. Pinned by filename here because
# the trainer supplies no model argument at any of its four call sites.
DEFAULT_MODEL = Path(__file__).parent / "models" / "en_US-libritts_r-medium.pt"

TARGET_SR = 16000


def _resample_file(path: Path) -> bool:
    """Rewrite one clip at TARGET_SR. Returns False if it was already there."""
    data, sr = sf.read(path, dtype="int16")
    if sr == TARGET_SR:
        return False
    # 22050 -> 16000 is 441:320, so polyphase resampling is exact and avoids
    # the interpolation artefacts a naive stride would introduce.
    g = np.gcd(sr, TARGET_SR)
    resampled = resample_poly(data.astype(np.float32), TARGET_SR // g, sr // g)
    sf.write(path, np.clip(resampled, -32768, 32767).astype(np.int16), TARGET_SR)
    return True


def resample_dir(output_dir: str | Path, workers: int = 16) -> int:
    """Bring every clip in a directory to 16 kHz. Idempotent."""
    paths = sorted(Path(output_dir).glob("*.wav"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(_resample_file, paths))


def generate_samples(
    text: list[str] | str,
    output_dir: str | Path,
    max_samples: int | None = None,
    file_names: Iterable[str] | None = None,
    batch_size: int = 1,
    model: str | Path | None = None,
    **kwargs,
) -> None:
    """Generate synthetic speech clips. See piper_sample_generator for details.

    `auto_reduce_batch_size` arrives from openWakeWord and is swallowed by
    **kwargs on purpose: the current generator picks its own batch behaviour,
    so passing it through would raise on an unexpected keyword.
    """
    resolved = Path(model) if model else DEFAULT_MODEL
    if not resolved.is_file():
        raise FileNotFoundError(
            f"voice checkpoint not found at {resolved}.\n"
            "Download it with:\n"
            "  curl -L -o models/en_US-libritts_r-medium.pt "
            "https://github.com/rhasspy/piper-sample-generator/releases/"
            "download/v2.0.0/en_US-libritts_r-medium.pt"
        )

    kwargs.pop("auto_reduce_batch_size", None)

    _generate_samples(
        text=text,
        output_dir=str(output_dir),
        model=str(resolved),
        max_samples=max_samples,
        file_names=file_names,
        batch_size=batch_size,
        **kwargs,
    )

    # The generator writes at the voice's native rate. See the module docstring
    # for why leaving it there breaks the very next stage.
    resample_dir(output_dir)
