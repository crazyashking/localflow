"""Turn the AudioSet parquet shards into 16 kHz WAV files for augmentation.

openWakeWord's notebook points at `data/bal_train09.tar`, a tar of audio files.
That URL now returns HTTP 404: the dataset was reconverted to parquet, so the
audio arrives as bytes inside a column and `torch_audiomentations` sees no
audio files at all and raises

    EmptyPathException: There are no supported audio files found.

This script extracts the audio back out. Sixteen kHz mono, matching the rate
the rest of the pipeline works at.

One old tar shard held roughly 2,200 clips. The parquet shards hold 500 each,
so five of them is the closest equivalent to what the notebook used.
"""

import argparse
import io
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

REPO = "agkphysics/AudioSet"
TARGET_SR = 16000
DEFAULT_SHARDS = ["00", "01", "02", "03", "04"]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "training" / "data" / "background"
WAV_DIR = DATA / "wav"

# video_id comes from YouTube and can start with a dash or hold anything, so it
# is not safe to use as a filename unchanged.
UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def _to_target_sr(mono: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return mono
    g = np.gcd(sr, TARGET_SR)
    return resample_poly(mono, TARGET_SR // g, sr // g)


def extract_shard(shard: str) -> int:
    path = hf_hub_download(
        repo_id=REPO,
        filename=f"data/bal_train/{shard}.parquet",
        repo_type="dataset",
        local_dir=str(DATA),
    )
    written = 0
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(batch_size=32, columns=["video_id", "audio"]):
        for row in batch.to_pylist():
            name = UNSAFE.sub("_", row["video_id"]) or f"clip{written}"
            out = WAV_DIR / f"{shard}_{name}.wav"
            if out.exists():
                continue
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
            mono = _to_target_sr(audio.mean(axis=1), sr)
            peak = np.abs(mono).max()
            if peak == 0:
                continue  # a handful of AudioSet clips are pure silence
            sf.write(out, (mono * 32767).astype(np.int16), TARGET_SR)
            written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", default=DEFAULT_SHARDS,
                    help="parquet shard numbers, two digits each (38 exist)")
    args = ap.parse_args()

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for shard in args.shards:
        n = extract_shard(shard)
        total += n
        print(f"  shard {shard}: {n} clips")

    files = list(WAV_DIR.glob("*.wav"))
    seconds = sum(sf.info(f).duration for f in files)
    size = sum(f.stat().st_size for f in files) / 2**30
    print(f"\n{len(files)} wav files ({total} new), {seconds / 3600:.1f} hours, {size:.2f} GiB")
    print(f"point background_paths at {WAV_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
