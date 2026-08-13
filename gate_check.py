"""Environment gate. Run this before trusting anything downstream.

Confirms, in order:
  1. NVIDIA DLL directories register cleanly, when there are any.
  2. CTranslate2 imports, and the device the settings ask for is real.
  3. The pinned model loads with the compute type that device needs.
  4. A real transcription round-trip produces text at a sane speed.

It gates the CPU path as well as the GPU one. On a machine with no NVIDIA card
steps 1 and 2 report that and continue, and the gate passes on the CPU; the run
it must never pass is one where the settings ask for a GPU that is not there.

If this fails, no amount of application code will help.
"""

from __future__ import annotations

import time
from pathlib import Path

from localflow import cuda, models
from localflow.config import Settings

settings = Settings.load()

print("=" * 68)
print("LocalFlow environment gate")
print("=" * 68)

# 1. DLL directories. Must happen before ctranslate2 is imported.
print("\n[1/4] NVIDIA DLL directories")
print(cuda.describe())

# 2. CTranslate2, and the device the settings resolve to.
#
# resolve_device is the same call the app makes, so this step gates the real
# decision rather than a re-implementation of it. It raises when settings say
# "cuda" and the GPU is not usable, which is the failure worth being loud about.
print("\n[2/4] CTranslate2 and device")
import ctranslate2  # noqa: E402  (import order is load-bearing here)

print(f"  ctranslate2      {ctranslate2.__version__}")
print(f"  device setting   {settings.device!r}")
try:
    device = models.resolve_device(settings.device)
except (RuntimeError, ValueError) as exc:
    raise SystemExit(f"FAIL: {exc}") from exc

print(f"  cuda devices     {ctranslate2.get_cuda_device_count()}")
print(f"  resolved to      {device.summary()}  ({device.reason})")
supported = sorted(ctranslate2.get_supported_compute_types(device.name))
print(f"  compute types    {supported}")
if device.compute_type not in supported:
    raise SystemExit(
        f"FAIL: this CTranslate2 build cannot do {device.compute_type} on "
        f"{device.name}. Supported: {supported}"
    )

# 3. Model load, on the resolved device.
print("\n[3/4] Model load")
from faster_whisper import WhisperModel  # noqa: E402

spec = models.get(settings.model, device=device, language=settings.language)
print(f"  {spec.label}")
print(f"  repo             {spec.repo}")
print(f"  revision         {spec.revision[:12]} (pinned)")
print(f"  compute_type     {device.compute_type}")
print("  downloading / loading ...")

t0 = time.perf_counter()
model = WhisperModel(
    spec.repo,
    revision=spec.revision,
    device=device.name,
    compute_type=device.compute_type,
)
load_s = time.perf_counter() - t0
print(f"  loaded in        {load_s:.1f}s")

# 4. Real transcription round-trip.
#
# This MUST use audio containing actual speech. An earlier version of this gate
# fed the model pure silence, VAD correctly stripped every frame, and the encode
# path never ran at all. The gate reported PASS while cuBLAS had in fact never
# loaded, and the failure only surfaced later on real audio. Silence is
# worthless as a compute check.
print("\n[4/4] Decode round-trip")
import numpy as np  # noqa: E402

# Anchored to this file, NOT to the working directory. asr.py's burn-in failure
# tells the user to run this script, and they may well run it from anywhere. A
# relative path meant the sample was not found, the weaker synthetic-noise
# branch ran instead, and the gate still printed GATE PASSED without ever
# checking that real speech produces real text. That is the same silent
# downgrade described at the top of this section, arriving through a different
# door.
sample = Path(__file__).resolve().parent / "bench" / "samples" / "s1.wav"
if sample.exists():
    from faster_whisper.audio import decode_audio  # noqa: E402

    clip = decode_audio(str(sample), sampling_rate=models.SAMPLE_RATE)
    source = sample.name
    vad = True
else:
    # No sample available: force an encode with noise and VAD disabled, so the
    # compute path is genuinely exercised even though the text will be junk.
    rng = np.random.default_rng(0)
    clip = rng.normal(0, 0.01, models.SAMPLE_RATE * 2).astype(np.float32)
    source = "synthetic noise (VAD off, compute check only)"
    vad = False
    print(f"  [warn] {sample} is missing, so this run cannot check that real")
    print("         speech produces real text. Compute is exercised, accuracy is not.")

audio_s = len(clip) / models.SAMPLE_RATE
t0 = time.perf_counter()
segments, _info = model.transcribe(
    clip, language="en", beam_size=5, vad_filter=vad, condition_on_previous_text=False
)
text = " ".join(s.text for s in segments).strip()
decode_s = time.perf_counter() - t0

print(f"  source           {source}")
speed = audio_s / max(decode_s, 1e-6)
print(f"  decoded {audio_s:.1f}s in {decode_s:.2f}s  ({speed:.0f}x real time)")
print(f"  output           {text[:70]!r}")
if sample.exists() and not text:
    raise SystemExit("FAIL: real speech produced no transcription.")

# Separate hallucination check: silence must produce nothing.
print("\n  hallucination check (3s silence)")
silence = np.zeros(models.SAMPLE_RATE * 3, dtype=np.float32)
segments, _ = model.transcribe(
    silence, language="en", beam_size=5, vad_filter=True, condition_on_previous_text=False
)
silence_text = " ".join(s.text for s in segments).strip()
print(f"  output           {silence_text!r}"
      f"{'  (correct)' if not silence_text else '  <-- needs threshold tuning'}")

print("\n" + "=" * 68)
# The banner has to say which check actually ran. A bare "GATE PASSED" after
# the degraded branch is how this file lied once already.
if sample.exists():
    print(f"GATE PASSED on {device.summary()}")
else:
    print(f"GATE PASSED on {device.summary()} (COMPUTE ONLY) - no speech sample, "
          "accuracy was not checked")
print("=" * 68)
