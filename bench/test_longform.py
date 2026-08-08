"""Find out whether long dictation gets truncated, and where.

Uses a 190 second sample containing 24 numbered sentences, so any missing
stretch is identifiable by which numbers fail to come back rather than by
impression.

Checks three separate things that could each silently drop speech:
  1. the transcriber's VAD and thresholds
  2. the recorder's max_seconds cap
  3. whether the cap warns or truncates in silence

**What it does NOT assert, and why.** An earlier version failed unless all 24
markers came back. Whisper's GPU decode is not bit-reproducible, so the same
audio decoded four times in a row recovered 21, 20, 24 and 24 of them, and the
text differed every run. That version therefore failed roughly half the time on
code that was working perfectly, which is worse than having no test at all.

Truncation and mis-recognition have different shapes, and this test keys on the
difference. Truncation removes a contiguous run at the END, because that is what
running out of buffer does. Whisper mis-phrasing a marker ("sentence, number
five") removes scattered single numbers from the middle. So a lost tail fails,
and scattered misses are reported as a note.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_whisper.audio import decode_audio  # noqa: E402

from localflow import asr, audio  # noqa: E402
from localflow.config import DEFAULTS  # noqa: E402
from localflow.models import SAMPLE_RATE  # noqa: E402

SAMPLES = Path(__file__).parent / "samples"
EXPECTED = 24

# A phrase from the very end of the reference. Truncation loses the tail before
# anything else, so this is the single most direct check in the file.
TAIL_PHRASE = "truncates speech after a while"

# Floor on marker recovery. Well below the observed run-to-run spread, so it
# only trips when something is genuinely broken rather than when the decoder
# phrased a couple of markers differently.
MIN_MARKERS = 18


def found_numbers(text: str) -> set[int]:
    """Which 'sentence number N' markers survived."""
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20,
    }
    low = text.lower()
    found = {int(n) for n in re.findall(r"sentence number (\d+)", low)}
    for match in re.findall(r"sentence number ([a-z\- ]{3,20})", low):
        token = match.strip().replace("-", " ")
        if token in words:
            found.add(words[token])
        else:
            parts = token.split()
            if len(parts) == 2 and parts[0] == "twenty" and parts[1] in words:
                found.add(20 + words[parts[1]])
    return found


clip = decode_audio(str(SAMPLES / "longform.wav"), sampling_rate=SAMPLE_RATE)
duration = len(clip) / SAMPLE_RATE
print(f"sample: {duration:.1f}s of speech, {EXPECTED} numbered sentences\n")

failures = 0


def report(label: str, text: str, seconds: float) -> set[int]:
    got = found_numbers(text)
    missing = sorted(set(range(1, EXPECTED + 1)) - got)
    print(f"[{label}]")
    print(f"  chars={len(text)}  decode={seconds:.1f}s  sentences found={len(got)}/{EXPECTED}")
    if missing:
        print(f"  MISSING: {missing}")
    print(f"  tail: ...{text[-90:].strip()}")
    print()
    return got


transcriber = asr.Transcriber()
transcriber.load()

# 1. Current production settings.
t0 = time.perf_counter()
result = transcriber.transcribe(clip, beam_size=DEFAULTS["beam_size"])
got_current = report("current settings (vad_filter=True)", result.text, time.perf_counter() - t0)

# 2. Same audio with VAD disabled, to isolate whether VAD is dropping speech.
t0 = time.perf_counter()
segments, _ = transcriber._model.transcribe(
    clip, language="en", beam_size=DEFAULTS["beam_size"],
    vad_filter=False, condition_on_previous_text=False,
)
raw = " ".join(s.text.strip() for s in segments).strip()
got_novad = report("vad_filter=False", raw, time.perf_counter() - t0)

def lost_tail(got: set[int]) -> int:
    """How many sentences are missing in one unbroken run at the end.

    This is the signature of truncation. Scattered gaps in the middle are the
    signature of the decoder phrasing a marker differently, which it does
    non-deterministically and which is not what this test is for.
    """
    run = 0
    n = EXPECTED
    while n >= 1 and n not in got:
        run += 1
        n -= 1
    return run


tail_run = lost_tail(got_current)
scattered = sorted(set(range(1, EXPECTED + 1)) - got_current)

if tail_run:
    print(f"  -> TRUNCATED: the last {tail_run} sentence(s) never came back")
    failures += 1
if TAIL_PHRASE not in result.text.lower():
    print(f"  -> TRUNCATED: the transcript does not reach {TAIL_PHRASE!r}")
    failures += 1
if len(got_current) < MIN_MARKERS:
    print(f"  -> only {len(got_current)}/{EXPECTED} markers, below the {MIN_MARKERS} floor")
    failures += 1
if scattered and not tail_run:
    print(f"  -> {len(scattered)} marker(s) phrased differently by the decoder, "
          f"not truncated: {scattered}")
if len(got_novad) > len(got_current):
    print(f"  -> disabling VAD recovers {len(got_novad) - len(got_current)} marker(s)\n")

# 3. The recorder's own cap. Does it truncate, and does it say so?
print("[recorder max_seconds cap]")
recorder = audio.Recorder(max_seconds=DEFAULTS["max_seconds"])
print(f"  configured cap: {DEFAULTS['max_seconds']}s")
if duration > DEFAULTS["max_seconds"]:
    lost = duration - DEFAULTS["max_seconds"]
    print(f"  a {duration:.0f}s utterance would lose the last {lost:.0f}s")
else:
    print(f"  this {duration:.0f}s sample fits under the cap")

# Drive the cap directly rather than trusting the reading of the code.
recorder._recording = True
block = np.zeros(1024, dtype=np.float32).reshape(-1, 1)
blocks_needed = int((DEFAULTS["max_seconds"] * SAMPLE_RATE) / 1024) + 40
for _ in range(blocks_needed):
    recorder._callback(block, 1024, None, type("S", (), {"input_overflow": False})())
captured = recorder._n_samples / SAMPLE_RATE
recorder._recording = False
print(f"  fed {blocks_needed * 1024 / SAMPLE_RATE:.0f}s, recorder kept {captured:.0f}s")
truncated = captured < (blocks_needed * 1024 / SAMPLE_RATE) - 1
print(f"  truncates past the cap: {truncated}")
print(f"  warns when it truncates: {recorder.hit_cap}")
if not truncated:
    print("  -> FAIL: the cap did not hold, so max_seconds is not enforced")
    failures += 1
# hit_cap must be the flag's VALUE, not merely its existence. It is a property
# on Recorder, so an earlier hasattr() check here could never fail no matter
# what the recorder did.
if truncated and not recorder.hit_cap:
    print("  -> FAIL: audio is dropped silently, with nothing to tell the user")
    failures += 1

print(f"\n{'no truncation found' if not failures else f'{failures} truncation problem(s)'}")
raise SystemExit(1 if failures else 0)
