"""Does holding the hotkey for five minutes capture everything, or restart?

Answers a specific question: while the key is held continuously, does anything
in the app time out, stop, and start a new recording? Two independent places
could do that, so both are checked here.

  1. The keyboard hook. Holding a key makes Windows send a stream of repeat
     WM_KEYDOWN events, one every ~30ms. If each repeat were treated as a new
     press, a five minute hold would restart recording thousands of times.

  2. The recorder. It appends every audio block for as long as _recording is
     set. Nothing polls a clock, but a silence timeout or a ring buffer with a
     maxlen would both silently discard the earlier part of a long utterance.

The audio is fed synthetically through the real callback, so this runs in
seconds rather than in five minutes, while exercising the same code path the
microphone uses.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import hotkey  # noqa: E402
from localflow.audio import SAMPLE_RATE, Recorder  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures += 1


class _Status:
    """Stand-in for PortAudio's CallbackFlags. Nothing dropped."""

    input_overflow = False


def feed(rec: Recorder, seconds: float, blocksize: int = 1024) -> int:
    """Push `seconds` of audio through the real callback. Returns block count."""
    n_blocks = int(seconds * SAMPLE_RATE / blocksize)
    # Speech-level tone rather than silence, so the level tracking runs too.
    t = np.arange(blocksize, dtype=np.float32) / SAMPLE_RATE
    block = (0.12 * np.sin(2 * np.pi * 180.0 * t)).reshape(-1, 1)
    for _ in range(n_blocks):
        rec._callback(block, blocksize, None, _Status())
    return n_blocks


# --- 1. The keyboard hook over a long hold ---------------------------------

print("holding the hotkey (hook layer)")

presses: list[int] = []
releases: list[int] = []
VK = 0x88  # unassigned, so real typing during the run cannot interfere

listener = hotkey.HotkeyListener(
    vk=VK,
    on_press=lambda: presses.append(1),
    on_release=lambda: releases.append(1),
    suppress=False,
)


def send(vk: int, message: int) -> None:
    """Drive the hook procedure directly with a synthetic key event."""
    kb = hotkey.KBDLLHOOKSTRUCT(vkCode=vk, scanCode=0, flags=0, time=0, dwExtraInfo=None)
    listener._hook_proc(hotkey.HC_ACTION, message, ctypes.cast(
        ctypes.pointer(kb), ctypes.c_void_p
    ).value)


# Windows repeats WM_KEYDOWN about every 30ms while a key is held. Five
# minutes of that is roughly 10,000 events.
REPEATS = 10_000
send(VK, hotkey.WM_KEYDOWN)
for _ in range(REPEATS - 1):
    send(VK, hotkey.WM_KEYDOWN)
send(VK, hotkey.WM_KEYUP)

check(
    "5 min of key repeat = exactly one press",
    presses == [1],
    f"{REPEATS} down events produced {len(presses)} press callback(s)",
)
check("release fires once, on key up", releases == [1], f"got {len(releases)}")

# --- 2. The recorder over a long hold --------------------------------------

print("\ncapturing a long utterance (recorder layer)")

HOLD = 300.0  # five minutes
rec = Recorder(max_seconds=600.0)
rec.begin()
blocks = feed(rec, HOLD)
clip = rec.end()

expected = blocks * 1024
check(
    "5 min of audio is kept whole",
    clip is not None and len(clip) == expected,
    f"{len(clip) / SAMPLE_RATE:.1f}s captured of {HOLD:.0f}s fed" if clip is not None else "got None",
)
check("no restart: capped flag not set", not rec.hit_cap)
check("no samples dropped", not rec.overflowed)

# Continuity: a restart or a dropped span would leave a discontinuity. The fed
# signal is a fixed-amplitude tone, so any silent gap is visible as a run of
# near-zero samples that the source never contained.
if clip is not None:
    window = SAMPLE_RATE // 10  # 100ms
    trimmed = clip[: len(clip) // window * window].reshape(-1, window)
    quietest = float(np.abs(trimmed).max(axis=1).min())
    check(
        "no silent gap anywhere in the five minutes",
        quietest > 0.05,
        f"quietest 100ms window peaks at {quietest:.3f}",
    )

# --- 3. Where the limit actually is ----------------------------------------

print("\nthe one real limit")

capped = Recorder(max_seconds=10.0)
capped.begin()
feed(capped, 12.0)
short = capped.end()
check(
    "past max_seconds the remainder is dropped",
    short is not None and abs(len(short) / SAMPLE_RATE - 10.0) < 0.2,
    f"kept {len(short) / SAMPLE_RATE:.1f}s of 12s with a 10s cap" if short is not None else "got None",
)
check("and it is reported, not silent", capped.hit_cap)

print(f"\n{'no timeout, no restart' if not failures else f'{failures} check(s) failed'}")
raise SystemExit(1 if failures else 0)
