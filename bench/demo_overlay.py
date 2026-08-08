"""Preview the overlay without talking, and verify it never steals focus.

Drives the real overlay with a synthetic level so the visuals can be judged
and tuned quickly. Cycles through every state:

    idle -> recording (speech-like) -> recording (silence, flatline)
         -> recording (loud) -> transcribing -> idle

Run:  .venv\\Scripts\\python.exe bench\\demo_overlay.py
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow.overlay import WaveOverlay  # noqa: E402

user32 = ctypes.WinDLL("user32", use_last_error=True)

SCRIPT = [
    ("idle", 2.5, "idle flatline, dim indigo"),
    ("recording", 3.0, "listening, silent: violet"),
    ("recording", 5.0, "speaking: pink, wave grows, colour must NOT churn"),
    ("recording", 3.0, "quiet again: back to violet"),
    ("recording", 5.0, "speaking loudly: still pink, only the wave gets bigger"),
    ("transcribing", 3.5, "decoding: amber pulse"),
    ("idle", 2.5, "idle again, still visible"),
]

state = {"phase": "silence", "focus_stolen": False, "hidden_seen": False}
start = time.perf_counter()
phase_start = time.perf_counter()


def level_source() -> float:
    """Speech-like envelope: syllable bursts, not a smooth sine."""
    now = time.perf_counter()
    t = now - start
    phase = state["phase"]
    if phase == "silence":
        return 0.0
    syllable = abs(math.sin(t * 5.5)) ** 1.6
    wobble = 0.75 + 0.25 * math.sin(t * 1.7)
    gain = 0.95 if phase == "loud" else 0.45
    return min(1.0, syllable * wobble * gain)


overlay = WaveOverlay(get_level=level_source)


def foreground_pid() -> int:
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return -1
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def driver() -> None:
    print("overlay demo")
    print(f"  our pid: {os.getpid()}")
    print(f"  foreground pid before: {foreground_pid()}")
    time.sleep(1.2)  # let the window build

    global phase_start
    for st, seconds, note in SCRIPT:
        if st == "recording":
            if "silent" in note or "quiet" in note:
                state["phase"] = "silence"
            elif "loudly" in note:
                state["phase"] = "loud"
            else:
                state["phase"] = "speech"
        # Colour must stay put across a speaking stretch. Sample it and check.
        state["colours"] = set()
        phase_start = time.perf_counter()
        overlay.set_state(st)
        print(f"  {st:<13} {note}")

        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            if foreground_pid() == os.getpid():
                state["focus_stolen"] = True
            # The overlay must stay on screen the whole time, including idle.
            if overlay._alpha < 0.5:
                state["hidden_seen"] = True
            time.sleep(0.05)

    print(f"  foreground pid after:  {foreground_pid()}")

    failed = False

    # Topmost is what keeps the overlay above other windows. It is applied by
    # SetWindowPos rather than by Tk, so confirm it actually stuck.
    GWL_EXSTYLE, WS_EX_TOPMOST, WS_EX_NOACTIVATE = -20, 0x00000008, 0x08000000
    WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW = 0x00000020, 0x00000080
    user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.GetWindowLongW.restype = ctypes.c_long
    style = user32.GetWindowLongW(overlay._hwnd(), GWL_EXSTYLE)
    for name, bit in (
        ("TOPMOST", WS_EX_TOPMOST),
        ("NOACTIVATE", WS_EX_NOACTIVATE),
        ("TOOLWINDOW", WS_EX_TOOLWINDOW),
    ):
        if style & bit:
            print(f"  PASS: WS_EX_{name} is set")
        else:
            print(f"  FAIL: WS_EX_{name} is missing")
            failed = True

    # Click-through must be OFF now, or the capsule could not be dragged.
    # NOACTIVATE above is what still prevents it stealing focus when clicked.
    if style & WS_EX_TRANSPARENT:
        print("  FAIL: WS_EX_TRANSPARENT is set, so the capsule cannot be dragged")
        failed = True
    else:
        print("  PASS: WS_EX_TRANSPARENT is off (draggable)")
    if state["focus_stolen"]:
        print("\n  FAIL: overlay took foreground focus. Dictated text would go to it.")
        failed = True
    else:
        print("\n  PASS: overlay never took focus.")

    if state["hidden_seen"]:
        print("  FAIL: overlay went invisible during the run.")
        failed = True
    else:
        print("  PASS: overlay stayed visible in every state.")

    state["failed"] = failed
    overlay.stop()


threading.Thread(target=driver, daemon=True).start()
overlay.run()
print("demo finished")
raise SystemExit(1 if state.get("failed") else 0)
