"""Verify text actually lands in a real focused text field.

Creates a real Tk entry widget, gives it focus, injects text through the same
code path the app uses, and reads back what the widget received. This is the
test that would have caught the SendInput INPUT-struct size bug, which broke
every injection while leaving no other symptom.

A small window flashes on screen for a moment. That is expected.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
import tkinter as tk
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import inject  # noqa: E402

# The non-ASCII case below would otherwise die printing its own failure detail
# on a cp1252 console. See the same note in test_clipboard.py.
sys.stdout.reconfigure(errors="replace")

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def pump(root: tk.Tk, seconds: float) -> None:
    """Run the Tk event loop for a while so injected keys are processed."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        root.update()
        time.sleep(0.01)


_user32 = ctypes.WinDLL("user32", use_last_error=True)


def _foreground_is_ours() -> bool:
    """True when the foreground window belongs to this process.

    SendInput delivers to whatever owns the foreground. Injecting before our
    window gets there sends the keys somewhere else entirely, which is why the
    first window of a process needs an explicit wait rather than a fixed sleep.
    """
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return False
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value == os.getpid()


INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001


def _claim_foreground_right() -> None:
    """Send a zero-pixel mouse move so this process may take the foreground.

    Windows only lets a process call SetForegroundWindow if it produced the
    most recent input event. Without this, the FIRST window a console process
    creates can never come forward, while later ones can, because by then the
    process has sent input of its own. That asymmetry is exactly what made the
    first test case fail while the identical second case passed.

    A relative move of (0, 0) does not move the cursor.
    """
    evt = inject._INPUT(
        type=INPUT_MOUSE,
        u=inject._INPUTunion(
            mi=inject._MOUSEINPUT(
                dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=None
            )
        ),
    )
    _user32.SendInput(1, ctypes.byref(evt), ctypes.sizeof(inject._INPUT))


def wait_for_foreground(root: tk.Tk, timeout: float = 5.0) -> bool:
    end = time.perf_counter() + timeout
    claimed = False
    while time.perf_counter() < end:
        root.update()
        if _foreground_is_ours():
            return True
        if not claimed:
            _claim_foreground_right()
            claimed = True
        root.lift()
        root.focus_force()
        time.sleep(0.05)
    return False


def run_case(label: str, text: str, method: str) -> str:
    root = tk.Tk()
    root.title("localflow injection test")
    root.geometry("420x90+120+120")
    entry = tk.Entry(root, width=60)
    entry.pack(padx=10, pady=20)

    root.update()
    root.deiconify()
    root.lift()
    root.focus_force()
    entry.focus_force()
    if not wait_for_foreground(root):
        root.destroy()
        raise RuntimeError(f"[{label}] window never became foreground")

    # The window owning the foreground is not enough: the Entry must also hold
    # keyboard focus inside it, or the injected characters land nowhere. Tk
    # grants widget focus asynchronously, so wait for it rather than assuming.
    deadline = time.perf_counter() + 3.0
    while time.perf_counter() < deadline:
        entry.focus_force()
        root.update()
        try:
            if root.focus_get() is entry:
                break
        except KeyError:
            pass  # focus is on a foreign window; keep trying
        time.sleep(0.02)
    else:
        root.destroy()
        raise RuntimeError(f"[{label}] entry never took keyboard focus")

    pump(root, 0.2)

    # Inject from another thread while this one pumps the event loop. A real
    # target app is always pumping messages, so a blocking sleep here would
    # misrepresent the timing and, for paste, hide restore races.
    result: dict[str, str] = {}

    def do_inject() -> None:
        result["method"] = inject.inject(
            text, method=method, paste_threshold=10_000 if method == "type" else 0
        )

    worker = threading.Thread(target=do_inject)
    worker.start()
    while worker.is_alive():
        root.update()
        time.sleep(0.01)
    worker.join()
    pump(root, 0.4)

    try:
        return entry.get()
    finally:
        root.destroy()


def run_case_retrying(label: str, text: str, method: str, attempts: int = 4) -> str:
    """Run a case, retrying if the text clearly went nowhere.

    Focus on a shared desktop is genuinely racy: any other process can take
    the foreground mid-test, and the injected keys then land somewhere else.
    An empty result means focus was lost, not that SendInput is broken, and a
    real regression would fail every attempt rather than one. Retrying keeps
    the test honest without making it flaky.
    """
    for attempt in range(1, attempts + 1):
        got = run_case(label, text, method)
        if got:
            if attempt > 1:
                print(f"  (note: [{label}] needed {attempt} attempts, focus was taken)")
            return got
        if attempt < attempts:
            time.sleep(0.4)
    return ""


print("injection checks")
print(f"  sizeof(INPUT) = {ctypes.sizeof(inject._INPUT)} (must be 40 on x64)")

# 0. Characters outside the BMP are sent as a UTF-16 surrogate pair, one key
#    event per code unit. Asserted on the events themselves rather than through
#    a widget: Tcl/Tk's own astral-plane handling varies by version, so a
#    widget round-trip here would be testing Tk instead of type_text. Runs
#    first because it needs no window and no focus.
captured: list = []
_real_send = inject._send
inject._send = lambda events: captured.extend(events) or len(events)
try:
    inject.type_text("\U0001d11e")  # G clef, U+1D11E
finally:
    inject._send = _real_send
scans = [e.ki.wScan for e in captured]
check(
    "astral char is split into a surrogate pair",
    scans == [0xD834, 0xD834, 0xDD1E, 0xDD1E],
    f"(got {[hex(s) for s in scans]})",
)

# 1. Simulated typing, the path used for short dictation.
got = run_case_retrying("type", "hello from localflow", "type")
check("type: text arrives intact", got == "hello from localflow", f"got {got!r}")

# 2. Non-ASCII, because Whisper writes accented words and curly punctuation in
#    ordinary English output, so injection has to carry them through intact.
#    This read "cafe resume 123 naive" until it was noticed that the string
#    contains no non-ASCII characters, so it proved nothing.
tricky = "café naïve Zürich Ω 123"
got = run_case_retrying("type-unicode", tricky, "type")
check("type: non-ASCII arrives intact", got == tricky, f"got {ascii(got)}")

# 3. Clipboard paste, the path used for long dictation.
long_text = "This is a longer sentence that the app would paste rather than type."
prior = inject._clipboard_get_text()
got = run_case_retrying("paste", long_text, "paste")
check("paste: text arrives intact", got == long_text, f"got {got!r}")
check("paste: clipboard restored", inject._clipboard_get_text() == prior,
      f"(was {prior!r}, now {inject._clipboard_get_text()!r})")

print(f"\n{'injection works' if not failures else f'{failures} check(s) failed'}")
raise SystemExit(1 if failures else 0)
