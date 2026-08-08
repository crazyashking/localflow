"""Prove the keyboard hook plumbing works, without a human pressing anything.

Installs the real hook on its own thread, synthesises Right Ctrl down and up
with SendInput, and checks the callbacks fire exactly once each. This isolates
"the hook is broken" from "the hook works but something else is wrong".
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import hotkey, inject  # noqa: E402

# Unassigned virtual-key codes, deliberately not Right Ctrl. A real keypress
# during the run would otherwise be indistinguishable from the synthetic ones,
# so typing while the test ran could make it fail or pass for the wrong reason.
# No physical keyboard produces these, so only our own SendInput can trigger them.
VK_TEST = 0x88
VK_OTHER = 0x89
failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures += 1


presses: list[float] = []
releases: list[float] = []

listener = hotkey.HotkeyListener(
    vk=VK_TEST,
    on_press=lambda: presses.append(time.perf_counter()),
    on_release=lambda: releases.append(time.perf_counter()),
    suppress=True,
    allow_injected=True,  # so our synthetic events are not filtered out
)

print("hotkey hook checks")

thread = threading.Thread(target=listener.run, daemon=True)
thread.start()

for _ in range(50):
    if listener.running:
        break
    time.sleep(0.02)

check("hook installed", listener.running)
if not listener.running:
    print("\nSetWindowsHookExW failed; nothing else can be checked.")
    raise SystemExit(1)

# Synthesise a hold: down, wait, up.
inject._send([inject._key_event(vk=VK_TEST)])
time.sleep(0.15)
# Key repeat: a held key sends repeated WM_KEYDOWN. on_press must fire once.
inject._send([inject._key_event(vk=VK_TEST)])
time.sleep(0.15)
inject._send([inject._key_event(vk=VK_TEST, flags=inject.KEYEVENTF_KEYUP)])
time.sleep(0.25)

check("hook received events", listener.events_seen > 0, f"({listener.events_seen} events)")
check("on_press fired exactly once", len(presses) == 1, f"({len(presses)})")
check("on_release fired exactly once", len(releases) == 1, f"({len(releases)})")
if presses and releases:
    held = releases[0] - presses[0]
    check("hold duration measured", 0.2 < held < 1.0, f"({held * 1000:.0f}ms)")

# A different key must not trigger the callbacks.
before = (len(presses), len(releases))
inject._send([inject._key_event(vk=VK_OTHER)])
inject._send([inject._key_event(vk=VK_OTHER, flags=inject.KEYEVENTF_KEYUP)])
time.sleep(0.2)
check("other keys ignored", (len(presses), len(releases)) == before)

listener.stop()
thread.join(timeout=2)
check("hook uninstalled cleanly", not thread.is_alive())

print(f"\n{'hook plumbing is fine' if not failures else f'{failures} check(s) failed'}")
raise SystemExit(1 if failures else 0)
