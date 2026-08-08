"""Layer-by-layer diagnostic. Run this when dictation is not working.

Checks each stage independently so the failing one is obvious, rather than
having to guess which part of the chain is broken.

  .venv\\Scripts\\python.exe doctor.py
"""

from __future__ import annotations

import sys
import threading
import time

problems: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def bad(msg: str, fix: str) -> None:
    print(f"  [FAIL] {msg}")
    print(f"         fix: {fix}")
    problems.append(msg)


print("=" * 70)
print("LocalFlow doctor")
print("=" * 70)

# --- 1. settings -------------------------------------------------------
print("\n[1] settings")
from localflow import audio, hotkey  # noqa: E402
from localflow.config import Settings  # noqa: E402
from localflow.models import SAMPLE_RATE  # noqa: E402

settings = Settings.load()
print(f"  hotkey        {hotkey.key_name(settings.hotkey_vk)} (vk={settings.hotkey_vk}), "
      f"suppress={settings.hotkey_suppress}")
print(f"  input_device  {settings.input_device!r}")
print(f"  inject_method {settings.inject_method}")

# --- 2. microphone -----------------------------------------------------
print("\n[2] microphone")
devices = audio.list_input_devices()
if not devices:
    bad("no input devices found at all", "check Windows Sound settings")
else:
    for idx, name in devices:
        print(f"    [{idx}] {name}")

recorder = None
try:
    recorder = audio.Recorder(device=settings.input_device)
    recorder.open()
    ok("microphone stream opened")
except Exception as exc:
    bad(f"could not open microphone: {exc}",
        "set input_device in settings.json to a name substring, e.g. \"Brio\"")

if recorder is not None:
    print("\n  Recording 4 seconds. SAY SOMETHING NOW, out loud.")
    recorder.begin()
    for _ in range(40):
        time.sleep(0.1)
        bar = int(min(recorder.peak, 1.0) * 45)
        sys.stdout.write(f"\r  level |{'#' * bar}{'.' * (45 - bar)}| peak={recorder.peak:.3f}")
        sys.stdout.flush()
    clip = recorder.end()
    print()

    if clip is None:
        bad("no audio captured", "the stream returned nothing; try a different input_device")
    else:
        stats = audio.analyse(clip)
        print(f"  captured {len(clip) / SAMPLE_RATE:.1f}s  peak={stats.peak:.4f}  "
              f"rms={stats.rms:.4f}  clipped={stats.clipped_fraction:.2%}")
        if stats.peak < 0.005:
            bad("microphone is essentially silent",
                "check the mic is not muted, is the default device, and that "
                "Windows Settings > Privacy > Microphone allows desktop apps")
        elif stats.is_clipping:
            bad(f"microphone is CLIPPING ({stats.clipped_fraction:.1%} of samples maxed out)",
                "lower Input volume in Settings > System > Sound > your mic (try 60-75), "
                "or move slightly further from the mic. Clipped audio hurts accuracy "
                "more than quiet audio does")
        elif stats.is_quiet:
            print("  [warn] very quiet. Raise the mic level in Windows Sound settings.")
            ok("some audio captured")
        else:
            ok(f"microphone is capturing audio, {stats.summary()}")

# --- 3. keyboard hook --------------------------------------------------
print("\n[3] keyboard hook")
hits = {"press": 0, "release": 0}
listener = hotkey.HotkeyListener(
    vk=settings.hotkey_vk,
    on_press=lambda: hits.__setitem__("press", hits["press"] + 1),
    on_release=lambda: hits.__setitem__("release", hits["release"] + 1),
    suppress=settings.hotkey_suppress,
)
thread = threading.Thread(target=listener.run, daemon=True)
thread.start()
for _ in range(50):
    if listener.running:
        break
    time.sleep(0.02)

if not listener.running:
    bad("keyboard hook failed to install",
        "another app may be blocking low-level hooks; try closing overlay/macro tools")
else:
    ok("hook installed")
    key = hotkey.key_name(settings.hotkey_vk)
    print(f"\n  Press and release {key} a few times over the next 6 seconds.")
    print("  IMPORTANT: keep this console window focused.")
    for _ in range(60):
        time.sleep(0.1)
        sys.stdout.write(f"\r  events={listener.events_seen}  "
                         f"press={hits['press']}  release={hits['release']}   ")
        sys.stdout.flush()
    print()
    if listener.events_seen == 0:
        bad("hook saw no keyboard events at all",
            "a security/anti-cheat tool is likely blocking the hook")
    elif hits["press"] == 0:
        bad(f"saw {listener.events_seen} key events but never {key}",
            "you may be pressing the wrong key. Change hotkey_vk in settings.json "
            "(162=Left Ctrl, 163=Right Ctrl, 165=Right Alt, 145=Scroll Lock)")
    else:
        ok(f"{key} detected ({hits['press']} press, {hits['release']} release)")
    listener.stop()

if recorder is not None:
    recorder.close()

# --- 4. injection ------------------------------------------------------
print("\n[4] text injection")
try:
    import ctypes

    from localflow import inject

    size = ctypes.sizeof(inject._INPUT)
    expected = inject._EXPECTED_INPUT_SIZE
    if size != expected:
        bad(f"INPUT struct is {size} bytes, must be {expected}",
            "inject.py struct layout regressed")
    else:
        ok("INPUT struct size correct (SendInput will be accepted)")
    ok(f"clipboard restorable right now: {inject.clipboard_is_restorable()}")
except Exception as exc:
    bad(f"injection layer error: {exc}", "see traceback")

# --- summary -----------------------------------------------------------
print("\n" + "=" * 70)
if problems:
    print(f"{len(problems)} problem(s) found:")
    for p in problems:
        print(f"  - {p}")
else:
    print("No problems found. All four layers work.")
    print("If dictation still does nothing, the text is probably going to a")
    print("different window than you expect: click into a text field FIRST,")
    print("then hold the hotkey.")
print("=" * 70)
raise SystemExit(1 if problems else 0)
