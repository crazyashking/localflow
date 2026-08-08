"""Console entry point: python -m localflow"""

from __future__ import annotations

import sys

from . import audio, hotkey
from .app import DictationApp
from .config import Settings, write_default_settings

ICONS = {
    "loading": "...",
    "ready": "  o",
    "recording": " ((",
    "transcribing": " ~~",
    "error": "  !",
}


def _print_status(state: str, detail: str) -> None:
    icon = ICONS.get(state, "   ")
    sys.stdout.write(f"\r{icon} {state:<13} {detail:<58}")
    sys.stdout.flush()
    if state == "error":
        sys.stdout.write("\n")


def main() -> int:
    settings = Settings.load()
    write_default_settings()

    print("LocalFlow: local dictation, nothing leaves this machine")
    print("-" * 62)
    print(f"  hotkey    hold {hotkey.key_name(settings.hotkey_vk)}"
          f"{' (suppressed)' if settings.hotkey_suppress else ''}")
    print(f"  model     {settings.model} ({settings.language})")

    devices = audio.list_input_devices()
    chosen = settings.input_device or "system default"
    print(f"  mic       {chosen}")
    if settings.input_device is None and devices:
        for idx, name in devices[:4]:
            print(f"              [{idx}] {name}")
    print(f"  output    cursor + {settings.transcript_dir}")
    print("-" * 62)

    app = DictationApp(settings, on_status=_print_status)
    try:
        app.warm_up()
    except Exception as exc:  # noqa: BLE001
        print(f"\nstartup failed: {exc}")
        return 1

    print("\n\nReady. Close this window to quit.\n")
    try:
        app.start()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        print(f"\n\nstopped after {app.utterances} utterance(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
