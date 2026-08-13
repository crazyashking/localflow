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

    # Before any window exists, including Tk's. Windows decides how a process
    # scales the first time it draws, and a late call is ignored. Without it
    # every overlay is rendered small and bitmap stretched up, which costs the
    # glow its soft falloff on any display above 100% scaling.
    from . import glow as _glow
    awareness = _glow.enable_dpi_awareness()

    print("LocalFlow: local dictation, nothing leaves this machine")
    print("-" * 62)
    print(f"  hotkey    hold {hotkey.key_name(settings.hotkey_vk)}"
          f"{' (suppressed)' if settings.hotkey_suppress else ''}")

    # Built before the rest of the banner, because it is what resolves "auto"
    # for the device and the model. Nothing here is loaded or downloaded yet;
    # warm_up() below does that.
    try:
        app = DictationApp(settings, on_status=_print_status)
    except Exception as exc:
        print(f"\nstartup failed: {exc}")
        return 1

    device = app.transcriber.device
    print(f"  model     {app.transcriber.spec.label}, {settings.language}")
    print(f"  device    {device.summary()}   {device.reason}")
    if not device.is_gpu:
        print("            CPU decoding is far slower than a GPU. If dictation")
        print("            lags, set \"model\": \"base.en\" in settings.json.")
    if settings.wake_word:
        from . import wake
        # phrase_label rather than a REGISTRY lookup: REGISTRY holds only the
        # three pretrained phrases, so a model trained in training/ printed as
        # "(unknown)" here while working perfectly everywhere else.
        try:
            phrase = wake.phrase_label(settings.wake_phrase)
        except Exception:
            phrase = f"{settings.wake_phrase} (not found)"
        ends = {
            "both": f"{settings.wake_endpoint_silence:.1f}s silence, or tap the hotkey",
            "silence": f"{settings.wake_endpoint_silence:.1f}s silence",
            "hotkey": "tap the hotkey",
        }.get(settings.wake_end_mode, settings.wake_end_mode)
        print(f"  wake word say \"{phrase}\", ends on {ends}")

    devices = audio.list_input_devices()
    chosen = settings.input_device or "system default"
    print(f"  mic       {chosen}")
    if settings.input_device is None and devices:
        for idx, name in devices[:4]:
            print(f"              [{idx}] {name}")
    print(f"  output    cursor + {settings.transcript_dir}")
    if settings.glow:
        lit = _glow.monitors()
        if settings.glow_monitors != "all":
            lit = lit[:1]
        sizes = ", ".join(
            f"{w}x{h} ({max(8, round(min(w, h) * settings.glow_thickness))}px)"
            for _, _, w, h in lit
        )
        print(f"  edge glow {sizes}   dpi {awareness}")
    print("-" * 62)

    try:
        app.warm_up()
    except Exception as exc:
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
