"""Settings, loaded from settings.json next to the project root.

Anything a user would plausibly want to change lives here rather than in code.
Missing keys fall back to the defaults below, so a partial settings.json is
always valid and a new release never breaks an existing file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "settings.json"

DEFAULTS: dict[str, Any] = {
    # --- hotkey ---
    # Virtual-key code to hold while speaking. 163 = VK_RCONTROL (Right Ctrl).
    # Common alternatives: 162 = Left Ctrl, 165 = Right Alt, 145 = Scroll Lock.
    "hotkey_vk": 163,
    # Swallow the key so the focused app never sees it. True is what you want
    # for a dedicated push-to-talk key; set False if you still use Right Ctrl
    # for normal shortcuts.
    "hotkey_suppress": True,
    # --- audio ---
    # None means the Windows default input device. Set to a device name
    # substring (for example "Brio") or an integer index to pin one.
    "input_device": None,
    # Longest single utterance. Whisper itself handles very long audio fine
    # (measured: 190s transcribed complete, no loss), so this is only a memory
    # guard, not a quality one. 600s is about 38MB of buffer.
    "max_seconds": 600,
    "min_seconds": 0.3,
    # Microphone RMS that fills the overlay meter to full height. Lower it if
    # the bars barely move when you speak, raise it if they sit pegged at the
    # top. This changes the display only; it has no effect on transcription.
    "mic_level_ceiling": 0.14,
    # --- model ---
    "model": "large-v3-turbo",
    "language": "en",
    "beam_size": 5,
    # --- output ---
    # "auto" picks clipboard paste for long text and simulated typing for short
    # text. Force with "paste" or "type".
    "inject_method": "auto",
    "paste_threshold_chars": 120,
    "transcript_dir": str(Path.home() / "Documents" / "LocalFlow" / "transcripts"),
    "save_transcripts": True,
    # --- overlay ---
    # Floating waveform capsule. Drag it anywhere, including another monitor.
    "overlay": True,
    # Pixels above the bottom of the screen, used only until you drag it.
    "overlay_bottom_margin": 130,
    # Saved automatically when you drag it. null means "use the default spot".
    "overlay_x": None,
    "overlay_y": None,
    # How solid the capsule looks, 0.0 to 1.0. Windows layered windows apply
    # one alpha to the whole window, so this dims the bars as well as the
    # background; the bar colours are bright enough to stay legible. Lower
    # values look more like smoked glass. Clamped to a visible range.
    "overlay_opacity": 0.78,
}


class Settings:
    def __init__(self, data: dict[str, Any] | None = None):
        self._data = {**DEFAULTS, **(data or {})}

    def __getattr__(self, name: str) -> Any:
        # __getattr__ runs for anything not found normally, including _data
        # itself before __init__ has set it (during copy or unpickling). Going
        # through self._data in that state would recurse until the stack blew,
        # hiding the real problem, so bail out explicitly first.
        if name == "_data":
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(f"Unknown setting {name!r}") from exc

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        path = path or SETTINGS_PATH
        if not path.exists():
            return cls()
        try:
            return cls(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            # A broken settings file should degrade to defaults, never crash
            # the app on startup.
            print(f"[config] could not read {path.name} ({exc}); using defaults")
            return cls()

    def set(self, **values: Any) -> None:
        self._data.update(values)

    def save(self, path: Path | None = None) -> None:
        path = path or SETTINGS_PATH
        path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")


def write_default_settings(path: Path | None = None) -> Path:
    """Create settings.json, or add keys a newer version introduced.

    Without the upgrade step, a settings.json written by an older version
    silently lacks any new key. The app still works, because load() merges
    DEFAULTS, but the key is invisible in the file so there is no way to
    discover or change it. Existing values are never overwritten.
    """
    path = path or SETTINGS_PATH
    if not path.exists():
        Settings().save(path)
        return path

    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return path  # load() already warned and fell back to defaults

    missing = {k: v for k, v in DEFAULTS.items() if k not in current}
    if missing:
        current.update(missing)
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"[config] added {len(missing)} new setting(s): {', '.join(sorted(missing))}")
    return path
