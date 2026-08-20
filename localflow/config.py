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
    # --- startup ---
    # Start LocalFlow when you log in, by keeping a shortcut in the Startup
    # folder. The app reconciles the folder with this setting every time it
    # runs, so turning it off here and launching once removes the shortcut, and
    # moving the project or rebuilding the venv repairs it. See autostart.py.
    "start_on_login": False,
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
    # Where Whisper runs. "auto" takes the GPU when one is usable and falls back
    # to the CPU when it is not. "cuda" refuses to start rather than fall back,
    # which is what you want if a silent drop to CPU speed would go unnoticed.
    # "cpu" forces the CPU even on a machine with a GPU.
    "device": "auto",
    # "auto" picks the checkpoint that suits the device: large-v3-turbo on the
    # GPU, small (or small.en for English) on the CPU, because large-v3-turbo on
    # a CPU decodes at around real time and is not worth waiting for. Override
    # with any key from models.REGISTRY: large-v3-turbo, small, small.en, base,
    # base.en. base is the one to try if small is still too slow.
    "model": "auto",
    "language": "en",
    "beam_size": 5,
    # --- output ---
    # "auto" picks clipboard paste for long text and simulated typing for short
    # text. Force with "paste" or "type".
    "inject_method": "auto",
    "paste_threshold_chars": 120,
    "transcript_dir": str(Path.home() / "Documents" / "LocalFlow" / "transcripts"),
    "save_transcripts": True,
    # --- wake word ---
    # Off by default. It costs a one-off model download, keeps a small model
    # scoring the microphone the whole time the app runs, and push-to-talk
    # already works, so turning it on should be a decision rather than a
    # surprise.
    "wake_word": False,
    # Either a phrase openWakeWord ships a pretrained model for (hey_jarvis,
    # alexa, hey_mycroft) or the filename of a model in models/wake/custom,
    # which is where a run of training/ puts one. hey_flow is trained by this
    # project and ships in the repo, so it needs no download and works from a
    # fresh clone; training/ is the recipe if you want your own phrase.
    "wake_phrase": "hey_flow",
    # Score a frame must reach to count. Raise it if the wake word fires on
    # its own, lower it if it ignores you.
    #
    # 0.6 is the value other voices can actually reach. This was 0.8 until
    # 2026-08-19, because 0.8 is where "hey flow" measured live: it scored 0.940
    # to 0.970 on every attempt through one voice and one microphone, mine, and
    # 0.8 caught all of them while holding false accepts to 0.09/hour on 10.7
    # hours of validation audio. Four people then installed from this repo and
    # none of them could trigger the phrase at all, and every one got past it by
    # lowering this value by hand. A threshold calibrated on a single speaker
    # does not transfer. 0.6 costs a false start every 2.7 hours instead of
    # every 11, which the README curve tabulates in full. Raise it if the wake
    # word fires on its own, and measure your own room with bench/demo_wake.py.
    "wake_threshold": 0.6,
    # Consecutive 80ms frames that must clear the threshold. This is the lever
    # that actually reduces false accepts: one noisy frame cannot start a
    # recording, a real phrase clears several in a row. Costs a little
    # sensitivity.
    "wake_patience": 2,
    # Ignore further detections for this long after one fires, so a single
    # phrase cannot trigger twice.
    "wake_debounce_seconds": 3.0,
    # How far back a wake-started recording reaches, in seconds. The detector
    # only knows the phrase happened once it has finished, so without this the
    # first word of a sentence begun immediately is lost. Too large and the
    # recording contains the wake phrase itself, which Whisper transcribes and
    # types: that is what "hey flow" appearing at the start of your text means.
    # Raise it if the first word gets clipped, lower it if the phrase shows up.
    "wake_preroll_seconds": 0.3,
    # What closes an utterance the wake word started:
    #   "both"    silence closes it, and tapping the hotkey closes it early
    #   "silence" silence only, fully hands free
    #   "hotkey"  no timer at all, tap the hotkey when you are done
    "wake_end_mode": "both",
    # Seconds of silence that end an utterance in "both" and "silence" modes.
    # See endpoint.py for why this is 2.0 rather than something snappier.
    "wake_endpoint_silence": 2.0,
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
    # --- edge glow ---
    # Light around the rim of the screen while an utterance is in flight. It
    # runs alongside the capsule rather than replacing it: the capsule is the
    # precise readout, the glow is the one you catch without looking away from
    # what you are dictating into. Costs about 4ms per frame at 60fps, and only
    # while an utterance is in flight.
    "glow": True,
    # How thick the band is, as a fraction of the screen's SHORTER side, so it
    # keeps its proportions on any display. 0.11 is 158px on a 1440p screen.
    "glow_thickness": 0.11,
    "glow_opacity": 0.85,
    # "primary" or "all". On "all" every display gets its own four windows and
    # its own band thickness, which roughly doubles the per-frame cost for two
    # screens.
    "glow_monitors": "primary",
    # The glow does not fade up all at once. Light enters at one point and
    # travels across the screen before settling into the rim. Where it enters:
    # an edge ("left", "right", "top", "bottom") or a corner ("top-left",
    # "top-right", "bottom-right", "bottom-left"), which sends it diagonally.
    "glow_sweep_from": "left",
    # How it travels.
    #   "wash"  light crosses the whole screen, glowing over everything on the
    #           way, then recedes into the rim. This is the one to keep.
    #   "lap"   a crest runs one way round the rim and back to where it started.
    #   "split" a crest goes both ways round the rim to meet on the far side.
    "glow_sweep_style": "wash",
    # How long the entrance takes. 0.7s is quick enough not to delay the first
    # word and slow enough to read as one movement. The rim styles cover more
    # distance and want closer to 1.0. 0 turns it off for a plain fade in.
    "glow_sweep_seconds": 0.7,
    # After the text lands, the glow holds dim for this long to show it is
    # listening again, then leaves. 0 makes it go the instant it is done, which
    # reads as the app switching off rather than going back to waiting.
    "glow_linger_seconds": 2.0,
}


class Settings:
    def __init__(self, data: dict[str, Any] | None = None):
        self._data = {**DEFAULTS, **(data or {})}
        # Which keys this object has changed since it was loaded. save() writes
        # only these, so a long-running app cannot revert an edit made to the
        # file while it was running. See save().
        self._dirty: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        # __getattr__ runs for anything not found normally, including the
        # internals below before __init__ has set them (during copy or
        # unpickling). Going through self._data in that state would recurse
        # until the stack blew, hiding the real problem, so bail out first.
        if name in ("_data", "_dirty"):
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
        self._dirty.update(values)

    def save(self, path: Path | None = None) -> None:
        """Write back only the keys this object changed.

        The app holds a snapshot of settings.json taken at startup, and the one
        thing that writes to the file during a session is dragging the capsule.
        Writing the whole snapshot back made that drag revert every edit made to
        the file since the app started, silently and with no way to tell it had
        happened. Re-reading here means an edit and a drag can coexist.
        """
        path = path or SETTINGS_PATH
        on_disk: dict[str, Any] = {}
        if path.exists():
            try:
                on_disk = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # A file too broken to read is one this write replaces. load()
                # has already warned about it by the time anything can save.
                on_disk = {}
        merged = {**DEFAULTS, **on_disk,
                  **{k: self._data[k] for k in self._dirty}}
        path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        self._dirty.clear()


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
