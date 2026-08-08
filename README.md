# LocalFlow

Push-to-talk dictation that runs entirely on this machine. Hold a key, speak,
release, and the text appears wherever your cursor is. No audio, text, or
telemetry ever leaves the computer, and it works with the network off.

Built as a replacement for Wispr Flow, which is cloud-only, has no offline mode,
caps the free tier at 2,000 words a week, and offers no accent tuning or custom
vocabulary.

## Status

Working: the core dictation loop and the waveform overlay. Still to come are
accent profiles, rule-based cleanup, and a system tray menu.

## Requirements

Windows, an NVIDIA GPU, and Python 3.11 or newer. The Windows dependency is not
incidental: the hotkey is a `WH_KEYBOARD_LL` hook, text injection uses
`SendInput`, and the overlay relies on layered non-activating windows. There is
no macOS or Linux path.

## Install

```powershell
git clone https://github.com/crazyashking/localflow
cd localflow
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --require-hashes --only-binary=:all:
```

Then confirm the GPU stack before trusting anything downstream:

```powershell
.\.venv\Scripts\python.exe gate_check.py
```

That downloads the pinned model on first run, roughly 1.6 GB.

## Running it

```powershell
.\.venv\Scripts\python.exe -m localflow
```

Hold **Right Ctrl**, speak, release. Text is typed at the cursor and appended to
`Documents\LocalFlow\transcripts\YYYY-MM-DD.md`.

Close the console window to quit.

## Measured on this machine

RTX 5060 Ti (16GB), i7-14700K, Whisper large-v3-turbo in FP16:

| | |
|---|---|
| Model warm-up at startup | ~2.0s, paid once |
| Decode speed | ~22x real time (a 6s sentence decodes in ~0.25s) |
| Mean WER on the test set | 2.1% |
| VRAM | ~1.6 GB resident |

## Settings

`settings.json` is created on first run. Notable keys:

| key | meaning |
|---|---|
| `hotkey_vk` | Virtual-key code to hold. `163` = Right Ctrl, `165` = Right Alt, `145` = Scroll Lock. |
| `hotkey_suppress` | If true the key is swallowed, so the focused app never sees it. Set false if you still use Right Ctrl for shortcuts. |
| `input_device` | `null` for the Windows default, or a name substring such as `"Brio"`. |
| `inject_method` | `auto`, `paste`, or `type`. See below. |
| `max_seconds` | Longest single utterance, default 600. You are warned if you hit it. |
| `min_seconds` | Shortest utterance worth transcribing, default 0.3. Anything briefer is treated as an accidental tap of the hotkey and dropped. |
| `overlay` | Floating waveform capsule. `false` disables it. |
| `overlay_bottom_margin` | Pixels above the bottom of the screen, used until you drag it. |
| `overlay_x` / `overlay_y` | Saved automatically when you drag. `null` means default spot. |
| `overlay_opacity` | How solid the capsule looks, default 0.78. See below. |

## The overlay

A capsule with a soft vertical gradient and true semicircular ends, not the
rounded rectangle every overlay uses. It stays visible the whole time the app
is running and only goes away when the app exits.

**Drag it anywhere.** Click and drag to move it, including onto another
monitor. The position is saved to `settings.json` and restored next launch. If
a saved position points at a monitor you have since unplugged, it is clamped
back onto the visible desktop rather than stranding itself off screen.

**Colour tells you what the app is doing:**

| state | colour |
|---|---|
| idle, waiting for the hotkey | dim indigo, flat line |
| listening, hearing nothing | violet |
| picking up your voice | pink |
| decoding | amber pulse travelling along the line |

The wave's height is what follows your voice. Hue was tied to loudness in an
early version and it churned through every syllable, which is noisy and says
nothing. Speaking versus quiet is decided by a hysteresis band (rises at 0.20,
falls at 0.09, with a short hang), so a voice sitting near the threshold cannot
make the colour stutter.

### Opacity

`overlay_opacity` controls how solid it looks, default 0.78. Compare levels
side by side and pick one:

```powershell
.\.venv\Scripts\python.exe bench\demo_opacity.py
```

Worth knowing what this can and cannot do. Windows layered windows offer
either a colour key (fully transparent) or one uniform alpha for the whole
window, and Tk cannot do per-pixel alpha. So opacity dims the wave along with
the background; there is no way here to have a translucent panel behind a
fully solid wave. The rim is drawn brighter than it would need to be on a
solid capsule, so the silhouette stays defined at lower values.

Acrylic blur through `SetWindowCompositionAttribute` would give a true frosted
panel, but it is an undocumented API and is known to lag while a window is
being dragged, which is the main thing this overlay is for. Not worth the
trade.

Values are clamped to at least 0.35, so a bad setting cannot leave the capsule
invisible and impossible to find.

Preview it without dictating:

```powershell
.\.venv\Scripts\python.exe bench\demo_overlay.py
```

### Why it is built this way

Five constraints shape the overlay. Each one is explained in full where it is
enforced in `localflow/overlay.py`, so this is a summary rather than the
reasoning:

- **It can never take focus.** `WS_EX_NOACTIVATE` and `WS_EX_TOOLWINDOW` are
  applied before the window is ever shown. Dictated text goes to whichever
  window holds the foreground, so an overlay that could activate would receive
  your own transcription. It has to stay clickable to be draggable, which makes
  that one style bit the only thing preventing the bug.
- **Colour is blended through HSV.** Interpolating opposite hues in RGB passes
  through muddy grey at the midpoint.
- **Every state hue sits above 0.606**, so no transition detours through vivid
  green on its way to the amber decoding colour.
- **Per-frame change is bounded by RGB distance** rather than per component,
  because that is what the eye actually measures.
- **Canvas items are created once and reconfigured**, since rebuilding ~70
  gradient scanlines per frame holds the GIL and starves the audio callback.

The last three are checked numerically by `bench/test_overlay_colour.py`, and
the focus rule by `bench/demo_overlay.py`, which polls the foreground window's
process id throughout a run.

## How long can one utterance be?

As long as `max_seconds` (default 600, so ten minutes). Whisper itself has no
practical limit: a 190 second sample transcribes complete, all 24 marker
sentences recovered, with VAD on. See `bench/test_longform.py`.

If you do speak past the cap, the app now says so. It previously discarded the
remainder in total silence, which produced a transcription that just stopped
with nothing to explain why.

**There is no timeout and no restart cycle.** Holding the key for five minutes
is one press and one recording, not a series of short ones. Windows repeats
WM_KEYDOWN roughly every 30ms while a key is held; the hook collapses those, so
10,000 repeat events produce exactly one `on_press`. The recorder appends every
block for as long as the key is down, with no timer, no rolling window, and no
`maxlen` on the buffer. `bench/test_long_hold.py` measures both, including a
continuity check that a five minute capture contains no silent gap anywhere.

## Quitting

The overlay is frameless and non-activating by design, so it has no X and
alt-F4 cannot reach it. **Close the console window you launched it from.** That
ends the process and the overlay goes with it.

## How text gets inserted

Two strategies, because neither works everywhere:

- **type**: Unicode key events. Touches nothing global, works in terminals.
- **paste**: clipboard plus Ctrl+V. Constant time regardless of length, and the
  only reliable option in some Electron apps.

`auto` types short text and pastes long text, and it checks the clipboard first.
Borrowing the clipboard means only the text can be put back, so if it holds
something richer (an image, a file drop), `auto` types instead and your copy
survives. Ordinary text still pastes, including the OLE bookkeeping formats that
Word, Excel, Explorer and browsers attach to every copy.

Two things worth knowing. Setting `inject_method` to `"paste"` forces the paste
path and skips that check, which is the one way dictating can cost you a
non-text clipboard. And if the clipboard cannot be borrowed at all, usually
because another process is holding the lock, injection falls back to typing
rather than losing the transcription.

`bench/test_clipboard.py` covers all of it without sending a single keystroke.

## When it is not working

```powershell
.\.venv\Scripts\python.exe doctor.py
```

Checks each layer separately (settings, mic, keyboard hook, injection) and names
the failing one with a suggested fix. It asks you to speak and to press the
hotkey, so it can confirm real capture and real key detection rather than
guessing.

The most common cause of "nothing happens" is focus: the text goes to whatever
window is focused, so click into a text field first, then hold the hotkey.

## Verification

```powershell
.\.venv\Scripts\python.exe gate_check.py            # CUDA, model load, decode
.\.venv\Scripts\python.exe bench\test_pipeline.py   # WER + speed vs known text
.\.venv\Scripts\python.exe bench\test_clipboard.py  # clipboard is never damaged
.\.venv\Scripts\python.exe bench\test_hotkey.py     # hook fires on synthetic keys
.\.venv\Scripts\python.exe bench\test_inject.py     # text really lands in a field
.\.venv\Scripts\python.exe bench\test_app_wiring.py # full path minus hardware
.\.venv\Scripts\python.exe bench\test_app_smoke.py  # real app + overlay + shutdown
.\.venv\Scripts\python.exe bench\test_longform.py   # 190s speech, truncation check
.\.venv\Scripts\python.exe bench\test_long_hold.py  # 5 min hold: no timeout, no restart
.\.venv\Scripts\python.exe bench\test_overlay_colour.py  # colour + hysteresis
.\.venv\Scripts\python.exe bench\test_overlay_drag.py    # drag, clamping, monitors
.\.venv\Scripts\python.exe bench\demo_overlay.py    # overlay visuals + focus safety
```

`bench/samples/*.wav` are generated by Windows SAPI, so the expected text is known
exactly and accuracy is measurable rather than a matter of impression.

Every one of these is shaped by a bug that already got through once. The theme
is that this app fails silently, so a test that only proves code ran is worse
than no test:

- The gate transcribes **real speech** rather than silence. An earlier version
  fed it silence, VAD stripped every frame, the GPU encode path never ran, and
  the gate reported PASS while cuBLAS had never loaded at all.
- The WER scorer canonicalises number words, because Whisper writes "400" where
  the reference says "four hundred". Scoring that as an error would make the
  harness lie and corrupt every tuning decision made against it.
- `test_inject.py` injects into a real focused widget instead of stubbing it.
  Stubbing hid a `SendInput` struct-size bug that rejected every call and typed
  nothing, with no other symptom. Detail in `localflow/inject.py`.
- `test_clipboard.py` replaced a check that read `inject.py` as text and grepped
  it for a substring. That check printed PASS the whole time the clipboard guard
  was rejecting nearly every real clipboard and quietly disabling the paste
  path. It now puts real formats on a real clipboard and asserts behaviour.

## Linting

```powershell
.\.venv\Scripts\python.exe -m pip install ruff
.\.venv\Scripts\python.exe -m ruff check .
```

Configured in `pyproject.toml`. The only suppression in the codebase is `E402`,
used where import order is genuinely load-bearing: the NVIDIA DLL directories
have to be registered before anything pulls in `ctranslate2`, and the bench
scripts have to extend `sys.path` before importing `localflow`.

## Dependencies and supply-chain handling

`requirements.txt` pins all 35 packages to an exact version and SHA256. Install
only with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --require-hashes --only-binary=:all:
```

- `--only-binary=:all:` forbids source distributions, so **no package's
  `setup.py` ever executes here**. This is the main defence against install-time
  code execution, and it is safe to enforce because every dependency ships a
  prebuilt Windows wheel.
- `--require-hashes` makes pip reject any artifact whose hash differs, even if a
  maintainer account is later compromised and a release is re-uploaded.
- Everything lives in the project-local `.venv`. Nothing is installed globally or
  elevated, no PATH or registry changes are made. Deleting this folder removes
  every trace.
- The model is pinned to an exact Hugging Face commit rather than tracking
  `main`. CTranslate2's `model.bin` is its own binary tensor format rather than a
  Python pickle, so loading it does not deserialize code.

## Architecture

```
localflow/
  __main__.py  console entry point (python -m localflow)
  app.py       wiring and threading
  cuda.py      registers NVIDIA DLL dirs; must import before ctranslate2
  hotkey.py    WH_KEYBOARD_LL global hook (ctypes, no admin needed)
  audio.py     16kHz mono float32 mic capture into a ring buffer
  asr.py       warm-resident Whisper model, anti-hallucination settings
  inject.py    clipboard paste / Unicode typing at the cursor
  overlay.py   the draggable Tk capsule and its colour engine
  history.py   dated markdown transcripts
  models.py    model registry, pinned by commit
  config.py    settings.json
```

Threading: the keyboard hook owns its own thread and must never block, since
Windows silently unhooks a slow hook procedure. Callbacks only flip recording
state and hand audio to a single worker thread, which keeps utterances in order.
Tk is not thread safe, so the overlay's animation loop owns the main thread.

## License

MIT. See [LICENSE](LICENSE).
