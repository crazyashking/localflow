# LocalFlow

Push-to-talk dictation that runs entirely on this machine. Hold a key, speak,
release, and the text appears wherever your cursor is. No audio, text, or
telemetry ever leaves the computer, and it works with the network off.

Built as a replacement for Wispr Flow, which is cloud-only, has no offline mode,
caps the free tier at 2,000 words a week, and offers no accent tuning or custom
vocabulary.

## Status

Working: the core dictation loop and the waveform overlay.

### Planned

**Multiple languages.** The pinned model already understands 100 languages; the
app simply fixes `language` to `en` in `config.py`. So this is a settings and
switching problem rather than a modelling one: pick a language per utterance
without breaking push-to-talk, and decide whether to auto-detect (Whisper can,
at the cost of a slower first pass and occasional wrong guesses on short clips).
Worth knowing that large-v3-turbo trades some multilingual accuracy for speed.
The model registry already pins by commit and takes more than one entry, so
offering full large-v3 for languages where turbo is weak is a config change.

**Wake word.** Speaking a phrase to start dictation instead of holding a key.
This needs a small always-on detector rather than Whisper, which is far too
heavy to run continuously and would keep the GPU busy for nothing. The real
design questions are the false-accept rate (a wake word that fires during a
meeting is worse than no wake word) and whether the microphone stream staying
open contradicts the privacy claim this project is built on. It does not, since
nothing leaves the machine either way, but the README will have to say so
plainly.

**Also queued:** accent profiles, rule-based cleanup of the raw transcript, and
a system tray menu.

## Requirements

Windows and an NVIDIA GPU. The Windows dependency is not incidental: the hotkey
is a `WH_KEYBOARD_LL` hook, text injection uses `SendInput`, and the overlay
relies on layered non-activating windows. There is no macOS or Linux path.

The code targets Python 3.11 and newer, but **`requirements.txt` is pinned to
CPython 3.14 on Windows x64**. It carries one SHA256 per package, and pip picks
a wheel tagged for the running interpreter, so a hash-pinned install on a
different minor version will fail on anything with a C extension. Use 3.14, or
re-resolve the pins for your version.

## Install

```powershell
git clone https://github.com/crazyashking/localflow
cd localflow
py -3.14 -m venv .venv
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
| `mic_level_ceiling` | Mic RMS that fills the overlay meter, default 0.14. Display only. Lower it if the bars barely move. |
| `overlay` | Floating waveform capsule. `false` disables it. |
| `overlay_bottom_margin` | Pixels above the bottom of the screen, used until you drag it. |
| `overlay_x` / `overlay_y` | Saved automatically when you drag. `null` means default spot. |
| `overlay_opacity` | How solid the capsule looks, default 0.78. See below. |

## The overlay

A 260x44 capsule with true semicircular ends rather than the rounded rectangle
every overlay uses. It stays visible the whole time the app is running and only
goes away when the app exits.

**It shows a flat line until you hold the hotkey.** The line becomes 24
round-capped bars the moment recording starts, and drops back to a line when it
stops. Bars mean the microphone is live, so showing them while nothing is being
captured would claim something untrue.

The bars follow your volume and nothing else. There is no idle animation and no
ripple: an earlier version multiplied every bar by a travelling sine so a quiet
meter still looked alive, which meant the display moved when you were silent.
Each bar eases toward its target height rather than snapping, which is what
makes the motion read as fluid at 40fps.

**Drag it anywhere.** Click and drag to move it, including onto another
monitor. The position is saved to `settings.json` and restored next launch. If
a saved position points at a monitor you have since unplugged, it is clamped
back onto the visible desktop rather than stranding itself off screen.

**Colour tells you what the app is doing:**

| state | shape | colour |
|---|---|---|
| idle, waiting for the hotkey | flat line | dim indigo |
| listening, hearing nothing | bars | violet |
| picking up your voice | bars | pink |
| decoding | pulse travelling along a line | amber |

The colour goes on the whole capsule, not only the bars: the background gradient
and the rim are both pulled toward the state colour, so the thing changes hue as
you talk. Bar height is what follows your voice.

The bars also fan the hue out across the row, each one offset a little from the
state colour, so the meter is a gradient rather than a block of flat colour. The
spread is bounded by the green rule below, and the colour test checks every bar
across every transition rather than only the base hue.

### If the bars barely move

Set `mic_level_ceiling` in `settings.json`. It is the microphone RMS that fills
the meter, default `0.14`. Lower it for a quiet mic, raise it if the bars sit
pegged at the top. It affects the display only and never the transcription. The
default was 0.28 until it turned out to put ordinary speech at about a third of
the available height, which looked broken even though it was working.

Hue was tied to loudness in an early version and it churned through every
syllable, which is noisy and says nothing. Speaking versus quiet is decided by a
hysteresis band (rises at 0.20, falls at 0.09, with a short hang), so a voice
sitting near the threshold cannot make the colour stutter.

### Opacity

`overlay_opacity` controls how solid it looks, default 0.78. Compare levels
side by side and pick one:

```powershell
.\.venv\Scripts\python.exe bench\demo_opacity.py
```

Worth knowing what this can and cannot do. Windows layered windows offer
either a colour key (fully transparent) or one uniform alpha for the whole
window, and Tk cannot do per-pixel alpha. So opacity dims the bars along with
the background; there is no way here to have a translucent panel behind fully
solid bars. The rim is drawn brighter than it would need to be on a solid
capsule, so the silhouette stays defined at lower values.

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

### Picking a different shape

```powershell
.\.venv\Scripts\python.exe bench\demo_styles.py
```

Draws five candidate meters at once, all fed the same level and all using the
real colour engine, so the only difference between rows is the shape. The
current one and the filled polygon it replaced are both in there, labelled, so a
change can be judged against what it replaced rather than against memory.

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
- **Canvas items are created once and reconfigured**, since rebuilding every
  gradient scanline per frame holds the GIL and starves the audio callback. The
  gradient repaint and the bar recolour are both skipped while the colour is not
  actually moving.

The three colour rules are checked numerically by
`bench/test_overlay_colour.py`, across every pair of states, and the focus rule
by `bench/demo_overlay.py`, which polls the foreground window's process id
throughout a run.

## How long can one utterance be?

As long as `max_seconds` (default 600, so ten minutes). Whisper itself has no
practical limit: a 190 second sample transcribes complete, all 24 marker
sentences recovered, with VAD on. See `bench/test_longform.py`.

If you do speak past the cap, the app tells you. Dropping the remainder in
silence would leave you with a transcription that just stops, and nothing on
screen to explain why.

**There is no timeout and no restart cycle.** Holding the key for five minutes
is a single press and a single recording. Windows repeats
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
Borrowing the clipboard means only plain text can be put back, so anything
richer would be destroyed. The check is an allow-list, and an unrecognised
format makes it type instead:

| what you last copied | `auto` will |
|---|---|
| nothing, or plain text | paste |
| text from Word, Excel or Explorer (adds OLE bookkeeping) | paste |
| text from a browser, or anything carrying HTML or rich text | type, since that formatting cannot be restored |
| an image, or files | type |

Two more things. Setting `inject_method` to `"paste"` forces the paste path and
skips the check entirely, which is the one way dictating can cost you a non-text
clipboard. And if the clipboard cannot be borrowed at all, usually because
another process is holding the lock, injection falls back to typing rather than
losing the transcription.

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
.\.venv\Scripts\python.exe bench\demo_styles.py     # compare meter shapes by eye
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

`requirements.txt` pins all 33 packages to an exact version and SHA256. Install
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
