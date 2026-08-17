# LocalFlow

Dictation that runs entirely on your own machine. Hold a key or say a wake word,
speak, and the text appears wherever your cursor is. No audio, text, or
telemetry ever leaves your computer, and it works with the network off.

Built as a replacement for Wispr Flow, which is cloud-only, has no offline mode,
caps the free tier at 2,000 words a week, and offers no accent tuning or custom
vocabulary.

## What it does

- **Push-to-talk.** Hold a key, speak, release. Working out of the box.
- **Wake word.** Say "hey flow" to start dictation with no key at all, and stop
  talking to end it. The model ships with the repo. Off by default, one setting
  to turn on.
- **Runs without a GPU.** An NVIDIA card is the fast path. Without one, the app
  falls back to the CPU and a smaller model on its own.
- **Two status displays.** A small draggable capsule showing a live level meter,
  and light around the edge of the screen while an utterance is in flight.
  Either can be turned off without affecting the other.
- **Transcripts.** Every utterance is appended to a dated markdown file.
- **Train your own wake phrase.** `training/` builds a detector for a phrase
  nobody has trained yet.

## Planned

**Multiple languages.** The pinned model already understands 100 languages; the
app simply fixes `language` to `en` in `config.py`. So this is a settings and
switching problem rather than a modelling one: pick a language per utterance
without breaking push-to-talk, and decide whether to auto-detect (Whisper can,
at the cost of a slower first pass and occasional wrong guesses on short clips).
Worth knowing that large-v3-turbo trades some multilingual accuracy for speed.
The model registry already pins by commit and takes more than one entry, so
offering full large-v3 for languages where turbo is weak is a config change.

**Also queued:** accent profiles, rule-based cleanup of the raw transcript, and
a system tray menu.

## Requirements

| | minimum | why |
|---|---|---|
| OS | Windows 10 or 11, 64-bit | The hotkey is a `WH_KEYBOARD_LL` hook, text injection uses `SendInput`, and both overlays are layered non-activating windows. There is no macOS or Linux path. |
| GPU | NVIDIA with 4 GB VRAM, or none | The fast path. Measured footprint is 2.2 GB, so a 4 GB card leaves room for your desktop. Without one the app falls back to the CPU, which works and is much slower. |
| Driver | 527 or newer, on the GPU path | Anything that supports CUDA 12, which the pinned cuBLAS and cuDNN wheels need. |
| Python | 3.14, 64-bit | See below. This one is strict. |
| Disk | 4.5 GB on the GPU path, 1 GB on the CPU path | Packages plus the speech model. The CPU install skips 2 GB of CUDA libraries and uses a smaller model. |
| RAM | 8 GB | The app holds about 700 MB once the model is warm. |
| Mic | any | Whatever Windows already uses, or name a specific one in settings. |

**The GPU is optional, and it is what makes this feel instant.** With an NVIDIA
card, `device` resolves to CUDA and a 6-second sentence decodes in a quarter of a
second. Without one, it resolves to the CPU and the same sentence takes about two
seconds on a fast desktop chip. Both are usable for dictation. Only one of them
disappears while you are still thinking about the next sentence. The numbers for
each are in [What to expect](#what-to-expect).

**Python 3.14 specifically.** The code itself targets 3.11 and newer, but
`requirements.txt` is pinned to CPython 3.14 on Windows x64 with one SHA256 per
package. pip picks a wheel tagged for the running interpreter, so a hash-pinned
install on a different minor version fails on anything with a C extension. Use
3.14, or re-resolve the pins for your own version.

On the GPU path, 4 GB is the floor and 6 GB or more is comfortable, since the
figure above is LocalFlow alone and your desktop and browser want VRAM too.

The wake word detector always runs on the CPU and never touches the GPU, so
leaving it on all day costs no VRAM on either path.

## Install

```powershell
git clone https://github.com/crazyashking/localflow
cd localflow
py -3.14 -m venv .venv
```

Then install for the hardware you have. **With an NVIDIA GPU:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --require-hashes --only-binary=:all:
```

**Without one:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt --require-hashes --only-binary=:all:
```

`requirements-cpu.txt` is the same 39 pins to the same hashes, minus the three
`nvidia-*` wheels, which are about 2 GB of CUDA libraries a machine with no
NVIDIA card will never load. CTranslate2 ships one wheel that does both devices,
so nothing else changes. Installing the GPU file on a CPU-only machine also
works; it just costs the 2 GB.

Either way, confirm the stack before trusting anything downstream:

```powershell
.\.venv\Scripts\python.exe gate_check.py
```

It reports which device your settings resolve to and runs a real decode on it.
That downloads the pinned model on first run: roughly 1.6 GB on the GPU path,
460 MB on the CPU path.

## Running it

```powershell
.\.venv\Scripts\python.exe -m localflow
```

Click into whatever you want to dictate into, then hold **Right Ctrl**, speak,
and release. The text is typed at your cursor and appended to
`Documents\LocalFlow\transcripts\YYYY-MM-DD.md`.

Close the console window to quit.

Push-to-talk is all you get on a first run. To start by voice instead, set
`"wake_word": true` in `settings.json`. That is the only edit needed: the phrase
defaults to **"hey flow"**, whose model is trained by this project and ships in
the repo, so there is nothing to download and nothing to train.

```json
"wake_word": true,
```

The [Wake word](#wake-word) section covers choosing a phrase and setting a
threshold that does not fire while you are on a call.

### Starting it with Windows

Set `"start_on_login": true` and launch once. From then on LocalFlow comes up
minimised whenever you log in, ready for the hotkey or the wake word.

```json
"start_on_login": true,
```

It works by keeping a shortcut in your Startup folder
(`shell:startup` in the Run box), which needs no admin rights and stays visible
somewhere you can delete it by hand. The app reconciles that folder with the
setting on every launch, so setting the key back to `false` and running once
removes the shortcut, and moving the project or rebuilding the venv repairs a
shortcut that would otherwise point at an interpreter that is no longer there.

Two copies running at once means two processes fighting over the same
microphone and the same hotkey, so if it is already started for you, do not
start it again from a terminal.

## What to expect

Numbers from the development machine, an RTX 5060 Ti and an i7-14700K. Yours
will differ, and the shape is what matters: on the GPU, decoding finishes far
faster than you can talk, so the wait after you stop speaking is a fraction of a
second.

| | |
|---|---|
| Model warm-up at startup | about 2s, paid once |
| Decode speed | about 22x real time, so a 6s sentence takes 0.25s |
| Mean word error rate | 2.1% on the bundled test set |
| VRAM while running | 2.2 GB, of which 1.6 GB is the model |
| RAM while running | about 700 MB |

Reproduce the accuracy and speed figures yourself with
`bench\test_pipeline.py`, which decodes known sentences and reports the error
rate against them.

### On the CPU

Same machine with the GPU taken out of the picture, decoding on the i7-14700K
alone. Read these as an optimistic case: that is a 20-core desktop chip, and a
laptop will be slower.

| model | short sentence | 190s of continuous speech |
|---|---|---|
| `small.en`, the CPU default | 3.4x real time, so a 6s sentence takes 1.8s | 1.1x real time |
| `base.en` | 9.7x real time, so a 6s sentence takes 0.6s | not measured |

Two things worth knowing before deciding the CPU path is fine. Decode speed
falls off with the length of the utterance, so a dictated paragraph costs far
more than the short-sentence figure suggests. And `small.en` produced text
identical to the GPU's on the bundled samples, so the accuracy loss on the CPU
path comes from choosing a smaller model, not from the CPU itself.

If the wait after you stop speaking is too long, set `"model": "base.en"`. It is
roughly three times faster and noticeably less accurate on unusual words.

## Settings

`settings.json` is created on first run. Notable keys:

| key | meaning |
|---|---|
| `hotkey_vk` | Virtual-key code to hold. `163` = Right Ctrl, `165` = Right Alt, `145` = Scroll Lock. |
| `hotkey_suppress` | If true the key is swallowed, so the focused app never sees it. Set false if you still use Right Ctrl for shortcuts. |
| `start_on_login` | Start LocalFlow when you log in, `false` by default. See below. |
| `input_device` | `null` for the Windows default, or a name substring such as `"Brio"`. |
| `device` | `auto` (default) takes the GPU when one is usable and the CPU when it is not. `cuda` refuses to start rather than fall back, which is what you want if a silent drop to CPU speed would go unnoticed. `cpu` forces the CPU even on a machine with a GPU. |
| `model` | Whisper model. `auto` (default) picks `large-v3-turbo` on the GPU and `small.en` on the CPU, since large-v3-turbo on a CPU decodes at about real time. Override with `large-v3-turbo`, `small`, `small.en`, `base` or `base.en`. The `.en` builds are English-only and are both faster and more accurate on English. |
| `language` | Language code, default `en`. `null` lets Whisper detect it, which costs a little speed. |
| `inject_method` | `auto`, `paste`, or `type`. See below. |
| `save_transcripts` | Whether every utterance is also appended to a file on disk, default `true`. Set `false` to keep nothing. |
| `transcript_dir` | Where those files go, default `Documents\LocalFlow\transcripts`. |
| `max_seconds` | Longest single utterance, default 600. You are warned if you hit it. |
| `min_seconds` | Shortest utterance worth transcribing, default 0.3. Anything briefer is treated as an accidental tap of the hotkey and dropped. |
| `mic_level_ceiling` | Mic RMS that fills the overlay meter, default 0.14. Display only. Lower it if the bars barely move. |
| `overlay` | Floating waveform capsule. `false` disables it. |
| `overlay_bottom_margin` | Pixels above the bottom of the screen, used until you drag it. |
| `overlay_x` / `overlay_y` | Saved automatically when you drag. `null` means default spot. |
| `overlay_opacity` | How solid the capsule looks, default 0.78. See below. |
| `glow` | Light around the edge of the screen while an utterance is in flight. `false` disables it. See below. |
| `glow_thickness` | Band width as a fraction of the screen's shorter side, default 0.11. |
| `glow_opacity` | Brightest the band ever gets, default 0.85. |
| `glow_monitors` | `primary` or `all`. |
| `glow_sweep_from` | Where the light enters. An edge (`left`, `right`, `top`, `bottom`) or a corner (`top-left`, `top-right`, `bottom-right`, `bottom-left`), which sends it diagonally. Default `left`. |
| `glow_sweep_style` | `wash` crosses the whole screen, default. `lap` and `split` run a crest around the rim instead. See below. |
| `glow_sweep_seconds` | How long the entrance takes, default 0.7. `0` restores a plain fade in. |
| `glow_linger_seconds` | How long it holds dim after your text lands, showing it is listening again. Default 2.0, `0` to leave immediately. |
| `wake_word` | Start dictation by speaking a phrase. `false` by default. See below. |
| `wake_phrase` | `hey_flow` (default, ships with the repo), `hey_jarvis`, `alexa`, `hey_mycroft`, or the name of any other model in `models/wake/custom`. |
| `wake_threshold` | Score a phrase must reach, default 0.8, which is where "hey flow" was measured live. Raise it if the wake word fires on its own, lower it if it ignores you. The curve is tabulated below. |
| `wake_patience` | Consecutive frames above the threshold before it counts, default 2. This is what keeps single-frame spikes from firing. |
| `wake_debounce_seconds` | Hold-off after a detection, default 3.0. |
| `wake_preroll_seconds` | How far a voice-started recording reaches back, default 0.3. Raise it if your first word gets clipped. Lower it if the wake phrase itself shows up in your text. |
| `wake_end_mode` | `both`, `silence`, or `hotkey`. How a spoken utterance ends. |
| `wake_endpoint_silence` | Seconds of quiet that end an utterance in `silence` or `both`, default 2.0. |

## Edge glow

Light around the rim of the screen while an utterance is in flight. It arrives
with a wash: light enters from the left edge, crosses the whole screen with a
soft glow over everything on the way, and recedes into the rim as it reaches the
far side. Once it has settled the band thickens and brightens with your voice,
pulses slowly in amber while the model decodes, holds dim for a couple of
seconds to show it is listening again, then fades out evenly.

Three deliberate asymmetries:

- **The arrival travels and the dismissal does not.** Arriving is worth
  watching. Leaving should get out of the way.
- **The wash covers the screen and the steady state only the rim.** The
  entrance is 0.7 seconds and wants your attention. The rest of the utterance
  can be minutes long and has to stay out of your way while you read what you
  are dictating into.
- **It lingers after your text lands.** Without it the amber decoding pulse
  vanishes the instant the text appears, which reads as the app switching off
  rather than going back to waiting for you.

It runs alongside the capsule instead of replacing it. The capsule is the exact
readout and sits in one place. The glow is peripheral, so it tells you the app
is listening while you are looking at whatever you are dictating into. Turning
either off leaves the other working.

The band is a Win32 layered window with a real alpha channel per pixel, drawn
with numpy. Both were already dependencies, so it adds nothing to install and
no second process. Some detail on how it stays cheap:

- Four windows tile the rim rather than one covering the screen, which is a
  third of the pixels for the same picture. They meet without a seam because
  each measures distance to the nearest screen edge with the same function.
- Colour and alpha depend only on that distance, so a frame is one lookup table
  and one indexing pass instead of full-resolution float maths.
- The gain along the light's path is a second axis on that table rather than a
  multiply over every pixel. Scaling pixels directly costs about 10ms a frame at
  1440p, which does not fit; folding the gain into the index keeps the entrance
  at the same one indexing pass as the steady state.
- The wash is the exception that does cover the screen, so it draws into a
  buffer a quarter of the width and height and hands GDI the job of stretching
  it up. Premultiplied alpha interpolates correctly, so the upscale is free of
  artefacts and costs a sixteenth of the pixels.
- Measured cost is 4.2ms per frame on one 1440p screen and 7.0ms across two
  monitors, against a 16.7ms budget at 60fps. The wash measures 1.4ms and the
  rim styles 6.3ms. `bench\test_glow.py` fails if any of them regresses.

Two other entrances are available through `glow_sweep_style`, both of which run
a crest around the rim instead of across the screen: `lap` sends it one way
round and back to where it started, `split` sends it both ways at once to meet
on the far side. Those need the corners to flow, and the four bands are four
separate windows that know nothing about each other, so each band carries its
position along one shared clockwise path around the screen. The front also
overruns the far side by three trail lengths before the sweep ends, because
stopping it exactly at the meeting point parks the crest there and leaves a
permanent bright spot.

It runs at 60fps rather than the capsule's 40. The band is a large, slow shape
in peripheral vision, which is where frame rate shows up worst, and at 30fps the
travelling ripple visibly stepped.

The band ripples in thickness rather than in brightness, because a brightness
ripple reads as twinkling. Level drives it through an envelope follower with a
45ms attack and a 220ms release, the same shape as an audio compressor: raw RMS
moves faster than the eye reads as motion, so it strobes.

Four things keep the motion clean, each of which read as a glitch before it was
fixed:

- **Visibility and brightness are separate numbers**, multiplied at the end. As
  one value, the fade-in curve also bends the response to your voice, so the
  band looks like it is reacting while it is still arriving.
- **The fade is smoothstepped**, so it leaves and arrives at rest slowly. A
  linear fade hits full brightness at full speed and stops dead, and the eye
  catches that corner even when the fade is slow.
- **Loudness runs through a saturating curve** instead of a multiply that
  clips. With a clip, everything above about two thirds volume looks identical.
- **The ripple eases in and out** on its own time constant. Switching it on the
  speaking flag changed the shape of the whole band in a single frame the
  moment you stopped talking.

Every easing is written as a time constant in seconds and converted per frame,
so changing the frame rate does not silently retime the animation.

Preview it without recording anything:

```powershell
.\.venv\Scripts\python.exe bench\demo_glow.py
.\.venv\Scripts\python.exe bench\demo_glow.py --repeat 3        # watch the sweep again
.\.venv\Scripts\python.exe bench\demo_glow.py --sweep-from top-left
.\.venv\Scripts\python.exe bench\demo_glow.py --sweep-style lap --sweep-seconds 1.2
.\.venv\Scripts\python.exe bench\demo_glow.py --monitors all --thickness 0.16
```

It is click-through, never takes focus, and stays out of Alt+Tab. Exclusive
fullscreen games will cover it; borderless windowed is fine.

### Surviving a sleep

A layered window's device context and bitmap belong to a display configuration.
Sleeping the machine, unplugging a monitor, or changing resolution invalidates
all of it, after which `UpdateLayeredWindow` fails quietly on every call and the
glow is gone for the rest of the session with nothing logged.

Three things catch that, and each ends in a full rebuild, since none of it can
be repaired in place:

- A gap of more than 5 seconds between frames. A frame takes 16ms, so a gap that
  size means the process was frozen. This needs no window and no power API.
- The monitor layout differing from the last check, two seconds apart.
- `UpdateLayeredWindow` failing several times in a row.

Animation state survives a rebuild, so one landing mid-utterance does not replay
the sweep or lose your level. The capsule reasserts topmost and re-clamps itself
to the desktop on the same schedule, which covers being buried by a full screen
app or stranded off screen by an unplugged monitor.

The keyboard hook needs the same treatment for a different reason. Windows drops
a low-level hook across a sleep without reporting it: no message arrives, the
thread stays alive, and the hook simply stops firing, so the app looks perfectly
healthy while no keypress reaches it. On resume it is reinstalled and any key
held at the moment of sleep is forgotten, since its key-up was never delivered.

## Wake word

Off by default. Turned on, LocalFlow keeps a small detector running on the
microphone and starts recording when it hears the phrase. Push-to-talk keeps
working exactly as before; the two do not interfere.

Whisper is far too heavy to run continuously, so the detector is
[openWakeWord](https://github.com/dscripka/openWakeWord), a 215 KB model on
ONNX runtime. It never uses the GPU and never sees Whisper. **Nothing leaves
the machine.** An always-on microphone sounds like it contradicts the privacy
claim this project is built on, so to be explicit: the audio is scored locally
by a model on your disk, the rolling buffer is a second long and is overwritten
in place, and there is no network code in the detector at all.

The default phrase is **"hey flow"**, trained by this project and committed at
`models/wake/custom/hey_flow.onnx`, so it works from a fresh clone with nothing
to download. Three of openWakeWord's own phrases also work without training
anything: `hey_jarvis`, `alexa`, `hey_mycroft`. Those are downloaded on first use
and checked against a pinned SHA256.

### The false-accept problem, measured

A wake word that fires during a meeting is worse than no wake word. So both
candidate phrases were trained identically and measured on the same 10.7 hours
of validation audio, through LocalFlow's own rule (`wake_patience` consecutive
frames, then `wake_debounce_seconds` of hold-off) rather than through the
trainer's single-frame count.

| phrase | threshold | detected | false accepts/hour |
|---|---|---|---|
| `hey flow` | 0.5 | 76.8% | 0.47 |
| `hey flow` | 0.6 | 75.2% | 0.37 |
| `hey flow` | 0.7 | 73.6% | 0.28 |
| `hey flow` | 0.8 | 69.7% | 0.09 |
| `hey localflow` | 0.5 | 81.4% | 0.09 |
| `hey localflow` | 0.6 | 80.2% | 0.00 |

The curve is shallow, which is the useful part: dropping `hey flow` from 0.8 to
0.6 buys 5.5 points of detection and costs a false start every 2.7 hours
instead of every 11. Going below 0.5 buys almost nothing and costs a lot.
Lowering `wake_patience` to 1 is a worse trade at every threshold, roughly
doubling false accepts to save 80ms.

Two syllables was the problem. "hey flow" sits close to "hello", "hey Joe" and
"cash flow", and no threshold fixes that: at a matched false-accept rate of
0.09/hour it detects 69.7% where "hey localflow" detects 81.4%. Four syllables
is simply further from ordinary English. Both models were trained on 30,000
synthetic positives and 30,000 adversarial negatives under identical settings,
so the phrase is the only thing that differs.

Reproduce the shipped phrase's row with:

```powershell
.\.venv\Scripts\python.exe bench\eval_wake.py hey_flow
```

The `hey_localflow` rows need that model trained first, since it is not
committed.

The detection figures come from held-out synthetic clips spanning many voices
and speeds, which is harsher than one person saying the phrase into their own
microphone. Treat them as a floor and measure your own room with
`bench\demo_wake.py`. Measured live on one voice and one microphone, "hey flow"
scored between 0.940 and 0.970 on every attempt, so a threshold of 0.8 caught
all of them. That is far above the synthetic figure, which is the point of
measuring your own room.

**"hey flow" is in the repo.** `models/` is gitignored as a rule, because
downloaded weights are not source, and `hey_flow.onnx` is the one exception:
215 KB, no upstream to fetch it from, and the phrase the default settings point
at. So one edit turns the feature on:

```json
"wake_word": true,
```

`hey_localflow` is not committed. It won the measurement above and lost the
decision: two syllables is easier to say a hundred times a day than four, the
gap closes at a threshold you can actually pick, and shipping one phrase means
one default that is known to work. Its config is still at
`training/configs/hey_localflow.yml`, so anyone who prefers the false-accept
number can train it themselves.

For a phrase of your own, run the training in `training/` and drop the resulting
`.onnx` into `models/wake/custom/`. Both config files that produced the table
above are committed, so the runs are reproducible.

One number from that table is worth carrying into your own tuning whatever
phrase you pick: a hold-off of `wake_debounce_seconds` follows every detection,
so saying the phrase again inside that window does nothing. Testing at a natural
three second rhythm looks exactly like a detector missing one attempt in three.

### Training your own phrase

`training/` builds a model for a phrase nobody has trained yet, and is entirely
separate from running LocalFlow. See `training/README.md`.

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
ends the process and the overlay goes with it. Started by `start_on_login`, that
window is minimised at login instead of absent: find it on the taskbar and close
it the same way.

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
.\.venv\Scripts\python.exe gate_check.py            # device, model load, decode
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
.\.venv\Scripts\python.exe bench\test_wake_modes.py # wake state machine, all end modes
.\.venv\Scripts\python.exe bench\test_glow.py       # rim tiling, falloff, frame budget
.\.venv\Scripts\python.exe bench\test_autostart.py  # Startup shortcut, in a temp folder
.\.venv\Scripts\python.exe bench\demo_glow.py       # watch the edge glow, scripted
```

`bench/samples/*.wav` are generated by Windows SAPI, so the expected text is known
exactly and accuracy is measurable rather than a matter of impression.

Every one of these is shaped by a bug that already got through once. The theme
is that this app fails silently, so a test that only proves code ran is worse
than no test:

- The gate transcribes **real speech** rather than silence. An earlier version
  fed it silence, VAD stripped every frame, the encode path never ran, and
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

`requirements.txt` pins all 42 packages to an exact version and SHA256, and
`requirements-cpu.txt` pins 39 of the same packages to the same hashes, leaving
out only the three `nvidia-*` wheels. Install only with:

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
  cuda.py      registers NVIDIA DLL dirs; must import before ctranslate2.
               Reports rather than raises when there are none, which is how
               the CPU fallback is detected
  hotkey.py    WH_KEYBOARD_LL global hook (ctypes, no admin needed)
  audio.py     16kHz mono float32 mic capture into a ring buffer
  asr.py       warm-resident Whisper model, anti-hallucination settings
  inject.py    clipboard paste / Unicode typing at the cursor
  overlay.py   the draggable Tk capsule and its colour engine
  glow.py      screen edge glow: Win32 layered windows, pixels built in numpy
  wake.py      openWakeWord detector, model resolution and digest pinning
  endpoint.py  decides when a spoken utterance has ended
  history.py   dated markdown transcripts
  models.py    model registry pinned by commit, and the device decision
  autostart.py the Startup folder shortcut behind start_on_login
  config.py    settings.json
```

Threading: the keyboard hook owns its own thread and must never block, since
Windows silently unhooks a slow hook procedure. Callbacks only flip recording
state and hand audio to a single worker thread, which keeps utterances in order.
Tk is not thread safe, so the overlay's animation loop owns the main thread. The
edge glow draws on a thread of its own, because its windows are pure Win32 and
take no input, so it works whether or not the capsule is enabled and its frame
rate does not compete with Tk's. It reports failures instead of raising: nothing
about it can take dictation down.

## License

MIT. See [LICENSE](LICENSE).
