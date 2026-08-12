"""Try the wake word on a live microphone, and measure how often it lies.

Two jobs in one script, because they need the same rig.

**Tuning.** Run it, say the phrase, and watch the score. A live meter is the
only way to tell "it ignored me" apart from "it scored 0.46 against a threshold
of 0.5", because those call for opposite fixes. The peak score of every attempt
is reported at the end so wake_threshold can be set from what the model
actually returns rather than from guessing.

**False accepts.** Pass --minutes and leave it running through a meeting, a
call, music, or a normal working hour without ever saying the phrase. Anything
it reports is a false accept, and the summary converts that to a rate per hour.
That number belongs in the README, because a wake word that fires on its own
during a meeting is worse than having no wake word at all.

Nothing is transcribed and nothing is typed. This only listens.

Run:  .venv\\Scripts\\python.exe bench\\demo_wake.py
      .venv\\Scripts\\python.exe bench\\demo_wake.py --minutes 60
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import audio, wake  # noqa: E402
from localflow.config import Settings  # noqa: E402

# Read the real settings.json rather than the defaults, so this demo listens to
# the same microphone the app does. Pinning input_device and then tuning the
# threshold against a different mic would produce a number that means nothing.
settings = Settings.load()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--minutes", type=float, default=0.0,
                    help="run unattended for this long and report a false-accept rate")
# Deliberately not restricted to the pretrained phrases: a model trained in
# training/ is named by file, and this is the script used to measure it.
parser.add_argument("--phrase", default=settings.wake_phrase,
                    help=f"pretrained ({', '.join(sorted(wake.REGISTRY))}) "
                         f"or the name of a model in {wake.CUSTOM_DIR}")
parser.add_argument("--threshold", type=float, default=settings.wake_threshold)
parser.add_argument("--patience", type=int, default=settings.wake_patience)
args = parser.parse_args()

hits: list[tuple[float, float]] = []   # (seconds since start, score at the hit)
started = time.perf_counter()


def on_wake() -> None:
    elapsed = time.perf_counter() - started
    hits.append((elapsed, listener.last_score))
    mins, secs = divmod(elapsed, 60)
    sys.stdout.write(f"\r  DETECTED at {int(mins):02d}:{secs:05.2f}  "
                     f"score={listener.last_score:.3f}{' ' * 24}\n")
    sys.stdout.flush()


listener = wake.WakeListener(
    on_wake=on_wake,
    phrase=args.phrase,
    threshold=args.threshold,
    patience=args.patience,
    debounce_seconds=settings.wake_debounce_seconds,
)

# A model trained in training/ carries no pinned digest, so what this prints
# about verification depends on which kind of phrase it is.
target = wake.resolve_phrase(args.phrase)
trained_here = isinstance(target, Path)
label = wake.phrase_label(args.phrase)

print(f"wake word demo: \"{label}\"")
print(f"  threshold {args.threshold}   patience {args.patience} frames "
      f"({args.patience * 80}ms above threshold)")
if trained_here:
    print(f"  model: {target}")
    print("  verifying the shared feature models ...")
    listener.load()
    # No pin to check against: this file never came from a release.
    print("  feature models verified, wake model loaded from disk")
else:
    print("  downloading and verifying models ...")
    listener.load()
    print("  models verified against their pinned digests")

recorder = audio.Recorder(device=settings.input_device, on_block=listener.feed)
recorder.open()
threading.Thread(target=listener.run, daemon=True).start()
for _ in range(50):
    if listener.running:
        break
    time.sleep(0.02)

if args.minutes:
    print(f"\n  Listening for {args.minutes:g} minutes. Do NOT say the phrase.")
    print("  Anything reported below is a false accept.\n")
else:
    print(f"\n  Say \"{label}\". Ctrl+C to stop.\n")

peak = 0.0
mic_peak = 0.0
try:
    deadline = started + args.minutes * 60 if args.minutes else None
    while deadline is None or time.perf_counter() < deadline:
        score = listener.last_score
        peak = max(peak, score)
        # Microphone level next to the score, because without it a dead
        # microphone and a model that simply did not fire look identical on
        # screen. If mic stays at 0.00 while you are talking, the problem is
        # input_device and no amount of threshold tuning will help.
        level = recorder.live_level
        mic_peak = max(mic_peak, level)
        filled = int(min(score, 1.0) * 28)
        elapsed = time.perf_counter() - started
        sys.stdout.write(
            f"\r  mic {level:.2f} |{'#' * filled}{'.' * (28 - filled)}| "
            f"score={score:.3f} peak={peak:.3f}  "
            f"{int(elapsed // 60):02d}:{elapsed % 60:04.1f}  hits={len(hits)}  "
        )
        sys.stdout.flush()
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    listener.stop()
    recorder.close()

elapsed = time.perf_counter() - started
print("\n")
print("-" * 62)
print(f"  listened        {elapsed / 60:.1f} minutes")
print(f"  frames scored   {listener.frames_seen} of 80ms")
print(f"  detections      {len(hits)}")
print(f"  highest score   {peak:.3f}")
print(f"  loudest mic     {mic_peak:.3f}")
if mic_peak < 0.05:
    # The single most useful line this script can print. A silent microphone
    # explains a zero score completely, and sends you to settings.json rather
    # than to the threshold.
    print("  -> the microphone barely registered. Check input_device in "
          "settings.json\n     and that Windows lets desktop apps use the mic.")
elif peak < 0.05:
    print("  -> the mic worked but the model never responded. Try saying it "
          "closer to the\n     mic, or check you used the right phrase.")
elif peak < args.threshold:
    print(f"  -> it heard you (peak {peak:.3f}) but never crossed "
          f"{args.threshold}.\n     Lower wake_threshold to about "
          f"{max(peak - 0.05, 0.05):.2f} in settings.json.")
if listener.dropped_frames:
    # Non-zero means the detector could not keep up with the microphone, which
    # would show up as missed wake words rather than as any visible error.
    print(f"  DROPPED FRAMES  {listener.dropped_frames} (the detector fell behind)")

if args.minutes:
    per_hour = len(hits) / (elapsed / 3600) if elapsed else 0.0
    print(f"  false accepts   {per_hour:.2f} per hour")
    if hits:
        print("  times:          " + ", ".join(f"{m:.0f}m{s:04.1f}s"
              for m, s in ((h // 60, h % 60) for h, _ in hits)))
else:
    for i, (when, score) in enumerate(hits, 1):
        print(f"  hit {i:<2}          {when:6.2f}s  score {score:.3f}")
print("-" * 62)
