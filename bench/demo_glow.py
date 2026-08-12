"""Watch the screen edge glow without running LocalFlow, and time it.

Two jobs, same rig, for the same reason bench/demo_wake.py has two.

**Look.** Drives the glow through a scripted utterance: armed and waiting, a
voice rising and falling, then decoding, then away. Nothing records and nothing
is transcribed, so this can run while you are on a call.

**Cost.** Every frame is timed. The glow draws on a thread of its own at 60fps,
so the number that matters is whether a frame fits in 16.7ms with room to spare.
If it does not, the band stutters and the whole effect dies, and the fix is a
thinner band or fewer monitors rather than anything clever.

Run:  .venv\\Scripts\\python.exe bench\\demo_glow.py
      .venv\\Scripts\\python.exe bench\\demo_glow.py --monitors all
      .venv\\Scripts\\python.exe bench\\demo_glow.py --seconds 30 --thickness 0.16
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import glow  # noqa: E402
from localflow.config import Settings  # noqa: E402

# Read the real settings.json rather than this script's own defaults, for the
# same reason bench/demo_wake.py does: a demo that shows something the app will
# not do is worse than no demo. Every default below comes from the file the app
# reads, so what you watch here is what you get.
settings = Settings.load()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seconds", type=float, default=18.0)
parser.add_argument("--monitors", default=settings.glow_monitors,
                    choices=("primary", "all"))
parser.add_argument("--thickness", type=float, default=settings.glow_thickness,
                    help="fraction of the screen's shorter side")
parser.add_argument("--opacity", type=float, default=settings.glow_opacity)
parser.add_argument("--sweep-from", default=settings.glow_sweep_from,
                    choices=glow.SWEEP_ORIGINS,
                    help="corner or edge the light starts from")
parser.add_argument("--sweep-style", default=settings.glow_sweep_style,
                    choices=glow.SWEEP_STYLES,
                    help="wash crosses the screen, lap and split run round the rim")
parser.add_argument("--sweep-seconds", type=float,
                    default=settings.glow_sweep_seconds,
                    help="0 for a plain fade in with no sweep")
parser.add_argument("--linger-seconds", type=float,
                    default=settings.glow_linger_seconds,
                    help="how long it holds dim after the text lands")
parser.add_argument("--repeat", type=int, default=1,
                    help="run the scripted session this many times, to watch "
                         "the sweep more than once")
args = parser.parse_args()

awareness = glow.enable_dpi_awareness()
screens = glow.monitors()
print(f"dpi awareness   {awareness}")
for i, (x, y, w, h) in enumerate(screens):
    tag = "primary" if i == 0 else f"monitor {i + 1}"
    print(f"  {tag:<12} {w}x{h} at ({x}, {y})   "
          f"band {max(8, round(min(w, h) * args.thickness))}px")

# A scripted voice, so the demo is repeatable and needs no microphone. Two
# phrases with a breath between them, which is the case the release time exists
# to smooth over.
def scripted_level(t: float) -> float:
    if t < 2.0 or t > 11.0:
        return 0.0
    if 6.0 < t < 7.1:
        return 0.02
    syllables = 0.55 + 0.45 * math.sin(t * 11.0)
    swell = math.sin((t - 2.0) / 9.0 * math.pi)
    return max(0.0, min(1.0, syllables * swell * 0.95))


CYCLE = 18.0        # one whole session, so --repeat can just run it again

elapsed = 0.0
cycle_t = 0.0
frames: list[float] = []


def level() -> float:
    # Timed here rather than inside the glow: this callback is the one place the
    # render thread reaches into the rest of the app, so it is where a stall
    # would come from in the real thing.
    t0 = time.perf_counter()
    value = scripted_level(cycle_t)
    frames.append(time.perf_counter() - t0)
    return value


band = glow.BorderGlow(
    level, opacity=args.opacity, thickness_frac=args.thickness,
    which_monitors=args.monitors, sweep_from=args.sweep_from,
    sweep_style=args.sweep_style, sweep_seconds=args.sweep_seconds,
    linger_seconds=args.linger_seconds,
)
band.start()

script = [
    (0.0, "recording", "armed: the light enters from the "
                       f"{args.sweep_from}"),
    (2.0, "recording", "speaking"),
    (11.0, "transcribing", "decoding"),
    (14.5, "ready", "dismissed, fading out evenly"),
]
total = CYCLE * args.repeat if args.repeat > 1 else args.seconds
step = 0
lap = 0
started = time.perf_counter()
next_line = started
sweep = (f"{args.sweep_seconds:g}s {args.sweep_style} from the {args.sweep_from}"
         if args.sweep_seconds > 0 else "off, plain fade in")
print(f"\n  sweep {sweep}")
print(f"  {total:g}s"
      + (f", {args.repeat} sessions" if args.repeat > 1 else "")
      + ". Ctrl+C to stop early.\n")
try:
    while elapsed < total:
        elapsed = time.perf_counter() - started
        cycle_t = elapsed % CYCLE if args.repeat > 1 else elapsed
        if args.repeat > 1 and int(elapsed // CYCLE) != lap:
            lap = int(elapsed // CYCLE)
            step = 0
        if step < len(script) and cycle_t >= script[step][0]:
            _, state, label = script[step]
            band.set_state(state)
            sys.stdout.write(f"\r  {elapsed:5.1f}s  {label:<44}\n")
            sys.stdout.flush()
            step += 1
        if band.failure:
            print(f"\n  the glow stopped: {band.failure}")
            break
        # Redrawn a few times a second rather than every pass. The carriage
        # return makes this one updating line in a terminal, and hundreds of
        # lines anywhere the output is captured instead.
        if time.perf_counter() >= next_line:
            next_line = time.perf_counter() + 0.5
            sys.stdout.write(
                f"\r  {elapsed:5.1f}s  level {scripted_level(cycle_t):.2f}  ")
            sys.stdout.flush()
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    band.stop()

print("\n")
print("-" * 62)
if band.failure:
    print(f"  FAILED          {band.failure}")
else:
    print(f"  monitors lit    {1 if args.monitors == 'primary' else len(screens)}")
    print(f"  windows         {4 * (1 if args.monitors == 'primary' else len(screens))}")
    if frames:
        ms = sorted(f * 1000 for f in frames)
        print(f"  level callback  {statistics.median(ms):.4f}ms median, "
              f"{ms[int(len(ms) * 0.95)]:.4f}ms p95")
    print(f"  frames drawn    ~{int(elapsed * glow.FPS)} at {glow.FPS}fps")
    print("  -> if the band stuttered, lower --thickness or use one monitor.")
print("-" * 62)
