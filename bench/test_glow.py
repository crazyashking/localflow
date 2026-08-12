"""Checks on the edge glow that a demo cannot make.

demo_glow.py shows what it looks like, and looking at it is the only way to
judge the look. These are the parts where "it looked fine" is not evidence: the
four bands tiling the rim exactly, the falloff reaching zero, the state machine
following the app, and the whole thing staying inside a frame budget.

The geometry checks run against fake monitor sizes and never open a window, so
they are safe in a background session. The one test that does create windows is
marked and skipped if there is no desktop to put them on.

Run:  .venv\\Scripts\\python.exe bench\\test_glow.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import glow  # noqa: E402
from localflow.overlay import STATE_COLOURS  # noqa: E402

failures = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global failures
    if not ok:
        failures += 1
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag}  {label}" + (f"  ({detail})" if detail else ""))


def band_rects(w: int, h: int, thickness: int) -> list[tuple[int, int, int, int]]:
    """The same split _build uses, kept here so the test can check it alone."""
    mid = max(0, h - 2 * thickness)
    rects = [(0, 0, w, thickness), (0, h - thickness, w, thickness)]
    if mid > 0:
        rects += [(0, thickness, thickness, mid),
                  (w - thickness, thickness, thickness, mid)]
    return rects


print("band geometry")
for w, h, frac in ((2560, 1440, 0.11), (1920, 1080, 0.11), (3840, 2160, 0.11),
                   (1366, 768, 0.20), (800, 600, 0.30)):
    thickness = max(8, int(round(min(w, h) * frac)))
    rects = band_rects(w, h, thickness)
    # Paint every band onto one canvas and count. Twice-covered pixels would
    # composite twice on screen and the corners would read as bright blobs.
    canvas = np.zeros((h, w), dtype=np.uint8)
    for x, y, bw, bh in rects:
        canvas[y:y + bh, x:x + bw] += 1
    lit = canvas > 0
    expected = np.zeros((h, w), dtype=bool)
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.minimum(np.minimum(xs, w - 1 - xs), np.minimum(ys, h - 1 - ys))
    expected = dist < thickness
    check(canvas.max() <= 1, f"{w}x{h}: no pixel covered twice",
          f"max coverage {canvas.max()}")
    check(bool(np.array_equal(lit, expected)),
          f"{w}x{h}: bands cover exactly the rim", f"{thickness}px band")

print("\nfalloff")
g = glow.BorderGlow(lambda: 0.0)
g._alpha = 1.0
g._colour = STATE_COLOURS["speaking"]
for thickness in (119, 158, 237):
    g._level = 1.0
    lut = g._build_lut(thickness)
    check(lut.shape == (thickness, 4), f"table spans the {thickness}px band",
          str(lut.shape))
    check(lut[-1, 3] == 0, f"{thickness}px: alpha reaches zero at the inner edge",
          f"got {lut[-1, 3]}")
    check(lut[0, 3] > 150, f"{thickness}px: bright at the screen edge",
          f"alpha {lut[0, 3]}")
    check(bool(np.all(np.diff(lut[:, 3].astype(int)) <= 0)),
          f"{thickness}px: fades monotonically inward")
    # Premultiplied alpha: no colour channel may exceed the alpha it is scaled
    # by. Straight (unpremultiplied) colour here washes the band toward white
    # as it fades, which is the classic ULW_ALPHA mistake.
    check(bool(np.all(lut[:, :3].max(axis=1) <= lut[:, 3])),
          f"{thickness}px: colour is premultiplied by alpha")

print("\nthickness follows the voice")
g._level = 0.0
quiet = g._build_lut(158)[:, 3]
g._level = 1.0
loud = g._build_lut(158)[:, 3]
quiet_reach = int(np.count_nonzero(quiet))
loud_reach = int(np.count_nonzero(loud))
check(loud_reach > quiet_reach, "a loud frame reaches further in",
      f"{quiet_reach}px quiet vs {loud_reach}px loud")
check(quiet_reach > 0, "a silent armed frame is still visible", f"{quiet_reach}px")

print("\nstate machine")
g = glow.BorderGlow(lambda: 0.0)
check(not g._visible(), "hidden while ready")
g.set_state("loading")
check(not g._visible(), "hidden while loading")

g.set_state("recording")
g._level = 0.0
check(g._visible(), "visible the moment recording starts")
armed = g._target_brightness()
check(armed > 0.0, "and armed has a brightness of its own", f"{armed:.2f}")
check(g._target_colour() == STATE_COLOURS["quiet"], "armed takes the quiet colour")

g._level = 0.5
g._target_colour()
check(g._speaking, "speech flips it to speaking above the on threshold")
check(g._target_colour() == STATE_COLOURS["speaking"], "and takes the speaking colour")
loud = g._target_brightness()
check(loud > armed, "and brightens past armed", f"{armed:.2f} -> {loud:.2f}")

# A plain multiply saturates partway up and then stops responding, which makes
# the loud half of your range look identical. Every step must move.
steps = [glow._response(x / 10) for x in range(11)]
check(all(b > a + 0.004 for a, b in zip(steps, steps[1:], strict=False)),
      "loudness keeps responding all the way to full scale",
      f"{steps[7]:.3f} at 0.7, {steps[10]:.3f} at 1.0")
check(abs(steps[0]) < 1e-9 and abs(steps[10] - 1.0) < 1e-9,
      "and still spans exactly 0 to 1")

# The hang is what stops the colour flapping between words.
g._level = 0.0
held = [bool(g._target_colour() == STATE_COLOURS["speaking"]) for _ in range(4)]
check(all(held), "a gap between words does not drop it out of speaking")
for _ in range(40):
    g._target_colour()
check(not g._speaking, "a long silence does drop it back to quiet")

g.set_state("transcribing")
check(g._target_colour() == STATE_COLOURS["busy"], "decoding takes the busy colour")
samples = []
for _ in range(int(glow.FPS * 2)):
    g._phase += glow.FRAME_S
    samples.append(g._target_brightness())
check(max(samples) > min(samples) + 0.05, "decoding pulses rather than sitting still",
      f"{min(samples):.2f} to {max(samples):.2f}")
check(min(samples) > 0.0, "and never blinks fully out")

g.set_state("ready")
check(g._visible(), "it lingers after the text lands instead of vanishing")
check(g._target_colour() == STATE_COLOURS["quiet"],
      "and returns to the listening colour rather than holding amber")
check(g._target_brightness() < armed,
      "dimmer than an armed session, so it reads as waiting",
      f"{g._target_brightness():.2f} vs {armed:.2f}")
g._settled_at = time.perf_counter() - glow.LINGER_S - 0.1
check(not g._visible(), "then leaves once the linger is up")

# Only a finished utterance earns the linger. Reaching "ready" at startup must
# not light the whole screen for two seconds.
fresh = glow.BorderGlow(lambda: 0.0)
fresh.set_state("loading")
fresh.set_state("ready")
check(not fresh._visible(), "startup reaching ready does not trigger the linger")
fresh.set_state("error")
fresh.set_state("ready")
check(not fresh._visible(), "and neither does recovering from an error")

zero = glow.BorderGlow(lambda: 0.0, linger_seconds=0.0)
zero.set_state("recording")
zero.set_state("transcribing")
zero.set_state("ready")
check(not zero._visible(), "linger_seconds 0 leaves the instant it is done")

print("\nsession cap")
g = glow.BorderGlow(lambda: 0.0, max_seconds=1.0)
g.set_state("recording")
g._speaking = True
g._level = 1.0
check(g._visible(), "lit inside the cap")
g._session_started = time.perf_counter() - 30.0
check(not g._visible(), "and gives up after max_seconds + 5",
      "a glow cannot outlive its utterance")

print("\nsmoothness")
# Nothing may jump. Every one of these ran as a visible step before the easing
# was reworked, and a step in a large peripheral shape reads as a glitch.
#
# The fade in is measured with the sweep switched off. With it on the reveal is
# spatial, so overall brightness is meant to come up at once while the gain
# along the rim does the hiding; measuring a global fade there would be
# measuring something the design deliberately does not do.
g = glow.BorderGlow(lambda: 0.9, sweep_seconds=0.0, linger_seconds=0.0)
g.set_state("recording")
alphas, ripples = [], []
for _ in range(int(glow.FPS * 1.5)):
    g._advance()
    alphas.append(g._alpha)
    ripples.append(g._ripple)
def steps_of(series: list[float]) -> list[float]:
    # strict=False throughout this file: every zip here pairs each value with
    # the next one, so the two sides differ in length by one by construction.
    return [b - a for a, b in zip(series, series[1:], strict=False)]


def jerk_of(series: list[float]) -> float:
    """Largest frame-to-frame change in SPEED.

    Speed itself is the wrong thing to bound: a 0.22s fade has to cover ground,
    so a tight cap on it would only be a slower fade. What reads as a glitch is
    speed changing abruptly, which is what this measures.
    """
    s = steps_of(series)
    return max(abs(b - a) for a, b in zip(s, s[1:], strict=False))


jump = max(abs(x) for x in steps_of(alphas))
check(jump < 0.15, "brightness never snaps between frames", f"largest step {jump:.4f}")
check(jerk_of(alphas) < 0.05, "and speeds up and slows down smoothly",
      f"largest change in speed {jerk_of(alphas):.4f}")
check(alphas[0] < 0.02, "starts from dark")
check(alphas[-1] > 0.5, "and arrives lit", f"{alphas[-1]:.2f}")
# Smoothstep means the fade leaves and arrives slowly, so the fastest frame sits
# in the middle of the ramp rather than at its start.
fastest = max(range(len(alphas) - 1), key=lambda i: alphas[i + 1] - alphas[i])
check(fastest > 2, "the fade eases out of rest instead of snapping",
      f"fastest frame is {fastest} of {len(alphas)}")

g.set_state("ready")
g._speaking = False
tail = []
for _ in range(int(glow.FPS * 1.5)):
    g._advance()
    tail.append(g._alpha)
check(max(abs(x) for x in steps_of(tail)) < 0.15, "and never snaps on the way out",
      f"largest step {max(abs(x) for x in steps_of(tail)):.4f}")
check(jerk_of(tail) < 0.05, "easing out as smoothly as it eased in",
      f"largest change in speed {jerk_of(tail):.4f}")
check(tail[-1] < 0.01, "fading fully away", f"{tail[-1]:.4f}")
ripple_jump = max(abs(b - a) for a, b in zip(ripples, ripples[1:], strict=False))
check(ripple_jump < 0.06, "the travelling ripple eases in rather than switching on",
      f"largest step {ripple_jump:.4f}")

# Easing is written as time constants, so the animation must last the same
# wall-clock time whatever the frame rate. A fraction-per-frame ease would
# double in speed here.
check(abs(glow._coef(0.1) - (1 - 2.718281828 ** (-glow.FRAME_S / 0.1))) < 1e-6,
      "easing is derived from the frame rate, not hardcoded per frame")

# With the sweep on, brightness still must not snap, and the sweep itself has to
# advance smoothly. The reveal being spatial is no excuse for a jolt.
g = glow.BorderGlow(lambda: 0.9, linger_seconds=0.0)
g.set_state("recording")
swept, alphas = [], []
for _ in range(int(glow.FPS * 1.2)):
    g._advance()
    swept.append(g._sweep)
    alphas.append(g._alpha)
check(max(abs(x) for x in steps_of(alphas)) < 0.15,
      "with the sweep on, brightness still never snaps",
      f"largest step {max(abs(x) for x in steps_of(alphas)):.4f}")
check(swept[-1] >= 1.0, "the sweep completes", f"{swept[-1]:.2f}")
finished = next(i for i, s in enumerate(swept) if s >= 1.0)
# Constant rate only while it is travelling. The last step is short because
# progress clamps at 1.0, which is the arrival rather than a stutter. The
# easing that shapes the visible motion lives in _sweep_front, not here.
travelling = swept[:finished]
check(jerk_of(travelling) < 1e-6, "and advances at a constant rate on the way",
      f"{len(travelling)} frames")
check(abs(finished / glow.FPS - glow.SWEEP_S) < 0.08,
      "in about the time it is configured to take",
      f"{finished / glow.FPS:.2f}s vs {glow.SWEEP_S:g}s")

# The sweep must restart for the next utterance, and must not reset while the
# band is still fading out, or the fade out would turn back into a wipe.
g.set_state("ready")
g._advance()
check(g._sweep >= 1.0, "the sweep holds while the band fades out",
      f"{g._sweep:.2f}")
for _ in range(int(glow.FPS * 2)):
    g._advance()
check(g._sweep == 0.0, "and rearms once it is fully gone")

print("\nsweep")
W, H, T = 2560, 1440, 158
PERIM = 2.0 * (W + H)
coords = {e: glow._perimeter_coord(e, W, H, T)
          for e in ("top", "right", "bottom", "left")}
# Walking clockwise, each edge must pick up where the last left off. A gap here
# is a corner the light jumps across instead of flowing through.
order = np.concatenate([coords["top"], coords["right"],
                        coords["bottom"], coords["left"]])
check(bool(np.all(np.diff(coords["top"]) > 0)), "top runs left to right")
check(bool(np.all(np.diff(coords["right"]) > 0)), "right carries on downward")
check(bool(np.all(np.diff(coords["bottom"]) < 0)),
      "bottom runs back the other way, as a clockwise walk does")
check(bool(np.all(np.diff(coords["left"]) < 0)), "left closes the loop upward")
check(order.min() >= 0 and order.max() < PERIM,
      "every position lands on the rim", f"0 to {order.max():.0f} of {PERIM:.0f}")
# The corner is where two windows meet, and the two sides of it must be within a
# band's thickness of each other or the wavefront visibly tears.
seam = abs(coords["top"][-1] - coords["right"][0])
check(seam <= T + 1, "the top and right bands meet without a tear",
      f"{seam:.0f}px apart, band is {T}px")

for edge in ("right", "left", "top", "bottom"):
    origin = glow._origin_coord(edge, W, H)
    arc = glow._arc_distance(coords[edge], origin, PERIM, lap=False)
    check(float(arc.min()) < T, f"light starting {edge} begins on the {edge} edge",
          f"nearest point {arc.min():.0f}px")
    far = max(float(glow._arc_distance(coords[e], origin, PERIM, lap=False).max())
              for e in coords)
    check(abs(far - PERIM / 2) < T * 2, "and the far point is half the rim away",
          f"{far:.0f} of {PERIM / 2:.0f}")

check(bool(np.all(glow._arc_distance(coords["top"], glow._origin_coord(
    "top", W, H), PERIM, lap=False) <= PERIM / 2 + 1)),
    "splitting never travels more than half the perimeter")

# A lap goes one way round the whole screen instead of splitting. Every edge
# must sit in clockwise order and the last of them must be nearly a full
# perimeter out, or the light is not actually going all the way round.
lap_origin = glow._origin_coord("top-left", W, H)
lap_arc = {e: glow._arc_distance(coords[e], lap_origin, PERIM, lap=True)
           for e in coords}
check(float(lap_arc["top"].min()) < 1.0, "a lap starts at the origin",
      f"{lap_arc['top'].min():.0f}px")
reach = max(float(a.max()) for a in lap_arc.values())
check(reach > PERIM - 2 * T, "and travels a full perimeter, not half",
      f"{reach:.0f} of {PERIM:.0f}")
clockwise = ["top", "right", "bottom", "left"]
starts = [float(lap_arc[e].min()) for e in clockwise]
check(starts == sorted(starts),
      "the edges light in clockwise order",
      " then ".join(f"{e}@{s:.0f}" for e, s in zip(clockwise, starts, strict=True)))
check(bool(np.all(np.diff(lap_arc["top"]) > 0)),
      "and the light runs one way along an edge rather than outward from a point")

# Corners are the four points where the clockwise walk turns, so the light runs
# away down two edges at once instead of spreading from the middle of one side.
CORNER_EDGES = {
    "top-left": ("top", "left"), "top-right": ("top", "right"),
    "bottom-right": ("bottom", "right"), "bottom-left": ("bottom", "left"),
}
for corner, touching in CORNER_EDGES.items():
    origin = glow._origin_coord(corner, W, H)
    nearest = {e: float(glow._arc_distance(coords[e], origin, PERIM,
                                           lap=False).min())
               for e in coords}
    close = sorted(nearest, key=nearest.get)[:2]
    check(set(close) == set(touching),
          f"{corner} sends light down the {touching[0]} and {touching[1]} edges",
          f"nearest are {close[0]} and {close[1]}")
    opposite = {"top-left": "bottom-right", "top-right": "bottom-left",
                "bottom-right": "top-left", "bottom-left": "top-right"}[corner]
    far = glow._origin_coord(opposite, W, H)
    gap = abs(far - origin)
    check(abs(min(gap, PERIM - gap) - PERIM / 2) < 1.0,
          f"and converges on the {opposite} corner",
          f"{min(gap, PERIM - gap):.0f} of {PERIM / 2:.0f}")

check(all(o in glow.SWEEP_ORIGINS for o in CORNER_EDGES),
      "every corner is an accepted setting")
check(glow.BorderGlow(lambda: 0.0, sweep_from="nonsense").sweep_from
      in glow.SWEEP_ORIGINS,
      "an unknown origin falls back rather than crashing")
check(glow.BorderGlow(lambda: 0.0, sweep_style="nonsense").sweep_style
      in glow.SWEEP_STYLES, "and so does an unknown style")

# The sweep has to finish. If the front stops at the meeting point the crest
# parks there and leaves a permanent bright spot on the far side.
g = glow.BorderGlow(lambda: 0.0)
g._half_perimeter = PERIM / 2
g._sweep_span = PERIM          # a lap, so the front must cross the whole rim
g._sweep = 0.0
check(g._sweep_front() == 0.0, "the front starts at the origin")
g._sweep = 0.999
front = g._sweep_front()
check(front > PERIM, "and overruns the finish before it ends",
      f"{front:.0f} past {PERIM:.0f}")
g._sweep_span = PERIM / 2      # a split finishes at the far side instead
check(g._sweep_front() > PERIM / 2, "a split overruns its own meeting point",
      f"{g._sweep_front():.0f} past {PERIM / 2:.0f}")
g._sweep = 1.0
check(g._sweep_front() is None,
      "then stops being computed at all, so the steady state costs nothing")

g = glow.BorderGlow(lambda: 0.0, sweep_seconds=0.0)
check(g._sweep_front() is None, "sweep_seconds 0 disables it")

print("\nsweep gain")
g = glow.BorderGlow(lambda: 0.0)
g._half_perimeter = PERIM / 2
band_arc = glow._arc_distance(coords["top"], glow._origin_coord("right", W, H),
                              PERIM)
lead = (PERIM / 2) * glow.SWEEP_LEAD_FRAC
trail = (PERIM / 2) * glow.SWEEP_TRAIL_FRAC
probe = glow._Band.__new__(glow._Band)
probe.arc = band_arc
probe._gain = np.empty_like(band_arc)
probe._offset = np.empty(band_arc.shape, dtype=np.uint16)

# Derived from the band rather than hardcoded: a band's distance from the origin
# depends on the sweep style, and a fixed number here silently stopped landing
# inside the band when "lap" was added. Halfway along it is always valid.
probe_front = float(band_arc.min() + band_arc.max()) / 2
check(float(band_arc.min()) < probe_front < float(band_arc.max()),
      "the probe front lands inside the band being measured",
      f"front {probe_front:.0f} in {band_arc.min():.0f} to {band_arc.max():.0f}")
early = probe.sweep_offset(probe_front, lead, trail, T).copy()
check(int(early.min()) == 0, "ahead of the wavefront is still dark",
      f"lowest gain row {early.min() // T}")
check(int(early.max()) > 0, "and behind it is lit",
      f"highest gain row {early.max() // T}")

# The crest: somewhere just behind the front the gain must exceed the settled
# level, which is what makes it read as light arriving rather than a wipe.
settled_row = int(round((glow.SWEEP_GAIN_STEPS - 1) / (1.0 + glow.SWEEP_BOOST)))
check(int(early.max() // T) > settled_row,
      "the wavefront carries a crest brighter than the settled band",
      f"peak row {early.max() // T} vs settled {settled_row}")

done = probe.sweep_offset(PERIM, lead, trail, T)
rows = done // T
check(int(rows.min()) >= settled_row - 2,
      "once it has passed, the whole band sits at the settled level",
      f"rows {rows.min()} to {rows.max()}")
check(int(rows.max()) <= glow.SWEEP_GAIN_STEPS - 1,
      "and no index escapes the table", f"max row {rows.max()}")

print("\nthe wash")
# The default entrance. Light crosses the whole screen and then recedes into the
# rim, so the two things worth proving are that the crossing is complete and
# that the recession lands exactly on the steady rim: any mismatch there is a
# visible flash at the handoff, on every single utterance.
for direction, axis, growing in (("left", 1, True), ("right", 1, False),
                                 ("top", 0, True), ("bottom", 0, False)):
    field = glow._wash_coord(direction, 64, 48)
    step = np.diff(field, axis=axis)
    check(bool(np.all(step > 0 if growing else step < 0)),
          f"{direction} runs the way the light travels")
    check(float(field.min()) >= 0.0 and float(field.max()) <= 1.0,
          f"{direction} stays in the 0-to-1 range",
          f"{field.min():.2f} to {field.max():.2f}")
for direction in ("top-left", "top-right", "bottom-left", "bottom-right"):
    field = glow._wash_coord(direction, 64, 48)
    check(bool(np.all(np.diff(field, axis=0) != 0))
          and bool(np.all(np.diff(field, axis=1) != 0)),
          f"{direction} moves on both axes, so the front is diagonal")

g = glow.BorderGlow(lambda: 0.0, sweep_style="wash", sweep_seconds=0.7)
g._alpha, g.opacity = 1.0, 1.0
fronts = []
for i in range(41):
    g._sweep = i / 40
    fronts.append(g._wash_front())
check(fronts[0] <= -glow.FLOOD_EDGE + 1e-6,
      "the first frame has the light still off the near edge",
      f"front {fronts[0]:.2f}")
check(fronts[-1] >= 1.0,
      "and the last has it past the far edge", f"front {fronts[-1]:.2f}")
check(bool(np.all(np.diff(fronts) > 0)),
      "the front only ever moves forward")

# The flood term over the interior. Zero at both ends and a peak in between is
# what makes the wash arrive and leave rather than staying on top of the rim.
g._sweep = 0.0
first = g._build_wash_lut(T)
g._sweep = 0.5
middle = g._build_wash_lut(T)
g._sweep = 1.0
last = g._build_wash_lut(T)
plain = g._build_lut(T)
interior = (glow.FLOOD_STEPS - 1) * T + T - 1     # full gain, deepest rim row
check(int(first[interior][3]) == 0 and int(last[interior][3]) == 0,
      "the interior is dark at both ends of the wash",
      f"alpha {first[interior][3]} then {last[interior][3]}")
check(int(middle[interior][3]) > 40,
      "and lit across the middle of it", f"alpha {middle[interior][3]}")
check(np.array_equal(last[(glow.FLOOD_STEPS - 1) * T:], plain),
      "the wash ends on exactly the rim the bands take over with")
check(int(first[0][3]) == 0,
      "no gain means no light, so the far side starts dark",
      f"alpha {first[0][3]}")

# Index arithmetic on the real render path, with the window part skipped so this
# runs headless. Out-of-range indices would clip silently and read as a band of
# wrong colour, which is exactly the kind of thing a demo does not catch.
flood = glow._Flood.__new__(glow._Flood)
flood.thickness = T
flood.coord = glow._wash_coord("left", 64, 48)
flood.rim = np.clip(
    np.minimum(np.arange(64)[None, :], 63 - np.arange(64)[None, :])
    + 0 * np.arange(48)[:, None], 0, T - 1).astype(np.uint16)
flood._gain = np.empty_like(flood.coord)
flood._idx = np.empty(flood.coord.shape, dtype=np.uint16)
flood.pixels = np.zeros((48, 64, 4), dtype=np.uint8)
g._sweep = 0.5
half = g._build_wash_lut(T)
flood.render(half, 0.5)
check(int(flood._idx.max()) < len(half),
      "every index lands inside the table, with nothing clipped",
      f"max {flood._idx.max()} of {len(half)}")
near = int(flood.pixels[24, :8, 3].mean())
far = int(flood.pixels[24, -8:, 3].mean())
check(near > far, "behind the front is brighter than ahead of it",
      f"{near} vs {far}")
flood.render(half, -glow.FLOOD_EDGE)
check(int(flood.pixels[..., 3].max()) == 0,
      "and with the front off the edge the screen is untouched")

g._sweep = 1.0
check(not g._washing(), "the wash stops owning the screen once it is done")
g._sweep = 0.5
check(g.sweep_style == "wash" and g.sweep_seconds > 0
      and not g._washing(),
      "and it never runs at all when no flood window was built")

print("\nsurviving a sleep")
# A layered window's device context and bitmap belong to a display
# configuration. Sleeping the machine invalidates them, UpdateLayeredWindow then
# fails silently forever, and the glow is gone for the rest of the session. The
# recovery is a full rebuild, so what matters is that a rebuild is safe to do
# mid-utterance and that the triggers exist.
try:
    live = glow.BorderGlow(lambda: 0.6, which_monitors="all")
    live._build()
    before = [(b.w, b.h, b.thickness) for b in live._bands]
    live.state = "recording"
    live._speaking = True
    live._level = 0.6
    live._sweep = 0.42
    live._alpha = 0.8
    live._rebuild()
    after = [(b.w, b.h, b.thickness) for b in live._bands]
    check(after == before, "a rebuild puts back exactly the same bands",
          f"{len(after)} bands")
    check(live._sweep == 0.42 and live._level == 0.6,
          "and keeps the animation running rather than restarting it",
          "a rebuild mid-utterance must not replay the sweep")
    check(live._geometry() == tuple(glow.monitors()),
          "the layout it compares against matches what Windows reports")
    check(live._thicknesses and live._axes,
          "derived tables are rebuilt too, not left stale")
    for band in live._bands:
        band.destroy()
    # Rebuilding after everything is destroyed must not raise, since that is
    # what happens if a second display change lands during shutdown.
    live._bands = []
    raised = ""
    try:
        live._rebuild()
    except Exception as exc:
        # Deliberately broad. The claim being tested is that nothing at all
        # escapes _rebuild here, so narrowing this would let the very failure
        # it is looking for through.
        raised = f"{type(exc).__name__}: {exc}"
    check(not raised and bool(live._bands),
          "rebuilding from nothing puts the bands back without raising",
          raised or f"{len(live._bands)} bands")
    for band in live._bands:
        band.destroy()
except OSError as exc:
    print(f"  SKIP  no desktop to open windows on ({exc})")

check(glow.RESUME_GAP_S > glow.FRAME_S * 60,
      "the resume threshold is far longer than a slow frame",
      f"{glow.RESUME_GAP_S}s vs a {glow.FRAME_S * 1000:.0f}ms frame")

print("\ncolour rule")
# bench/test_overlay_colour.py owns the palette itself. What is untested there
# is that _ease_hsv, a SECOND easing implementation living in glow.py, obeys the
# same green rule as the capsule's. Measured the same way: green counts as green
# dominating both other channels, rather than by hue number, because that is
# what the eye actually catches.
import colorsys  # noqa: E402

for start, end in (("quiet", "speaking"), ("speaking", "busy"),
                   ("quiet", "busy"), ("busy", "quiet"), ("busy", "speaking")):
    a, b = STATE_COLOURS[start], STATE_COLOURS[end]
    # Step it the way the render loop does, easing a fraction each frame toward
    # the target, so this walks the real path and not a straight interpolation.
    current, greenest = a, 0.0
    for _ in range(120):
        current = glow._ease_hsv(current, b, 0.16)
        r, g, bl = colorsys.hsv_to_rgb(*current)
        greenest = max(greenest, g - max(r, bl))
    check(greenest < 0.15, f"no green detour, {start} to {end}",
          f"green excess {greenest:.3f}")

print("\nrender budget")
try:
    glow.enable_dpi_awareness()
    live = glow.BorderGlow(lambda: 0.8, which_monitors="all")
    live._build()
    live.state = "recording"
    live._speaking = True
    live._level = 0.8
    live._alpha = 0.9
    for _ in range(5):
        luts = {t: live._build_lut(t) for t in live._thicknesses}
        wave = live._shimmer()
        for band in live._bands:
            band.render(luts[band.thickness],
                        wave.get(band.w if band.horizontal else band.h), None)
    times = []
    for _ in range(40):
        t0 = time.perf_counter()
        luts = {t: live._build_lut(t) for t in live._thicknesses}
        wave = live._shimmer()
        for band in live._bands:
            band.render(luts[band.thickness],
                        wave.get(band.w if band.horizontal else band.h), None)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p95 = times[int(len(times) * 0.95)]
    budget = glow.FRAME_S * 1000
    check(p95 < budget * 0.6, f"every monitor renders inside the {budget:.0f}ms frame",
          f"p95 {p95:.1f}ms across {len(live._bands)} bands")
    covered = sum(b.w * b.h for b in live._bands)
    full = sum(w * h for _, _, w, h in glow.monitors())
    check(covered < full * 0.45, "bands cost a fraction of a full screen redraw",
          f"{covered / 1e6:.2f}M vs {full / 1e6:.2f}M pixels")

    # The sweep is the expensive path: a table with a gain axis, and an index
    # offset per band. It runs for half a second at the start of every
    # utterance, so it has to fit the same budget as the steady state.
    live._half_perimeter = 4000.0
    lead = live._half_perimeter * glow.SWEEP_LEAD_FRAC
    trail = live._half_perimeter * glow.SWEEP_TRAIL_FRAC
    sweep_times = []
    for i in range(40):
        front = (i / 40) * live._half_perimeter
        t0 = time.perf_counter()
        luts = {t: live._build_sweep_lut(t) for t in live._thicknesses}
        wave = live._shimmer()
        for band in live._bands:
            band.render(
                luts[band.thickness],
                wave.get(band.w if band.horizontal else band.h),
                band.sweep_offset(front, lead, trail, band.thickness),
            )
        sweep_times.append((time.perf_counter() - t0) * 1000)
    sweep_times.sort()
    sweep_p95 = sweep_times[int(len(sweep_times) * 0.95)]
    check(sweep_p95 < budget * 0.75, "the sweep fits the frame budget too",
          f"p95 {sweep_p95:.1f}ms of {budget:.0f}ms")

    # The wash covers the entire screen, which is the one thing the rim design
    # exists to avoid. It only fits because it draws into a buffer a sixteenth
    # of the size and lets GDI stretch it up. Without that it is about 15ms a
    # frame at 1440p and the whole entrance stutters.
    floods = [glow._Flood(mon, live._thicknesses[0], "left")
              for mon in glow.monitors()[:1]]
    wash_times = []
    for i in range(40):
        live._sweep = i / 40
        t0 = time.perf_counter()
        wash_luts = {t: live._build_wash_lut(t) for t in live._thicknesses}
        for flood in floods:
            flood.render(wash_luts[flood.thickness], live._wash_front())
        wash_times.append((time.perf_counter() - t0) * 1000)
    wash_times.sort()
    wash_p95 = wash_times[int(len(wash_times) * 0.95)]
    check(wash_p95 < budget * 0.5, "and so does the full-screen wash",
          f"p95 {wash_p95:.1f}ms of {budget:.0f}ms")
    check(floods[0].sw * floods[0].sh < floods[0].w * floods[0].h * 0.1,
          "which it manages by drawing a fraction of the pixels",
          f"{floods[0].sw}x{floods[0].sh} for a {floods[0].w}x{floods[0].h} screen")
    for flood in floods:
        flood.destroy()

    for band in live._bands:
        band.destroy()
except OSError as exc:
    print(f"  SKIP  no desktop to open windows on ({exc})")

print()
if failures:
    print(f"{failures} check(s) failed")
    raise SystemExit(1)
print("edge glow is sound")
