"""Check the colour engine numerically instead of trusting the eye.

"Smooth" has a testable meaning here: stepping the level or the state must
never produce a large jump in the drawn colour, and blends must stay saturated
rather than washing out to grey the way RGB interpolation does.
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import overlay as ov  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def rgb(hsv: tuple[float, float, float]) -> tuple[float, float, float]:
    return colorsys.hsv_to_rgb(*hsv)


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    ra, rb = rgb(a), rgb(b)
    return sum((x - y) ** 2 for x, y in zip(ra, rb)) ** 0.5


print("overlay colour checks")

# 1. Every state must be a clearly distinct colour, or the display says nothing.
names = list(ov.STATE_COLOURS)
closest, closest_pair = 9.9, ("", "")
for i, a in enumerate(names):
    for b in names[i + 1 :]:
        d = distance(ov.STATE_COLOURS[a], ov.STATE_COLOURS[b])
        if d < closest:
            closest, closest_pair = d, (a, b)
check("states are visually distinct", closest > 0.25,
      f"(closest pair {closest_pair[0]}/{closest_pair[1]} at {closest:.2f})")

# 2. Active states must stay vivid. Idle is dim on purpose, so it is excluded.
least_sat = min(ov.STATE_COLOURS[s][1] for s in ("quiet", "speaking", "busy"))
check("active states stay saturated", least_sat > 0.45, f"(min saturation {least_sat:.2f})")

# 3. The blend must take the short way around the wheel.
blended = ov._lerp_hsv((0.95, 1.0, 1.0), (0.05, 1.0, 1.0), 0.5)
check("hue blend takes the short path", blended[0] > 0.95 or blended[0] < 0.05,
      f"(midpoint hue {blended[0]:.3f})")

# 4. Easing between any pair of states must converge with no visible frame jump.
PAIRS = [
    (f"{a} -> {b}", ov.STATE_COLOURS[a], ov.STATE_COLOURS[b])
    for a in names
    for b in names
    if a != b
]
for label, source, target in PAIRS:
    colour = source
    biggest = 0.0
    for _ in range(900):
        nxt = ov._ease_colour(colour, target)
        biggest = max(biggest, distance(colour, nxt))
        colour = nxt
    converged = distance(colour, target)
    check(f"eases smoothly, {label}", biggest < 0.05 and converged < 0.02,
          f"(max frame step {biggest:.4f}, final gap {converged:.4f})")

# 6. No transition may detour through green, which belongs to no state and
#    reads as a glitch when it flashes past.
for label, source, target in PAIRS:
    colour = source
    greenest = 0.0
    for _ in range(900):
        colour = ov._ease_colour(colour, target)
        r, g, b = rgb(colour)
        # "Green" meaning green clearly dominates both other channels.
        greenest = max(greenest, g - max(r, b))
    check(f"no green detour, {label}", greenest < 0.15, f"(green excess {greenest:.3f})")

# 6. No drawn colour may collide with the transparency key, or Windows would
#    punch it into a hole and it would vanish.
key = ov._hex_to_hsv(ov.TRANSPARENT_KEY)
drawn = list(ov.STATE_COLOURS.values()) + [ov.BG_TOP, ov.BG_BOTTOM]
for state in ov.STATE_COLOURS.values():
    drawn.append(ov._lerp_hsv(ov.BG_TOP, state, 0.12))
    drawn.append(ov._lerp_hsv(ov.BG_BOTTOM, state, 0.06))
nearest = min(distance(key, c) for c in drawn)
check("no colour collides with the transparency key", nearest > 0.06,
      f"(closest {nearest:.3f})")

# 7. Loudness must NOT drive colour. Only state may. This is the behaviour
#    that was explicitly asked for: the wave moves with the voice, the colour
#    reports what the app is doing.
overlay_obj = ov.WaveOverlay.__new__(ov.WaveOverlay)
overlay_obj.state = "recording"
overlay_obj._speaking = False
overlay_obj._hang = 0
seen = set()
for level in (0.25, 0.45, 0.70, 0.95):
    overlay_obj._level = level
    seen.add(overlay_obj._target_colour())
check("loudness alone does not change colour", len(seen) == 1,
      f"({len(seen)} distinct colour(s) across loud levels)")

# 8. Speech detection must have hysteresis, so a voice near the threshold
#    cannot make the colour stutter between quiet and speaking.
check("speech thresholds are hysteretic", ov.SPEECH_ON > ov.SPEECH_OFF,
      f"(on {ov.SPEECH_ON}, off {ov.SPEECH_OFF})")

overlay_obj._speaking = False
overlay_obj._hang = 0
overlay_obj._level = ov.SPEECH_ON + 0.01
overlay_obj._target_colour()
check("rises into speaking", overlay_obj._speaking)
# Sitting between the two thresholds must hold the current state, not flip.
overlay_obj._level = (ov.SPEECH_ON + ov.SPEECH_OFF) / 2
overlay_obj._target_colour()
check("holds through the middle band", overlay_obj._speaking)
overlay_obj._level = ov.SPEECH_OFF - 0.01
for _ in range(ov.SPEECH_HANG_FRAMES + 2):
    overlay_obj._target_colour()
check("falls back to quiet after the hang", not overlay_obj._speaking)

print(f"\n{'colour engine is sound' if not failures else f'{failures} check(s) failed'}")
raise SystemExit(1 if failures else 0)
