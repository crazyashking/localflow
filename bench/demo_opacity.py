"""Show the capsule at several opacity levels so you can pick one by eye.

Cycles through candidate values, holding each long enough to judge, and prints
which is on screen. Put your favourite in settings.json as overlay_opacity.

Run:  .venv\\Scripts\\python.exe bench\\demo_opacity.py
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow.overlay import WaveOverlay  # noqa: E402

LEVELS = [1.00, 0.90, 0.78, 0.66, 0.55, 0.45]
SECONDS = 4.0

start = time.perf_counter()


def level_source() -> float:
    """Speech-like movement, so each opacity is judged with the wave active."""
    t = time.perf_counter() - start
    syllable = abs(math.sin(t * 5.5)) ** 1.6
    wobble = 0.75 + 0.25 * math.sin(t * 1.7)
    return min(1.0, syllable * wobble * 0.55)


overlay = WaveOverlay(get_level=level_source, opacity=LEVELS[0])


def driver() -> None:
    time.sleep(1.2)
    overlay.set_state("recording")
    print("comparing opacity levels")
    print("  put something bright and something dark behind the capsule to judge it\n")

    for value in LEVELS:
        # Drive the window directly; the tick loop only ever fades upward.
        overlay.opacity = value
        overlay._alpha = value
        if overlay._root is not None:
            overlay._root.attributes("-alpha", value)
        label = "current default" if abs(value - 0.78) < 1e-6 else ""
        print(f"  overlay_opacity = {value:.2f}   {label}")
        time.sleep(SECONDS)

    print("\n  set your preferred value as overlay_opacity in settings.json")
    overlay.stop()


threading.Thread(target=driver, daemon=True).start()
overlay.run()
print("done")
