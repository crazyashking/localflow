"""Verify the overlay can be moved, stays reachable, and remembers where it went.

Drives the real drag handlers with synthetic events rather than a live mouse,
so this runs unattended. Covers the failure that would be worst in practice: a
saved position pointing at a monitor that is no longer attached, which would
strand the capsule off screen with no way to drag it back.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow.overlay import WaveOverlay  # noqa: E402

failures = 0
moves: list[tuple[int, int]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures += 1


class FakeEvent:
    """Stands in for a tkinter mouse event."""

    def __init__(self, x_root: int, y_root: int):
        self.x_root = x_root
        self.y_root = y_root


from localflow.config import DEFAULTS  # noqa: E402

overlay = WaveOverlay(
    get_level=lambda: 0.0,
    opacity=DEFAULTS["overlay_opacity"],
    on_move=lambda x, y: moves.append((x, y)),
)

print("overlay drag checks")


def driver() -> None:
    time.sleep(1.5)
    root = overlay._root
    assert root is not None

    vx, vy, vw, vh = overlay._virtual_desktop()
    print(f"  virtual desktop: origin ({vx}, {vy}) size {vw}x{vh}")

    start = (root.winfo_x(), root.winfo_y())
    print(f"  start position: {start}")

    # Drag up and to the left by a known amount.
    overlay._on_drag_start(FakeEvent(500, 500))
    overlay._on_drag_move(FakeEvent(500 - 120, 500 - 200))
    root.update()
    moved = (root.winfo_x(), root.winfo_y())
    print(f"  after drag:     {moved}")
    check("window actually moved", moved != start)
    check("moved by the drag delta",
          abs((moved[0] - start[0]) - -120) <= 2 and abs((moved[1] - start[1]) - -200) <= 2,
          f"(delta {moved[0] - start[0]}, {moved[1] - start[1]})")

    overlay._on_drag_end(FakeEvent(380, 300))
    check("position reported for saving", len(moves) == 1, f"({moves})")
    check("reported position matches window", moves and moves[0] == moved)

    # Clamping: absurd coordinates must land back on the visible desktop.
    for label, (px, py) in {
        "far right": (vx + vw + 5000, vy + 100),
        "far below": (vx + 100, vy + vh + 5000),
        "far left": (vx - 5000, vy + 100),
        "far above": (vx + 100, vy - 5000),
    }.items():
        cx, cy = overlay._clamp_to_desktop(px, py)
        on_screen = (
            vx <= cx <= vx + vw - overlay.width and vy <= cy <= vy + vh - overlay.height
        )
        check(f"clamps {label} back on screen", on_screen, f"({px},{py} -> {cx},{cy})")

    # A position from a detached monitor must not strand the overlay.
    ghost_x, ghost_y = overlay._clamp_to_desktop(-99999, -99999)
    root.geometry(f"+{ghost_x}+{ghost_y}")
    root.update()
    check("recovers from an off-screen saved position",
          vx <= root.winfo_x() <= vx + vw and vy <= root.winfo_y() <= vy + vh,
          f"(landed at {root.winfo_x()}, {root.winfo_y()})")

    # Opacity must reach the configured value, and out-of-range settings must
    # not be able to make the overlay invisible or unfindable.
    deadline = time.perf_counter() + 3.0
    while time.perf_counter() < deadline and overlay._alpha < overlay.opacity - 1e-6:
        time.sleep(0.05)
    actual = float(root.attributes("-alpha"))
    check("fades in to the configured opacity", abs(actual - overlay.opacity) < 0.02,
          f"(configured {overlay.opacity:.2f}, actual {actual:.2f})")
    check("default opacity is translucent, not solid", overlay.opacity < 1.0,
          f"({overlay.opacity:.2f})")

    from localflow.overlay import MIN_OPACITY

    too_low = WaveOverlay(get_level=lambda: 0.0, opacity=0.0)
    too_high = WaveOverlay(get_level=lambda: 0.0, opacity=5.0)
    check("clamps a too-low opacity", too_low.opacity == MIN_OPACITY,
          f"(0.0 -> {too_low.opacity})")
    check("clamps a too-high opacity", too_high.opacity == 1.0, f"(5.0 -> {too_high.opacity})")

    # Negative coordinates are legal: monitors left of or above the primary.
    if vx < 0:
        root.geometry(f"+{vx}+{vy + 50}")
        root.update()
        check("accepts negative coordinates", root.winfo_x() == vx,
              f"(asked {vx}, got {root.winfo_x()})")
    else:
        print("  SKIP  negative coordinates (no monitor left of the primary)")

    overlay.stop()


threading.Thread(target=driver, daemon=True).start()
overlay.run()

print(f"\n{'drag works' if not failures else f'{failures} check(s) failed'}")
raise SystemExit(1 if failures else 0)
