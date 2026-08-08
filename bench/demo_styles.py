"""Compare overlay shapes side by side and pick one by eye.

Draws five candidate meters at once. All of them are fed the same synthetic
level and driven by the real colour engine from localflow.overlay, so the only
thing that differs between rows is the shape. Judging them in one glance is the
point: descriptions of "cleaner" do not survive contact with the actual pixels.

Row 2 is what the app draws today. Row 4 is what it drew before, kept here so a
change can be judged against what it replaced rather than against memory.

The amber decoding pulse is identical in every style, so it is left out. Use
demo_opacity.py to judge translucency; this window is about shape.

Nothing here touches the app. Say which row you want and it becomes the real one.

Run:  .venv\\Scripts\\python.exe bench\\demo_styles.py
"""

from __future__ import annotations

import math
import sys
import time
import tkinter as tk
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow.overlay import (  # noqa: E402
    BG_BOTTOM,
    BG_TINT_BOTTOM,
    BG_TINT_TOP,
    BG_TOP,
    BORDER_PX,
    SPEECH_OFF,
    SPEECH_ON,
    STATE_COLOURS,
    _capsule_extents,
    _ease_colour,
    _hsv_to_hex,
    _lerp_hsv,
)

CAPSULE_W = 260
CAPSULE_H = 44
ROW_H = 66
PAD_X = 24
LABEL_X = PAD_X + CAPSULE_W + 26
FRAME_MS = 33
INSET = 20  # horizontal inset inside the capsule, matches overlay.PAD


class Shape:
    """One candidate meter. Owns its canvas items and its own level history."""

    def __init__(self, samples: int):
        self.history: deque[float] = deque([0.0] * samples, maxlen=samples)
        self.items: list[int] = []

    def create(self, canvas: tk.Canvas, x: int, y: int) -> None:
        raise NotImplementedError

    def update(self, canvas: tk.Canvas, x: int, y: int, colour: str) -> None:
        raise NotImplementedError


class Bars(Shape):
    """Discrete bars. mirrored=False anchors them to a baseline instead."""

    def __init__(self, n: int, width: int, mirrored: bool = True, min_half: float = 2.0):
        super().__init__(n)
        self.n, self.width, self.mirrored, self.min_half = n, width, mirrored, min_half

    def _step(self) -> float:
        return (CAPSULE_W - INSET * 2) / (self.n - 1)

    def create(self, canvas: tk.Canvas, x: int, y: int) -> None:
        base = y + CAPSULE_H / 2
        for i in range(self.n):
            bx = x + INSET + i * self._step()
            self.items.append(
                canvas.create_line(
                    bx, base - self.min_half, bx, base + self.min_half,
                    width=self.width, capstyle=tk.ROUND, fill="#000000",
                )
            )

    def update(self, canvas: tk.Canvas, x: int, y: int, colour: str) -> None:
        # Bottom-anchored bars get the full height to grow into, since they only
        # grow one way. Halving it would make them look weaker than the mirrored
        # rows for no reason other than the maths.
        span = (CAPSULE_H / 2) - 8
        max_amp = span if self.mirrored else span * 2
        base = y + CAPSULE_H / 2 + (0 if self.mirrored else span)
        for i, amp in enumerate(self.history):
            bx = x + INSET + i * self._step()
            edge = math.sin(math.pi * (i / (self.n - 1))) ** 0.7
            a = max(amp * max_amp * edge, self.min_half if self.mirrored else 1.0)
            top = base - a
            bottom = base + a if self.mirrored else base
            canvas.coords(self.items[i], bx, top, bx, bottom)
            canvas.itemconfigure(self.items[i], fill=colour)


class Polygon(Shape):
    """The filled mirrored blob the app used to draw. ripple is the fake motion."""

    def __init__(self, samples: int, ripple: bool):
        super().__init__(samples)
        self.samples, self.ripple = samples, ripple
        self.phase = 0.0

    def create(self, canvas: tk.Canvas, x: int, y: int) -> None:
        base = y + CAPSULE_H / 2
        self.items.append(
            canvas.create_polygon(
                [x + INSET, base, x + CAPSULE_W - INSET, base],
                smooth=True, splinesteps=10, fill="#000000", outline="#000000",
            )
        )

    def update(self, canvas: tk.Canvas, x: int, y: int, colour: str) -> None:
        self.phase += 0.30
        base = y + CAPSULE_H / 2
        step = (CAPSULE_W - INSET * 2) / (self.samples - 1)
        max_amp = (CAPSULE_H / 2) - 8
        top: list[float] = []
        bottom: list[tuple[float, float]] = []
        for i, amp in enumerate(self.history):
            bx = x + INSET + i * step
            edge = math.sin(math.pi * (i / (self.samples - 1))) ** 0.7
            wobble = 0.70 + 0.30 * math.sin(self.phase + i * 0.38) if self.ripple else 1.0
            a = max(amp * max_amp * edge * wobble, 1.1)
            top.extend((bx, base - a))
            bottom.append((bx, base + a))
        canvas.coords(self.items[0], *(top + [c for xy in reversed(bottom) for c in xy]))
        canvas.itemconfigure(self.items[0], fill=colour, outline=colour)


class Hairline(Shape):
    """A single unfilled stroke. The most minimal option on the table."""

    def __init__(self, samples: int, width: int):
        super().__init__(samples)
        self.samples, self.width = samples, width

    def create(self, canvas: tk.Canvas, x: int, y: int) -> None:
        base = y + CAPSULE_H / 2
        self.items.append(
            canvas.create_line(
                x + INSET, base, x + CAPSULE_W - INSET, base,
                width=self.width, fill="#000000", smooth=True, capstyle=tk.ROUND,
            )
        )

    def update(self, canvas: tk.Canvas, x: int, y: int, colour: str) -> None:
        base = y + CAPSULE_H / 2
        step = (CAPSULE_W - INSET * 2) / (self.samples - 1)
        max_amp = (CAPSULE_H / 2) - 8
        points: list[float] = []
        for i, amp in enumerate(self.history):
            bx = x + INSET + i * step
            edge = math.sin(math.pi * (i / (self.samples - 1))) ** 0.7
            points.extend((bx, base - amp * max_amp * edge))
        canvas.coords(self.items[0], *points)
        canvas.itemconfigure(self.items[0], fill=colour)


STYLES: list[tuple[str, Shape]] = [
    ("16 bars x 5px, chunkier", Bars(n=16, width=5)),
    ("24 bars x 3px  <- the shape the app draws", Bars(n=24, width=3)),
    ("24 bars, bottom anchored", Bars(n=24, width=3, mirrored=False)),
    ("filled polygon, 72 samples  <- the old one", Polygon(samples=72, ripple=True)),
    ("hairline stroke", Hairline(samples=48, width=2)),
]


class Row:
    def __init__(self, canvas: tk.Canvas, label: str, shape: Shape, index: int):
        self.shape = shape
        self.x, self.y = PAD_X, 24 + index * ROW_H
        self.bg: list[int] = []
        self.border: list[int] = []

        for yy, x0, x1 in _capsule_extents(CAPSULE_W, CAPSULE_H):
            self.border.append(
                canvas.create_line(self.x + x0, self.y + yy, self.x + x1, self.y + yy)
            )
        for yy, x0, x1 in _capsule_extents(CAPSULE_W, CAPSULE_H, inset=BORDER_PX):
            self.bg.append(
                canvas.create_line(self.x + x0, self.y + yy, self.x + x1, self.y + yy)
            )
        shape.create(canvas, self.x, self.y)
        canvas.create_text(
            LABEL_X, self.y + CAPSULE_H / 2, text=label, anchor="w",
            fill="#8b93a7", font=("Segoe UI", 10),
        )

    def paint(self, canvas: tk.Canvas, tint: tuple[float, float, float]) -> None:
        """Same tinting the real overlay does: the whole capsule takes the hue."""
        top = _lerp_hsv(BG_TOP, tint, BG_TINT_TOP)
        bottom = _lerp_hsv(BG_BOTTOM, tint, BG_TINT_BOTTOM)
        rim = _hsv_to_hex(
            (tint[0], min(tint[1] * 0.75, 1.0), min(tint[2] * 0.85 + 0.15, 1.0))
        )
        for item in self.border:
            canvas.itemconfigure(item, fill=rim)
        for i, item in enumerate(self.bg):
            shade = _lerp_hsv(top, bottom, i / max(len(self.bg) - 1, 1))
            canvas.itemconfigure(item, fill=_hsv_to_hex(shade))
        self.shape.update(canvas, self.x, self.y, _hsv_to_hex(tint))


root = tk.Tk()
root.title("LocalFlow overlay styles")
WIDTH, HEIGHT = 760, 24 + ROW_H * len(STYLES) + 16
root.geometry(f"{WIDTH}x{HEIGHT}")
root.configure(bg="#0d0f16")
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="#0d0f16", highlightthickness=0)
canvas.pack()

rows = [Row(canvas, label, shape, i) for i, (label, shape) in enumerate(STYLES)]

state_label = canvas.create_text(
    PAD_X, HEIGHT - 12, anchor="w", fill="#5c6478", font=("Segoe UI", 9), text="",
)

start = time.perf_counter()
colour = STATE_COLOURS["idle"]
speaking = False

print("overlay style comparison")
print("  every row gets the same level and the same colour engine")
print("  close the window when you have picked one\n")
for name, _ in STYLES:
    print(f"    {name}")


def tick() -> None:
    """One frame: advance the synthetic level, ease the colour, repaint."""
    global colour, speaking
    t = time.perf_counter() - start

    # Twelve second cycle: four seconds silent, six speaking, two silent again,
    # so the rest state and the active state are both on screen long enough to
    # judge and the colour transition is visible in both directions.
    cycle = t % 12.0
    if 4.0 <= cycle < 10.0:
        syllable = abs(math.sin(t * 5.5)) ** 1.6
        level = min(1.0, syllable * (0.75 + 0.25 * math.sin(t * 1.7)) * 0.85)
    else:
        level = 0.0

    if level >= SPEECH_ON:
        speaking = True
    elif level <= SPEECH_OFF and cycle >= 10.0:
        speaking = False
    if cycle < 4.0:
        target, name = STATE_COLOURS["idle"], "idle"
    elif speaking:
        target, name = STATE_COLOURS["speaking"], "speaking"
    else:
        target, name = STATE_COLOURS["quiet"], "listening, quiet"

    colour = _ease_colour(colour, target)
    canvas.itemconfigure(state_label, text=f"state: {name}   level: {level:.2f}")

    for row in rows:
        row.shape.history.append(level)
        row.paint(canvas, colour)

    root.after(FRAME_MS, tick)


root.after(FRAME_MS, tick)
root.mainloop()
print("\nclosed")
