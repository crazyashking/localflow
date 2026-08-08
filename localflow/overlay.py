"""Floating waveform overlay.

A draggable capsule that stays on screen for the life of the process. It shows
a flat line when idle, a live waveform while you speak, and a travelling pulse
while the model decodes.

Three decisions here are load-bearing rather than cosmetic. Each is explained
where it is enforced: window focus in _apply_window_styles, the choice of
state colours above STATE_COLOURS, and per-frame cost in _paint_background.
"""

from __future__ import annotations

import colorsys
import ctypes
import math
import tkinter as tk
from collections import deque
from collections.abc import Callable
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetParent.argtypes = (wintypes.HWND,)
user32.GetParent.restype = wintypes.HWND
user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
user32.GetWindowLongW.restype = ctypes.c_long
user32.SetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_long)
user32.SetWindowLongW.restype = ctypes.c_long
user32.SetWindowPos.argtypes = (
    wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, wintypes.UINT,
)
user32.SetWindowPos.restype = wintypes.BOOL

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOPMOST = 0x00000008

HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# Rendered as a hole by Windows, giving the capsule its rounded silhouette.
# Never used by anything we draw.
TRANSPARENT_KEY = "#010203"

SAMPLES = 72
FRAME_MS = 33          # 30fps: smooth, and light enough not to starve audio
COLOUR_EASE = 0.10
LEVEL_EASE = 0.35
MAX_RGB_STEP = 0.045

# Hysteresis for "is he actually talking". Rising edge is higher than the
# falling edge, so a voice hovering near the threshold cannot make the colour
# stutter between states.
SPEECH_ON = 0.20
SPEECH_OFF = 0.09
SPEECH_HANG_FRAMES = 12   # keep "speaking" briefly through natural pauses

BORDER_PX = 2

# Floor on window opacity. Below roughly this, the capsule is hard to see and
# hard to find in order to drag it somewhere better.
MIN_OPACITY = 0.35


def _hex_to_hsv(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(r, g, b)


def _hsv_to_hex(hsv: tuple[float, float, float]) -> str:
    r, g, b = colorsys.hsv_to_rgb(*hsv)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _lerp_hsv(
    a: tuple[float, float, float], b: tuple[float, float, float], t: float
) -> tuple[float, float, float]:
    """Blend two colours through HSV, taking the shorter way around the wheel.

    Interpolating in RGB instead passes through muddy grey on opposite hues.
    """
    ah, asat, av = a
    bh, bsat, bv = b
    dh = bh - ah
    if dh > 0.5:
        dh -= 1.0
    elif dh < -0.5:
        dh += 1.0
    return ((ah + dh * t) % 1.0, asat + (bsat - asat) * t, av + (bv - av) * t)


def _rgb_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    ar, ag, ab = colorsys.hsv_to_rgb(*a)
    br, bg, bb = colorsys.hsv_to_rgb(*b)
    return ((ar - br) ** 2 + (ag - bg) ** 2 + (ab - bb) ** 2) ** 0.5


def _ease_colour(
    current: tuple[float, float, float], target: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Move current one frame toward target, bounded by what the eye sees.

    Clamping hue, saturation and value separately only approximates this: a hue
    step costs far more rendered change at high saturation than at low.
    Bounding RGB distance enforces the property that actually matters.
    """
    proposed = _lerp_hsv(current, target, COLOUR_EASE)
    distance = _rgb_distance(current, proposed)
    if distance <= MAX_RGB_STEP:
        return proposed
    t = COLOUR_EASE * (MAX_RGB_STEP / distance)
    for _ in range(6):
        proposed = _lerp_hsv(current, target, t)
        if _rgb_distance(current, proposed) <= MAX_RGB_STEP:
            break
        t *= 0.5
    return proposed


# One colour per state. Colour reports what the app is DOING, while the wave's
# height is what follows your voice. Hue was tied to loudness first and it
# churned through every syllable, which is noisy and says nothing.
#
# Every hue here sits above 0.606, which is a hard constraint rather than a
# preference. Amber (the decoding colour) is at hue 0.106, so the shortest way
# round the wheel from anything BELOW 0.606 runs straight through green, a
# colour that belongs to no state and flashes past as an obvious glitch. Above
# that line, every transition travels the violet, pink and red side instead.
# Cyan and ordinary blue both fail this test, which is why neither is used.
# bench/test_overlay_colour.py asserts it for every pair of states.
STATE_COLOURS: dict[str, tuple[float, float, float]] = {
    "idle": (0.700, 0.30, 0.42),          # dim indigo, waiting for the hotkey
    "quiet": _hex_to_hsv("#8b5cf6"),      # violet, listening but hearing nothing
    "speaking": _hex_to_hsv("#ec4899"),   # pink, picking up your voice
    "busy": _hex_to_hsv("#f59e0b"),       # amber, decoding
}

BG_TOP = _hex_to_hsv("#232735")
BG_BOTTOM = _hex_to_hsv("#12141c")


def _capsule_extents(width: int, height: int, inset: float = 0.0) -> list[tuple[int, float, float]]:
    """Left and right edge of a stadium shape for each scanline.

    A stadium (rectangle with semicircular caps) rather than a rounded
    rectangle: the caps are true half circles, which reads as a deliberate
    shape instead of the default rounded box every overlay uses.
    """
    rows: list[tuple[int, float, float]] = []
    top = inset
    bottom = height - inset
    radius = (bottom - top) / 2
    for y in range(height):
        centre = y + 0.5
        if centre < top or centre > bottom:
            continue
        dy = centre - (top + radius)
        if abs(dy) > radius:
            continue
        dx = radius - math.sqrt(max(radius * radius - dy * dy, 0.0))
        rows.append((y, inset + dx, width - inset - dx))
    return rows


class WaveOverlay:
    """Owns the Tk window. Call run() on the main thread."""

    def __init__(
        self,
        get_level: Callable[[], float],
        width: int = 320,
        height: int = 68,
        position: tuple[int, int] | None = None,
        bottom_margin: int = 130,
        opacity: float = 0.78,
        on_move: Callable[[int, int], None] | None = None,
    ):
        self.get_level = get_level
        self.width = width
        self.height = height
        self.position = position
        self.bottom_margin = bottom_margin
        # Clamped so a bad settings value cannot make the overlay invisible or
        # leave the user unable to find it again.
        self.opacity = max(MIN_OPACITY, min(float(opacity), 1.0))
        self.on_move = on_move

        # Written by other threads, read by the animation loop.
        self.state = "idle"

        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._history: deque[float] = deque([0.0] * SAMPLES, maxlen=SAMPLES)
        self._phase = 0.0
        self._level = 0.0
        self._speaking = False
        self._hang = 0
        self._colour = STATE_COLOURS["idle"]
        self._alpha = 0.0
        self._closing = False

        # Canvas item ids, created once and reconfigured per frame.
        self._bg_items: list[int] = []
        self._border_items: list[int] = []
        self._wave_item: int | None = None
        self._pulse_item: int | None = None
        self._flat_item: int | None = None
        self._last_bg_key: str = ""

        self._drag_origin: tuple[int, int] | None = None

    # --- public API (thread safe) -----------------------------------------

    def set_state(self, state: str) -> None:
        self.state = state

    def stop(self) -> None:
        self._closing = True

    # --- window construction ----------------------------------------------

    def _build(self) -> None:
        root = tk.Tk()
        self._root = root
        root.overrideredirect(True)
        # Topmost is applied via SetWindowPos with SWP_NOACTIVATE. Tk's own
        # "-topmost" can raise the window through a path that also activates it.
        root.attributes("-alpha", 0.0)
        root.attributes("-transparentcolor", TRANSPARENT_KEY)
        root.configure(bg=TRANSPARENT_KEY)

        x, y = self._initial_position(root)
        root.geometry(f"{self.width}x{self.height}+{x}+{y}")

        self._canvas = tk.Canvas(
            root, width=self.width, height=self.height,
            bg=TRANSPARENT_KEY, highlightthickness=0, bd=0, cursor="fleur",
        )
        self._canvas.pack()
        self._create_items()

        self._canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self._canvas.bind("<B1-Motion>", self._on_drag_move)
        self._canvas.bind("<ButtonRelease-1>", self._on_drag_end)

        # Build hidden, style it, then show. Tk creates a normal, activatable
        # window; applying WS_EX_NOACTIVATE afterwards leaves a gap in which
        # Windows can hand it the foreground.
        root.withdraw()
        root.update_idletasks()
        self._apply_window_styles()
        root.deiconify()
        root.update_idletasks()
        self._apply_window_styles()
        self._place_topmost_without_activating()

    def _initial_position(self, root: tk.Tk) -> tuple[int, int]:
        if self.position is not None:
            return self._clamp_to_desktop(*self.position)
        x = (root.winfo_screenwidth() - self.width) // 2
        y = root.winfo_screenheight() - self.height - self.bottom_margin
        return x, y

    @staticmethod
    def _virtual_desktop() -> tuple[int, int, int, int]:
        """Bounds of all monitors combined, so the overlay can live on any of them."""
        return (
            user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
        )

    def _clamp_to_desktop(self, x: int, y: int) -> tuple[int, int]:
        """Keep the capsule reachable.

        Without this a saved position from a monitor that is no longer attached
        would strand the overlay off screen with no way to drag it back.
        """
        vx, vy, vw, vh = self._virtual_desktop()
        x = max(vx, min(x, vx + vw - self.width))
        y = max(vy, min(y, vy + vh - self.height))
        return x, y

    def _hwnd(self) -> int:
        assert self._root is not None
        hwnd = self._root.winfo_id()
        return user32.GetParent(hwnd) or hwnd

    def _place_topmost_without_activating(self) -> None:
        user32.SetWindowPos(
            self._hwnd(), ctypes.c_void_p(HWND_TOPMOST), 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )

    def _apply_window_styles(self) -> None:
        """Make the window non-activating and keep it out of alt-tab.

        This is the correctness requirement the whole overlay hangs on.
        Dictated text goes to whichever window holds the foreground, so an
        overlay that could activate would end up receiving its own
        transcription. Tk gives no guarantee here, hence WS_EX_NOACTIVATE.

        Click-through (WS_EX_TRANSPARENT) would also solve it, and is
        deliberately left off, because the capsule has to receive mouse events
        to be draggable. That makes WS_EX_NOACTIVATE the only thing preventing
        the bug. bench/demo_overlay.py polls the foreground window's process id
        throughout a run to prove it holds.
        """
        target = self._hwnd()
        style = user32.GetWindowLongW(target, GWL_EXSTYLE)
        user32.SetWindowLongW(
            target, GWL_EXSTYLE,
            style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
        )

    # --- dragging -----------------------------------------------------------

    def _on_drag_start(self, event: tk.Event) -> None:
        self._drag_origin = (event.x_root, event.y_root)

    def _on_drag_move(self, event: tk.Event) -> None:
        if self._drag_origin is None or self._root is None:
            return
        dx = event.x_root - self._drag_origin[0]
        dy = event.y_root - self._drag_origin[1]
        if dx == 0 and dy == 0:
            return
        x, y = self._clamp_to_desktop(
            self._root.winfo_x() + dx, self._root.winfo_y() + dy
        )
        # Negative coordinates are legal and needed for monitors left of or
        # above the primary one. Tk parses "+-1920" correctly.
        self._root.geometry(f"+{x}+{y}")
        self._drag_origin = (event.x_root, event.y_root)

    def _on_drag_end(self, event: tk.Event) -> None:
        self._drag_origin = None
        if self._root is not None and self.on_move is not None:
            self.on_move(self._root.winfo_x(), self._root.winfo_y())

    # --- canvas items -------------------------------------------------------

    def _create_items(self) -> None:
        canvas = self._canvas
        assert canvas is not None

        # Border first, then the gradient fill inset on top, leaving a ring.
        for y, x0, x1 in _capsule_extents(self.width, self.height):
            self._border_items.append(canvas.create_line(x0, y, x1, y, fill="#000000"))
        for y, x0, x1 in _capsule_extents(self.width, self.height, inset=BORDER_PX):
            self._bg_items.append(canvas.create_line(x0, y, x1, y, fill="#000000"))

        centre_y = self.height / 2
        pad = 22
        self._flat_item = canvas.create_line(
            pad, centre_y, self.width - pad, centre_y, fill="#000000", width=2,
        )
        self._wave_item = canvas.create_polygon(
            [pad, centre_y, self.width - pad, centre_y],
            smooth=True, splinesteps=10, fill="#000000", outline="#000000",
        )
        self._pulse_item = canvas.create_line(
            pad, centre_y, pad + 40, centre_y, fill="#000000", width=4,
        )

    def _paint_background(self) -> None:
        """Vertical gradient, tinted slightly toward the current state colour."""
        canvas = self._canvas
        assert canvas is not None

        # Repainting ~70 scanlines is the expensive part of a frame, so it is
        # skipped while the colour is not actually moving. The key tracks the
        # tint itself, not the barely-moving blend of it: the rim below uses
        # the tint at full strength, so keying on the 12%-blended background
        # would let the rim lag a visible step behind the wave.
        tint = self._colour
        key = f"{tint[0]:.3f}{tint[1]:.3f}{tint[2]:.3f}"
        if key == self._last_bg_key:
            return
        self._last_bg_key = key

        top = _lerp_hsv(BG_TOP, tint, 0.12)
        bottom = _lerp_hsv(BG_BOTTOM, tint, 0.06)

        # A translucent capsule needs a brighter rim than a solid one, or the
        # silhouette dissolves into whatever is behind it.
        border = _hsv_to_hex((tint[0], min(tint[1] * 0.75, 1.0), min(tint[2] * 0.85 + 0.15, 1.0)))
        for item in self._border_items:
            canvas.itemconfigure(item, fill=border)

        rows = len(self._bg_items)
        for i, item in enumerate(self._bg_items):
            shade = _lerp_hsv(top, bottom, i / max(rows - 1, 1))
            canvas.itemconfigure(item, fill=_hsv_to_hex(shade))

    # --- drawing ------------------------------------------------------------

    def _draw(self) -> None:
        canvas = self._canvas
        assert canvas is not None
        self._paint_background()

        colour = _hsv_to_hex(self._colour)
        centre_y = self.height / 2
        pad = 22
        span = self.width - pad * 2

        if self.state == "transcribing":
            canvas.itemconfigure(self._wave_item, state="hidden")
            canvas.itemconfigure(self._flat_item, state="normal", fill=_hsv_to_hex(
                (self._colour[0], self._colour[1] * 0.45, self._colour[2] * 0.5)))
            head = (self._phase * 0.09) % 1.0
            segment = span * 0.28
            x0 = pad + head * (span + segment) - segment
            canvas.coords(
                self._pulse_item,
                max(x0, pad), centre_y, min(x0 + segment, pad + span), centre_y,
            )
            canvas.itemconfigure(self._pulse_item, state="normal", fill=colour)
            return

        canvas.itemconfigure(self._pulse_item, state="hidden")
        canvas.itemconfigure(self._flat_item, state="hidden")
        canvas.itemconfigure(self._wave_item, state="normal", fill=colour, outline=colour)

        step = span / (SAMPLES - 1)
        max_amp = (self.height / 2) - 12
        values = list(self._history)

        top: list[float] = []
        bottom: list[tuple[float, float]] = []
        for i, amp in enumerate(values):
            x = pad + i * step
            edge = math.sin(math.pi * (i / (SAMPLES - 1))) ** 0.7
            ripple = 0.70 + 0.30 * math.sin(self._phase + i * 0.38)
            a = max(amp * max_amp * edge * ripple, 1.1)
            top.append(x)
            top.append(centre_y - a)
            bottom.append((x, centre_y + a))

        points = top + [coord for xy in reversed(bottom) for coord in xy]
        canvas.coords(self._wave_item, *points)

    # --- state --------------------------------------------------------------

    def _read_level(self) -> float:
        if self.state != "recording":
            return 0.0
        try:
            return max(0.0, min(1.0, float(self.get_level())))
        except Exception:
            return 0.0

    def _target_colour(self) -> tuple[float, float, float]:
        if self.state == "transcribing":
            return STATE_COLOURS["busy"]
        if self.state != "recording":
            return STATE_COLOURS["idle"]

        # Hysteresis plus a short hang, so ordinary gaps between words do not
        # drop the colour back to quiet on every syllable.
        if self._level >= SPEECH_ON:
            self._speaking = True
            self._hang = SPEECH_HANG_FRAMES
        elif self._level <= SPEECH_OFF:
            if self._hang > 0:
                self._hang -= 1
            else:
                self._speaking = False
        return STATE_COLOURS["speaking" if self._speaking else "quiet"]

    # --- animation loop -----------------------------------------------------

    def _tick(self) -> None:
        if self._closing:
            if self._root is not None:
                self._root.destroy()
                self._root = None
            return

        assert self._root is not None
        self._phase += 0.30

        self._level += (self._read_level() - self._level) * LEVEL_EASE
        self._history.append(self._level)
        self._colour = _ease_colour(self._colour, self._target_colour())

        # Fade in once, then stay visible until the app exits.
        if self._alpha < self.opacity:
            self._alpha = min(self.opacity, self._alpha + 0.06)
            self._root.attributes("-alpha", self._alpha)

        self._draw()
        self._root.after(FRAME_MS, self._tick)

    def run(self) -> None:
        """Build the window and run the Tk loop. Must be the main thread."""
        self._build()
        assert self._root is not None
        self._root.after(FRAME_MS, self._tick)
        self._root.mainloop()
