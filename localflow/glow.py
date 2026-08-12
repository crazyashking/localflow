"""Screen edge glow.

A band of light around the rim of the screen that reports what LocalFlow is
doing: dim while it waits for you to speak, thickening and brightening with your
voice, then a slow warm pulse while the model decodes.

It runs alongside the capsule in overlay.py. The capsule is exact, small, and
always on screen. The glow is peripheral and only appears during an utterance,
so you get feedback while looking at whatever you are dictating into.

Four things here are load-bearing.

**Per-pixel alpha.** A soft falloff needs a real alpha value at every pixel,
which Tk cannot do: it only offers whole-window opacity and a colour key, and a
colour key gives hard edges. So this is a raw Win32 layered window pushed with
UpdateLayeredWindow, and the pixels are built in numpy. Both were already
dependencies, so the glow adds nothing to install.

**Four windows around the rim.** The lit region is a frame. Covering the whole
screen would mean handing the compositor a full-resolution bitmap every frame
for a picture that is empty in the middle. Four edge bands carry about a third
of those pixels. They tile the frame exactly, with no overlap and no seam,
because each one measures distance to the nearest screen edge with the same
function. See _distance_field.

**Colour comes from a lookup table.** Alpha and hue depend only on that
distance, so each frame fills a table of a few hundred entries and gathers the
whole image out of it in one indexing pass. The alternative is a stack of
full-resolution float operations per frame, which at 2560x1440 does not fit in a
33ms budget.

**It owns its own thread.** The windows are pure Win32 and never take input, so
they do not need Tk. Running the render loop separately means the glow works
whether or not the capsule is enabled, and its frame rate does not compete with
Tk's.

State colours, and the speech hysteresis that picks between them, are imported
from overlay.py rather than restated. The green rule documented above
STATE_COLOURS binds this module too, and a second copy of those numbers would
drift out of step with the test that enforces it.
"""

from __future__ import annotations

import colorsys
import contextlib
import ctypes
import math
import threading
import time
from collections.abc import Callable
from ctypes import wintypes

import numpy as np

from .overlay import SPEECH_HANG_FRAMES, SPEECH_OFF, SPEECH_ON, STATE_COLOURS

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)

WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020       # clicks fall through to the app underneath
WS_EX_TOPMOST = 0x00000008
WS_EX_NOACTIVATE = 0x08000000        # never takes focus from what you are typing in
WS_EX_TOOLWINDOW = 0x00000080        # keeps it out of Alt+Tab and the taskbar

SW_HIDE = 0
SW_SHOWNOACTIVATE = 4

ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01
BI_RGB = 0
DIB_RGB_COLORS = 0

PM_REMOVE = 0x0001

SM_CXSCREEN = 0
SM_CYSCREEN = 1


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", ctypes.c_byte),
        ("BlendFlags", ctypes.c_byte),
        ("SourceConstantAlpha", ctypes.c_byte),
        ("AlphaFormat", ctypes.c_byte),
    ]


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt_x", wintypes.LONG),
        ("pt_y", wintypes.LONG),
    ]


# Every one of these needs an explicit signature. Without argtypes ctypes
# guesses a 32-bit int for an integer argument, and an LPARAM or an HWND on
# 64-bit Windows routinely exceeds that: the call then dies inside a window
# procedure, where the exception is printed and swallowed rather than raised.
user32.DefWindowProcW.argtypes = (
    wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
)
user32.DefWindowProcW.restype = LRESULT
user32.PeekMessageW.argtypes = (
    ctypes.POINTER(MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT,
)
user32.PeekMessageW.restype = wintypes.BOOL
user32.TranslateMessage.argtypes = (ctypes.POINTER(MSG),)
user32.DispatchMessageW.argtypes = (ctypes.POINTER(MSG),)
user32.DispatchMessageW.restype = LRESULT
user32.RegisterClassW.argtypes = (ctypes.POINTER(WNDCLASSW),)
user32.RegisterClassW.restype = wintypes.ATOM
user32.EnumDisplayMonitors.argtypes = (
    wintypes.HDC, ctypes.c_void_p, ctypes.c_void_p, wintypes.LPARAM,
)
user32.EnumDisplayMonitors.restype = wintypes.BOOL
user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
user32.GetSystemMetrics.restype = ctypes.c_int
kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
kernel32.GetModuleHandleW.restype = wintypes.HMODULE
user32.CreateWindowExW.argtypes = (
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
)
user32.CreateWindowExW.restype = wintypes.HWND
user32.DestroyWindow.argtypes = (wintypes.HWND,)
user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
user32.GetDC.argtypes = (wintypes.HWND,)
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = (wintypes.HWND, wintypes.HDC)
user32.UpdateLayeredWindow.argtypes = (
    wintypes.HWND, wintypes.HDC, ctypes.c_void_p, ctypes.c_void_p,
    wintypes.HDC, ctypes.c_void_p, wintypes.DWORD,
    ctypes.POINTER(BLENDFUNCTION), wintypes.DWORD,
)
user32.UpdateLayeredWindow.restype = wintypes.BOOL
gdi32.CreateCompatibleDC.argtypes = (wintypes.HDC,)
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateDIBSection.argtypes = (
    wintypes.HDC, ctypes.POINTER(BITMAPINFO), wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD,
)
gdi32.CreateDIBSection.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = (wintypes.HDC, wintypes.HGDIOBJ)
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.DeleteObject.argtypes = (wintypes.HGDIOBJ,)
gdi32.DeleteDC.argtypes = (wintypes.HDC,)
gdi32.StretchBlt.argtypes = (
    wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.DWORD,
)
gdi32.StretchBlt.restype = wintypes.BOOL
gdi32.SetStretchBltMode.argtypes = (wintypes.HDC, ctypes.c_int)

SRCCOPY = 0x00CC0020
# Smooth interpolation on the upscale. COLORONCOLOR drops pixels instead, which
# turns the wash into visible blocks at this scale factor.
STRETCH_HALFTONE = 4


def enable_dpi_awareness() -> str:
    """Tell Windows this process draws in real pixels. Call before any window.

    Without it Windows renders the app at a virtual resolution and bitmap
    stretches the result, which turns a soft glow into a blurry one and puts the
    band edges on fractional pixels. Harmless at 100% display scaling, which is
    why it can go unnoticed until the app meets a 4K laptop panel.

    Three APIs, newest first, because the good one is Windows 10 1703 and later.
    Returns which one took, for the startup banner.
    """
    try:
        # -4 is DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2.
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per-monitor v2"
    except AttributeError:
        pass
    try:
        if ctypes.WinDLL("shcore").SetProcessDpiAwareness(2) == 0:
            return "per-monitor"
    except OSError:
        pass
    try:
        if user32.SetProcessDPIAware():
            return "system"
    except AttributeError:
        pass
    return "none"


def monitors() -> list[tuple[int, int, int, int]]:
    """Every display as (left, top, width, height), primary first.

    Queried at runtime and never cached across a resolution change. A size
    fixed at startup looks hairline on 4K and bloated on 1080p, and survives
    every monitor being unplugged.
    """
    found: list[tuple[int, int, int, int]] = []
    proc = ctypes.WINFUNCTYPE(
        ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
        ctypes.POINTER(wintypes.RECT), wintypes.LPARAM,
    )

    def _collect(_hmon, _hdc, rect, _param):
        r = rect.contents
        found.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
        return 1

    user32.EnumDisplayMonitors(0, 0, proc(_collect), 0)
    if not found:
        return [(0, 0, user32.GetSystemMetrics(SM_CXSCREEN),
                 user32.GetSystemMetrics(SM_CYSCREEN))]
    # The primary monitor is the one whose top left corner is the origin. It is
    # not reliably first out of EnumDisplayMonitors.
    found.sort(key=lambda m: (m[0] != 0 or m[1] != 0, m[0], m[1]))
    return found


# --- look ------------------------------------------------------------------
#
# The band is two lit shapes added together: a broad soft body that fades away
# inward, and a tight core right against the screen edge. The body alone reads
# as fog with no edge to it, and the core alone reads as a hard neon outline.
# Together they give the bright rim and long falloff that makes this read as
# light rather than as a drawn border.
BODY_GAMMA = 2.4          # how fast the broad part falls off. Higher is tighter
BODY_WEIGHT = 0.52
CORE_WEIGHT = 0.70
CORE_FALLOFF = 11.0       # e-folds across the band, so the core is its first 9%

# Thickness as a fraction of the monitor's SHORTER side, so the glow keeps its
# proportions on any display. On a 2560x1440 that is 158px.
THICKNESS_FRAC = 0.11

# Thickness at rest as a fraction of full, growing to 1.0 at full voice. Never
# zero: an armed session has to be visible before you say anything.
THICK_REST = 0.46

# 60 rather than 30. The band is a large, slow-moving shape in peripheral
# vision, which is where frame rate shows up worst: at 30fps the travelling
# ripple visibly steps. A frame costs about 4ms on one 1440p screen, so 60fps
# still leaves three quarters of the budget free, and the glow is only on screen
# during an utterance.
FPS = 60
FRAME_S = 1.0 / FPS

# Envelope follower on the microphone level, the same shape as an audio
# compressor. Raw RMS strobes: it moves faster than the eye reads as motion, so
# the band flickers instead of breathing. Fast attack keeps the response to a
# consonant honest, slow release carries it through the gaps between words.
ATTACK_TAU = 0.045
RELEASE_TAU = 0.220

FADE_IN_S = 0.22
FADE_OUT_S = 0.38

# --- staying alive across a sleep -------------------------------------------
#
# A layered window's device context and bitmap belong to a display
# configuration. Sleeping the machine, unplugging a monitor or changing
# resolution invalidates all of it, and UpdateLayeredWindow then fails quietly
# forever after. These are the three ways that gets noticed.
#
# A frame takes 16ms, so a gap of seconds means the process was frozen rather
# than merely busy. That is the strongest signal a resume happened, and it needs
# no window and no message hook to detect.
RESUME_GAP_S = 5.0
GEOMETRY_CHECK_S = 2.0
FAILED_PUSH_LIMIT = 8

# Every easing below is a time constant in seconds, converted to a per-frame
# coefficient with _coef. Easing by a fixed fraction per frame instead ties the
# speed of the animation to the frame rate, so changing FPS silently retimes
# every transition in the module.
COLOUR_TAU = 0.13         # hue travel between states
BRIGHT_TAU = 0.085        # brightness following the voice
RIPPLE_TAU = 0.30         # how fast the travelling ripple arrives and leaves

# How hard the band reacts to loudness. A plain multiply clips: past the point
# where it saturates, speaking louder changes nothing and the top of the range
# goes flat. This curve keeps responding all the way to full scale.
RESPONSE_KNEE = 2.4

ARMED_LEVEL = 0.34        # brightness while armed and silent
SPEAK_FLOOR = 0.46        # brightness at the first voiced frame
THINK_BASE = 0.40         # decoding pulse, centre and swing
THINK_SWING = 0.17
THINK_HZ = 0.55

# --- the rim sweep ----------------------------------------------------------
#
# The "lap" and "split" styles, which travel the rim instead of crossing the
# screen. A crest starts at one point on the perimeter and runs around it,
# trailing off behind. The default is the wash below; these are kept because
# they are cheaper and some people prefer the rim staying uncovered.
#
# The whole effect is a gain along the perimeter, so it has to be free. Scaling
# every pixel by a per-column value costs about 10ms a frame at 1440p, which
# does not fit. Instead the gain becomes a second axis on the lookup table: the
# table holds colour for every (gain, distance) pair, and a column's gain is
# folded into its index. That keeps a frame at one indexing pass, the same as
# with no sweep at all. SWEEP_GAIN_STEPS is how finely the gain is quantised,
# and 64 is below what the eye resolves across a band this size.
# Matches glow_sweep_seconds in config.py, which is what the app actually
# passes. This default only applies to a BorderGlow built directly.
SWEEP_S = 0.7
SWEEP_GAIN_STEPS = 64
SWEEP_BOOST = 0.65        # how much brighter the crest is than the settled band

# Both as a fraction of half the perimeter, so they scale with the display.
# LEAD is the soft edge ahead of the wavefront, which stops the front reading as
# a hard line. TRAIL is how far the crest reaches back, which is the glassy part.
SWEEP_LEAD_FRAC = 0.022
SWEEP_TRAIL_FRAC = 0.070

# The front travels past the meeting point by this many trail lengths, so the
# crest has decayed to nothing by the time the sweep finishes. Stopping it
# exactly at the far side leaves a permanent bright spot there.
SWEEP_OVERRUN = 3.0

# Where the light starts. An edge name starts it at that edge's midpoint, which
# spreads symmetrically from the middle of one side. A corner starts it there
# instead, so the light runs away down two edges at once and converges on the
# opposite corner, which covers the whole rim without ever looking centred.
SWEEP_ORIGINS = (
    "top-left", "top-right", "bottom-right", "bottom-left",
    "right", "left", "top", "bottom",
)

# "lap" sends the light one way round the whole screen and back to where it
# started. "split" sends it both ways at once to meet on the far side, which
# covers the rim in half the time and reads as opening outward from the origin.
SWEEP_STYLES = ("wash", "lap", "split")

# --- the wash ---------------------------------------------------------------
#
# The default intro, and the one taken from a screen recording of Siri. Light
# enters from one side, floods the WHOLE screen as it crosses, then recedes into
# the rim and stays there. The flood is the part that matters: a wavefront that
# only ever travels the rim reads as light crawling around the corners, which is
# what every edge-only version of this looked like.
#
# The rim bands cannot draw the middle of the screen, so the wash gets its own
# full-screen window, shown only while it runs. The two never overlap, because
# two layered windows over the same pixel composite twice and the rim would
# double in brightness for the length of the intro.
#
# Rendered at a quarter size in each axis and stretched up by GDI. A full 1440p
# buffer costs about 15ms a frame, which does not fit in 16.7ms. The wash is
# soft and has no fine detail, so the downscale is invisible, and interpolating
# premultiplied alpha is mathematically the right thing to do.
FLOOD_SCALE = 4
FLOOD_STEPS = 48          # quantisation of the wash gain, as with the sweep

# How far the leading edge is spread out, as a fraction of the crossing. A hard
# edge reads as a wipe rather than as light arriving.
FLOOD_EDGE = 0.42

# Peak brightness of the flood over the screen's interior, and the shape of its
# rise and fall across the intro. It has to reach zero by the end, because that
# is the moment the rim bands take over and any residue would show as a step.
FLOOD_PEAK = 0.42
FLOOD_SHAPE = 1.4

# After an utterance is transcribed, the glow holds at a low "still listening"
# level for this long before it leaves. Without it the amber decoding pulse
# vanishes the instant the text lands, which reads as the app switching off
# rather than as it going back to waiting.
LINGER_S = 2.0
LINGER_LEVEL = 0.22

# Slow travelling ripple in the band's thickness, strongest while you speak.
# This is what stops the glow reading as a static picture frame. Two waves at
# unrelated speeds, so the pattern does not visibly repeat.
SHIMMER_AMP = 0.20
SHIMMER_W1 = 2.3
SHIMMER_W2 = 3.7
SHIMMER_S1 = 0.55
SHIMMER_S2 = -0.31

def _coef(tau: float) -> float:
    """Per-frame easing fraction for a time constant, at the current frame rate."""
    return 1.0 - math.exp(-FRAME_S / tau)


def _smoothstep(x: float) -> float:
    """Ease a 0-to-1 ramp in and out of rest.

    A linear fade arrives at full brightness at full speed and stops dead, and
    the eye reads that corner as a snap even when the fade itself is slow
    enough. This flattens both ends of the ramp.
    """
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _response(level: float, knee: float = RESPONSE_KNEE) -> float:
    """Loudness to visual response, saturating smoothly instead of clipping."""
    return (1.0 - math.exp(-knee * level)) / (1.0 - math.exp(-knee))


_CLASS_NAME = "LocalFlowGlowBand"
_class_registered = False
_wndproc_ref: WNDPROC | None = None


def _register_class() -> None:
    """Register the window class once per process."""
    global _class_registered, _wndproc_ref
    if _class_registered:
        return
    # Held in a module global for the life of the process. If this reference is
    # dropped, ctypes frees the thunk and Windows calls into freed memory the
    # next time it dispatches a message to one of these windows.
    _wndproc_ref = WNDPROC(
        lambda hwnd, msg, wp, lp: user32.DefWindowProcW(hwnd, msg, wp, lp)
    )
    cls = WNDCLASSW()
    cls.style = 0
    cls.lpfnWndProc = _wndproc_ref
    cls.hInstance = kernel32.GetModuleHandleW(None)
    cls.lpszClassName = _CLASS_NAME
    if not user32.RegisterClassW(ctypes.byref(cls)):
        err = ctypes.get_last_error()
        # 1410 is ERROR_CLASS_ALREADY_EXISTS, which is fine and means a previous
        # Glow in this process already registered it.
        if err != 1410:
            raise OSError(f"RegisterClassW failed ({err})")
    _class_registered = True


def _distance_field(
    mon_w: int, mon_h: int, x0: int, y0: int, w: int, h: int, thickness: int
) -> np.ndarray:
    """Distance from every pixel of one band to the nearest edge of its screen.

    Measured in the monitor's own coordinates, which is what makes the four
    bands seamless. Each one is a window onto the same continuous field, so
    neighbouring bands already agree along the line where they meet and no
    corner blending is needed.
    """
    ys = np.arange(y0, y0 + h, dtype=np.float32)[:, None]
    xs = np.arange(x0, x0 + w, dtype=np.float32)[None, :]
    d = np.minimum(
        np.minimum(xs, mon_w - 1 - xs), np.minimum(ys, mon_h - 1 - ys)
    )
    return np.clip(d, 0.0, thickness - 1).astype(np.float32)


def _perimeter_coord(
    edge: str, w: int, h: int, thickness: int
) -> np.ndarray:
    """Where each slice of a band sits along the rim, walking clockwise.

    The wavefront has to cross a corner without a jump, and the four bands are
    four separate windows that know nothing about each other. Giving every band
    a position on one shared path around the screen is what makes the light
    appear to travel through the corners rather than restart at each one.

    Clockwise from the top left corner: across the top, down the right, back
    along the bottom, up the left.
    """
    if edge == "top":
        return np.arange(w, dtype=np.float32)
    if edge == "right":
        return w + np.arange(thickness, h - thickness, dtype=np.float32)
    if edge == "bottom":
        return w + h + (w - 1 - np.arange(w, dtype=np.float32))
    return 2 * w + h + (h - 1 - np.arange(thickness, h - thickness,
                                          dtype=np.float32))


def _origin_coord(where: str, w: int, h: int) -> float:
    """Perimeter position the light starts from.

    Corners are the four points where the clockwise walk turns, so starting at
    one sends the light down two edges at once.
    """
    return {
        "top-left": 0.0,
        "top-right": float(w),
        "bottom-right": float(w + h),
        "bottom-left": float(2 * w + h),
        "top": w / 2,
        "right": w + h / 2,
        "bottom": w + h + w / 2,
        "left": 2 * w + h + h / 2,
    }[where]


def _arc_distance(
    coord: np.ndarray, origin: float, perimeter: float, lap: bool = True
) -> np.ndarray:
    """How far the light has to travel from the origin to each position.

    Two shapes, and the choice is the whole character of the animation.

    `lap` measures one continuous clockwise walk, so the light leaves the origin
    in a single direction and chases all the way round the screen before
    arriving back where it started. Distances run to a full perimeter.

    Otherwise it is the shorter way round, so the light leaves in both
    directions at once and the two halves meet on the far side. Distances run to
    half a perimeter, and every point is lit in half the time.
    """
    if lap:
        return np.mod(coord - origin, perimeter).astype(np.float32)
    raw = np.abs(coord - origin)
    return np.minimum(raw, perimeter - raw).astype(np.float32)


def _make_dib(dc: int, w: int, h: int) -> tuple[int, ctypes.c_void_p, int]:
    """A 32-bit top-down bitmap selected into `dc`, plus a pointer to its bytes.

    Top-down (a negative height) so row 0 of the numpy view is the top row on
    screen. A bottom-up DIB draws the falloff upside down on the top and bottom
    bands, which looks correct on one edge and wrong on the other.
    """
    info = BITMAPINFO()
    info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    info.bmiHeader.biWidth = w
    info.bmiHeader.biHeight = -h
    info.bmiHeader.biPlanes = 1
    info.bmiHeader.biBitCount = 32
    info.bmiHeader.biCompression = BI_RGB
    bits = ctypes.c_void_p()
    bitmap = gdi32.CreateDIBSection(
        dc, ctypes.byref(info), DIB_RGB_COLORS, ctypes.byref(bits), None, 0)
    if not bitmap:
        raise OSError(f"CreateDIBSection failed ({ctypes.get_last_error()})")
    old = gdi32.SelectObject(dc, bitmap)
    return bitmap, bits, old


class _Band:
    """One edge of the frame: a layered window and the pixels behind it."""

    def __init__(
        self, mon: tuple[int, int, int, int], rect: tuple[int, int, int, int],
        thickness: int, horizontal: bool, arc: np.ndarray,
    ) -> None:
        # How far round the rim each slice of this band sits from where the
        # light starts. Fixed for the life of the window, so the sweep costs one
        # cheap pass over a 1D array per frame.
        self.arc = arc
        self._gain = np.empty_like(arc)
        self._offset = np.empty(arc.shape, dtype=np.uint16)
        mon_x, mon_y, mon_w, mon_h = mon
        x0, y0, self.w, self.h = rect
        self.horizontal = horizontal
        self.thickness = thickness
        self.dist = _distance_field(mon_w, mon_h, x0, y0, self.w, self.h, thickness)
        # The no-shimmer path indexes with this directly and skips a multiply
        # and a cast over every pixel in the band.
        self.dist_idx = self.dist.astype(np.uint16)
        self._scaled = np.empty_like(self.dist)
        self._idx = np.empty(self.dist.shape, dtype=np.uint16)

        self.hwnd = user32.CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
            | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
            _CLASS_NAME, None, WS_POPUP,
            mon_x + x0, mon_y + y0, self.w, self.h,
            None, None, kernel32.GetModuleHandleW(None), None,
        )
        if not self.hwnd:
            raise OSError(f"CreateWindowExW failed ({ctypes.get_last_error()})")

        screen_dc = user32.GetDC(None)
        self.dc = gdi32.CreateCompatibleDC(screen_dc)
        self.bitmap, bits, self.old_bitmap = _make_dib(self.dc, self.w, self.h)
        user32.ReleaseDC(None, screen_dc)

        # A numpy view straight onto the DIB's pixels, so a frame is written
        # once into the memory Windows will read, with no copy in between.
        raw = (ctypes.c_uint8 * (self.w * self.h * 4)).from_address(bits.value)
        self.pixels = np.ctypeslib.as_array(raw).reshape(self.h, self.w, 4)
        self.pixels[:] = 0

        self.pos = wintypes.POINT(mon_x + x0, mon_y + y0)
        self.size = wintypes.SIZE(self.w, self.h)
        self.src = wintypes.POINT(0, 0)

    def render(
        self, lut: np.ndarray, wave: np.ndarray | None,
        offset: np.ndarray | None,
    ) -> None:
        if wave is None and offset is None:
            idx = self.dist_idx
        else:
            if wave is None:
                self._idx[:] = self.dist_idx
            else:
                # Multiplying the distance rather than the alpha makes the
                # band's THICKNESS ripple along its length, which reads as
                # liquid. Scaling brightness instead just makes it twinkle.
                np.multiply(self.dist, wave[None, :] if self.horizontal
                            else wave[:, None], out=self._scaled)
                np.rint(self._scaled, out=self._scaled)
                self._idx[:] = self._scaled
            if offset is not None:
                # The sweep's gain is a whole row of the table, so applying it
                # is an add on the index rather than a multiply over every
                # pixel and every channel.
                self._idx += (offset[None, :] if self.horizontal
                              else offset[:, None])
            idx = self._idx
        # mode="clip" rather than bounds checking: the shimmer can push an index
        # past the end of the table, and a clamp is both correct here and
        # cheaper than raising.
        np.take(lut, idx, axis=0, mode="clip", out=self.pixels)

    def sweep_offset(
        self, front: float, lead: float, trail: float, thickness: int
    ) -> np.ndarray:
        """Row of the gain table each slice of this band should read from."""
        delta = front - self.arc
        # Soft leading edge: full ahead of the front by `lead`, so the wave has
        # no hard boundary. A step here reads as a drawn line sliding round the
        # screen rather than as light arriving.
        np.clip((delta + lead) / lead, 0.0, 1.0, out=self._gain)
        shaped = 3.0 - 2.0 * self._gain
        self._gain *= self._gain
        self._gain *= shaped
        crest = 1.0 + SWEEP_BOOST * np.exp(-np.maximum(delta, 0.0) / trail)
        self._gain *= crest
        self._gain *= (SWEEP_GAIN_STEPS - 1) / (1.0 + SWEEP_BOOST)
        np.clip(self._gain, 0, SWEEP_GAIN_STEPS - 1, out=self._gain)
        self._offset[:] = self._gain
        self._offset *= thickness
        return self._offset

    def push(self, screen_dc: int, blend: BLENDFUNCTION) -> bool:
        """Hand the frame to Windows. False means the window needs rebuilding.

        The result is checked rather than discarded because this is exactly what
        fails after the machine sleeps: the device contexts and the bitmaps
        behind them stop being valid, every call quietly returns false, and the
        glow is gone for the rest of the session with nothing logged.
        """
        return bool(user32.UpdateLayeredWindow(
            self.hwnd, screen_dc, ctypes.byref(self.pos), ctypes.byref(self.size),
            self.dc, ctypes.byref(self.src), 0, ctypes.byref(blend), ULW_ALPHA,
        ))

    def show(self, visible: bool) -> None:
        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE if visible else SW_HIDE)

    def destroy(self) -> None:
        gdi32.SelectObject(self.dc, self.old_bitmap)
        gdi32.DeleteObject(self.bitmap)
        gdi32.DeleteDC(self.dc)
        user32.DestroyWindow(self.hwnd)


def _wash_coord(direction: str, w: int, h: int) -> np.ndarray:
    """Normalised 0-to-1 position along the direction the light travels.

    0 is where the light enters, 1 is the far side. A corner gives a diagonal
    front, which is why this is computed as a full 2D field rather than a single
    row or column. It is only ever evaluated on the quarter-size buffer, so the
    2D form costs almost nothing.
    """
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    field = {
        "left": xs + 0 * ys,
        "right": (1.0 - xs) + 0 * ys,
        "top": ys + 0 * xs,
        "bottom": (1.0 - ys) + 0 * xs,
        "top-left": (xs + ys) / 2.0,
        "top-right": ((1.0 - xs) + ys) / 2.0,
        "bottom-left": (xs + (1.0 - ys)) / 2.0,
        "bottom-right": ((1.0 - xs) + (1.0 - ys)) / 2.0,
    }[direction]
    return np.ascontiguousarray(field, dtype=np.float32)


class _Flood:
    """Full-screen window for the intro wash, on one monitor.

    Draws into a quarter-size buffer that GDI stretches up, so the cost is a
    sixteenth of a full-resolution frame. Hidden whenever the wash is not
    running, which is nearly all the time.
    """

    def __init__(self, mon: tuple[int, int, int, int], thickness: int,
                 direction: str) -> None:
        mon_x, mon_y, w, h = mon
        self.thickness = thickness
        self.w, self.h = w, h
        self.sw = max(8, w // FLOOD_SCALE)
        self.sh = max(8, h // FLOOD_SCALE)

        # Distance to the nearest screen edge, in real pixels, measured on the
        # small buffer. Everything deeper than the band clips to the last row of
        # the table, where the rim profile is zero, so the interior is lit only
        # by the flood term and settles to nothing when the flood ends.
        ys = np.arange(self.sh, dtype=np.float32)[:, None] * (h / self.sh)
        xs = np.arange(self.sw, dtype=np.float32)[None, :] * (w / self.sw)
        dist = np.minimum(np.minimum(xs, w - 1 - xs), np.minimum(ys, h - 1 - ys))
        self.rim = np.clip(dist, 0, thickness - 1).astype(np.uint16)
        self.coord = _wash_coord(direction, self.sw, self.sh)
        self._gain = np.empty_like(self.coord)
        self._idx = np.empty(self.coord.shape, dtype=np.uint16)

        self.hwnd = user32.CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
            | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
            _CLASS_NAME, None, WS_POPUP, mon_x, mon_y, w, h,
            None, None, kernel32.GetModuleHandleW(None), None,
        )
        if not self.hwnd:
            raise OSError(f"CreateWindowExW failed ({ctypes.get_last_error()})")

        screen_dc = user32.GetDC(None)
        self.dc = gdi32.CreateCompatibleDC(screen_dc)       # full size, for ULW
        self.small_dc = gdi32.CreateCompatibleDC(screen_dc)  # what numpy fills
        self.bitmap, _, _ = _make_dib(self.dc, w, h)
        self.small_bitmap, bits, _ = _make_dib(self.small_dc, self.sw, self.sh)
        user32.ReleaseDC(None, screen_dc)
        gdi32.SetStretchBltMode(self.dc, STRETCH_HALFTONE)

        raw = (ctypes.c_uint8 * (self.sw * self.sh * 4)).from_address(bits.value)
        self.pixels = np.ctypeslib.as_array(raw).reshape(self.sh, self.sw, 4)
        self.pixels[:] = 0

        self.pos = wintypes.POINT(mon_x, mon_y)
        self.size = wintypes.SIZE(w, h)
        self.src = wintypes.POINT(0, 0)

    def render(self, lut: np.ndarray, front: float) -> None:
        # Everything behind the front is lit, with a soft leading edge ahead of
        # it. The front runs past 1.0 by the edge width so the far side finishes
        # fully lit rather than stopping partway up the ramp.
        np.subtract(front, self.coord, out=self._gain)
        self._gain /= FLOOD_EDGE
        np.clip(self._gain, 0.0, 1.0, out=self._gain)
        shaped = 3.0 - 2.0 * self._gain
        self._gain *= self._gain
        self._gain *= shaped
        self._gain *= FLOOD_STEPS - 1
        self._idx[:] = self._gain
        self._idx *= self.thickness
        self._idx += self.rim
        np.take(lut, self._idx, axis=0, mode="clip", out=self.pixels)

    def push(self, screen_dc: int, blend: BLENDFUNCTION) -> bool:
        gdi32.StretchBlt(self.dc, 0, 0, self.w, self.h,
                         self.small_dc, 0, 0, self.sw, self.sh, SRCCOPY)
        return bool(user32.UpdateLayeredWindow(
            self.hwnd, screen_dc, ctypes.byref(self.pos), ctypes.byref(self.size),
            self.dc, ctypes.byref(self.src), 0, ctypes.byref(blend), ULW_ALPHA,
        ))

    def show(self, visible: bool) -> None:
        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE if visible else SW_HIDE)

    def destroy(self) -> None:
        gdi32.DeleteObject(self.bitmap)
        gdi32.DeleteObject(self.small_bitmap)
        gdi32.DeleteDC(self.dc)
        gdi32.DeleteDC(self.small_dc)
        user32.DestroyWindow(self.hwnd)


class BorderGlow:
    """Edge glow for one or more monitors. Drive it with set_state().

    Takes the same state strings as WaveOverlay, so the app reports its status
    once and both overlays follow. Everything runs on an internal thread; start()
    and stop() are the whole interface besides set_state.
    """

    def __init__(
        self,
        get_level: Callable[[], float],
        *,
        opacity: float = 0.85,
        thickness_frac: float = THICKNESS_FRAC,
        which_monitors: str = "primary",
        sweep_from: str = "left",
        sweep_style: str = "wash",
        sweep_seconds: float = SWEEP_S,
        linger_seconds: float = LINGER_S,
        max_seconds: float = 600.0,
    ) -> None:
        self.get_level = get_level
        self.opacity = max(0.0, min(1.0, float(opacity)))
        self.thickness_frac = max(0.02, min(0.30, float(thickness_frac)))
        self.which_monitors = which_monitors
        self.sweep_from = (sweep_from if sweep_from in SWEEP_ORIGINS
                           else "left")
        self.sweep_style = (sweep_style if sweep_style in SWEEP_STYLES
                            else "wash")
        # Zero means no sweep at all, which is the old uniform fade in.
        self.sweep_seconds = max(0.0, float(sweep_seconds))
        # Zero means leave the instant the text lands.
        self.linger_seconds = max(0.0, float(linger_seconds))
        # Safety net only, tied to the real recording limit. A flat cap of a
        # few seconds would cut the glow off in the middle of a long dictation,
        # since max_seconds is 600 by default.
        self.session_cap = float(max_seconds) + 5.0

        self.state = "ready"
        self._bands: list[_Band] = []
        self._floods: list[_Flood] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._failure: str | None = None

        self._level = 0.0
        self._speaking = False
        self._hang = 0
        # Visibility and brightness are tracked apart on purpose. Folding them
        # into one number means the fade-in curve also bends the level response,
        # so the band appears to react to your voice while it is still arriving.
        self._show = 0.0        # 0 to 1, how far through the fade it is
        self._bright = 0.0      # what the voice is asking for, ignoring the fade
        self._ripple = 0.0      # travelling ripple, eased in rather than switched
        self._alpha = 0.0       # the product, which is what gets drawn
        self._sweep = 0.0       # 0 to 1 as the light travels round the rim
        self._settled_at = 0.0  # when the last utterance finished, for the linger
        # How far the light has to travel to cover everything: a whole
        # perimeter for a lap, half of one when it goes both ways at once.
        self._sweep_span = 1.0
        # Lead and trail stay keyed to half a perimeter in both styles, so the
        # crest is the same size on screen whichever way the light goes.
        self._half_perimeter = 1.0
        self._colour = STATE_COLOURS["quiet"]
        self._phase = 0.0
        self._session_started = 0.0
        self._waves: dict[int, np.ndarray] = {}
        self._axes: dict[int, np.ndarray] = {}

    # --- public ------------------------------------------------------------

    def set_state(self, state: str) -> None:
        if state == self.state:
            return
        if state == "recording" and self.state != "recording":
            self._session_started = time.perf_counter()
        # Only an utterance earns the linger. Reaching "ready" from startup, or
        # from an error, should not light the screen for two seconds.
        if state == "ready" and self.state in ("recording", "transcribing"):
            self._settled_at = time.perf_counter()
        elif state != "ready":
            self._settled_at = 0.0
        self.state = state

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="localflow-glow", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def failure(self) -> str | None:
        """Why the glow gave up, or None. Never raises into the app."""
        return self._failure

    # --- the loop ----------------------------------------------------------

    def _run(self) -> None:
        try:
            self._build()
        except Exception as exc:
            # A glow that cannot start must not take dictation down with it.
            self._failure = f"{type(exc).__name__}: {exc}"
            return
        try:
            self._loop()
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"
        finally:
            for window in (*self._bands, *self._floods):
                with contextlib.suppress(OSError):
                    window.destroy()
            self._bands = []
            self._floods = []

    def _geometry(self) -> tuple[tuple[int, int, int, int], ...]:
        """Every display's rectangle, for spotting a layout change."""
        try:
            return tuple(monitors())
        except OSError:
            # An empty result never equals a real layout, so a transient failure
            # here triggers one harmless rebuild instead of raising.
            return ()

    def _rebuild(self) -> None:
        """Throw the windows away and make them again.

        Called after the machine wakes or the displays change. Everything the
        band owns (window, device context, bitmap) is tied to a display
        configuration that no longer exists, and none of it can be repaired in
        place. Animation state is deliberately kept, so a rebuild in the middle
        of an utterance does not restart the sweep or lose the level.
        """
        for window in (*self._bands, *self._floods):
            # The handles are already stale by definition here, so a failure to
            # release one is expected and must not stop the rest being rebuilt.
            with contextlib.suppress(OSError):
                window.destroy()
        self._bands = []
        self._floods = []
        self._axes = {}
        self._waves = {}
        self._half_perimeter = 1.0
        self._sweep_span = 1.0
        self._build()

    def _build(self) -> None:
        _register_class()
        screens = monitors()
        if self.which_monitors != "all":
            screens = screens[:1]
        for mon in screens:
            _, _, w, h = mon
            thickness = max(8, int(round(min(w, h) * self.thickness_frac)))
            # Four rectangles that tile the rim exactly once. The top and bottom
            # run the full width and own the corners; the sides fill only the
            # gap between them, so no pixel is covered twice. Two layered
            # windows over the same pixel would composite twice and the corners
            # would read as four bright blobs.
            mid_h = max(0, h - 2 * thickness)
            rects = [
                ("top", (0, 0, w, thickness), True),
                ("bottom", (0, h - thickness, w, thickness), True),
            ]
            if mid_h > 0:
                rects += [
                    ("left", (0, thickness, thickness, mid_h), False),
                    ("right", (w - thickness, thickness, thickness, mid_h), False),
                ]
            perimeter = 2.0 * (w + h)
            origin = _origin_coord(self.sweep_from, w, h)
            lap = self.sweep_style == "lap"
            self._half_perimeter = max(self._half_perimeter, perimeter / 2.0)
            self._sweep_span = max(
                self._sweep_span, perimeter if lap else perimeter / 2.0)
            for edge, rect, horizontal in rects:
                arc = _arc_distance(
                    _perimeter_coord(edge, w, h, thickness), origin, perimeter,
                    lap=lap)
                self._bands.append(
                    _Band(mon, rect, thickness, horizontal, arc))
            if self.sweep_style == "wash" and self.sweep_seconds > 0.0:
                self._floods.append(_Flood(mon, thickness, self.sweep_from))
        # Thickness scales with each monitor, so a mixed pair of displays needs
        # a table per size. Sharing one would leave the smaller screen's band
        # cut off partway down the falloff, as a hard line rather than a fade.
        self._thicknesses = sorted({band.thickness for band in self._bands})
        for band in self._bands:
            length = band.w if band.horizontal else band.h
            if length not in self._axes:
                self._axes[length] = np.linspace(
                    0.0, 1.0, length, dtype=np.float32)

    def _loop(self) -> None:
        msg = MSG()
        screen_dc = user32.GetDC(None)
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        visible = False
        flooding = False
        next_frame = time.perf_counter()
        last_tick = time.perf_counter()
        last_geometry = self._geometry()
        next_geometry_check = time.perf_counter() + GEOMETRY_CHECK_S
        failed_pushes = 0
        try:
            while not self._stop.is_set():
                now = time.perf_counter()

                # Three ways the windows stop being valid, all of which used to
                # end with the glow silently gone until the app restarted.
                reason = ""
                if now - last_tick > RESUME_GAP_S:
                    # The loop sleeps 16ms at a time, so a gap this size means
                    # the process was frozen: the machine slept, or hibernated.
                    # Waking invalidates the device contexts underneath us.
                    reason = f"resumed after {now - last_tick:.0f}s asleep"
                elif now >= next_geometry_check:
                    next_geometry_check = now + GEOMETRY_CHECK_S
                    geometry = self._geometry()
                    if geometry != last_geometry:
                        # A monitor was plugged, unplugged, or the resolution
                        # changed. The bands are sized to the old screen.
                        reason = "display layout changed"
                        last_geometry = geometry
                elif failed_pushes > FAILED_PUSH_LIMIT:
                    reason = "the compositor stopped accepting frames"
                last_tick = now

                if reason:
                    user32.ReleaseDC(None, screen_dc)
                    self._rebuild()
                    screen_dc = user32.GetDC(None)
                    last_geometry = self._geometry()
                    failed_pushes = 0
                    visible = False
                    flooding = False
                    next_frame = time.perf_counter()
                    print(f"\n[glow] rebuilt: {reason}\n")
                    continue
                # Windows expects every thread that owns a window to drain its
                # queue. These windows get almost no messages, but a thread that
                # never pumps is one Windows can decide is hung.
                while user32.PeekMessageW(
                    ctypes.byref(msg), None, 0, 0, PM_REMOVE
                ):
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))

                self._advance()
                want = self._alpha > 0.004
                if want != visible and not flooding:
                    for band in self._bands:
                        band.show(want)
                    visible = want
                # The wash owns the screen while it runs, then hands over to the
                # rim. Showing both at once composites the rim twice and it
                # visibly doubles in brightness for the length of the intro.
                washing = want and self._washing()
                if washing != flooding:
                    for flood in self._floods:
                        flood.show(washing)
                    for band in self._bands:
                        band.show(want and not washing)
                    flooding = washing
                    visible = want

                if washing:
                    front = self._wash_front()
                    luts = {t: self._build_wash_lut(t) for t in self._thicknesses}
                    for flood in self._floods:
                        flood.render(luts[flood.thickness], front)
                        if not flood.push(screen_dc, blend):
                            failed_pushes += 1
                        elif failed_pushes:
                            failed_pushes = 0
                elif want:
                    front = self._sweep_front()
                    sweeping = front is not None
                    build = self._build_sweep_lut if sweeping else self._build_lut
                    luts = {t: build(t) for t in self._thicknesses}
                    wave = self._shimmer()
                    lead = self._half_perimeter * SWEEP_LEAD_FRAC
                    trail = self._half_perimeter * SWEEP_TRAIL_FRAC
                    for band in self._bands:
                        offset = band.sweep_offset(
                            front, lead, trail, band.thickness
                        ) if sweeping else None
                        band.render(
                            luts[band.thickness],
                            wave.get(band.w if band.horizontal else band.h),
                            offset,
                        )
                        if not band.push(screen_dc, blend):
                            failed_pushes += 1
                        elif failed_pushes:
                            failed_pushes = 0

                next_frame += FRAME_S
                delay = next_frame - time.perf_counter()
                if delay < -FRAME_S:
                    # Fell behind, so stop trying to catch up frame by frame.
                    next_frame = time.perf_counter()
                    delay = 0.0
                if delay > 0:
                    time.sleep(delay)
        finally:
            user32.ReleaseDC(None, screen_dc)

    # --- animation ---------------------------------------------------------

    def _advance(self) -> None:
        """One frame of state: level, colour, visibility, and brightness."""
        self._phase += FRAME_S

        raw = 0.0
        if self.state == "recording":
            try:
                raw = max(0.0, min(1.0, float(self.get_level())))
            except Exception:
                raw = 0.0
        tau = ATTACK_TAU if raw > self._level else RELEASE_TAU
        self._level += (raw - self._level) * _coef(tau)

        # Colour is read first: it is what decides whether this counts as
        # speaking, which the brightness and the ripple both then use.
        self._colour = _ease_hsv(self._colour, self._target_colour(),
                                 _coef(COLOUR_TAU))

        visible = self._visible()
        if visible and self.sweep_seconds > 0.0:
            # The sweep IS the fade in, so _show goes straight to full and the
            # travelling light does the reveal. Running both would fade the
            # whole rim up underneath the wave and hide the effect entirely.
            self._show = 1.0
            self._sweep = min(1.0, self._sweep + FRAME_S / self.sweep_seconds)
        else:
            step = FRAME_S / (FADE_IN_S if visible else FADE_OUT_S)
            self._show = max(0.0, min(1.0, self._show + (step if visible else -step)))
        # Reset only once it is fully gone, so the fade out stays uniform and
        # the next utterance sweeps again from the start.
        if not visible and self._show <= 0.0:
            self._sweep = 0.0

        self._bright += (self._target_brightness() - self._bright) * _coef(BRIGHT_TAU)

        want_ripple = 1.0 if (self._speaking and self.state == "recording") else 0.0
        self._ripple += (want_ripple - self._ripple) * _coef(RIPPLE_TAU)

        self._alpha = _smoothstep(self._show) * self._bright

    def _visible(self) -> bool:
        if self.state == "transcribing":
            return True
        if self.state == "ready":
            return bool(self._settled_at) and (
                time.perf_counter() - self._settled_at < self.linger_seconds)
        if self.state != "recording":
            return False
        if not self._session_started:
            return True
        # Nothing in-process should ever reach this. It exists so a glow cannot
        # outlive the utterance that started it and sit there lit.
        return time.perf_counter() - self._session_started <= self.session_cap

    def _washing(self) -> bool:
        """True while the full-screen intro owns the display."""
        return (self.sweep_style == "wash" and self.sweep_seconds > 0.0
                and self._sweep < 1.0 and bool(self._floods))

    def _wash_front(self) -> float:
        """Leading edge of the wash, in the same 0-to-1 space as the coordinate.

        It starts one edge-width short of the screen and finishes one past it,
        so the near side is not already lit on the first frame and the far side
        is fully lit on the last. Eased so the light is quick to cross and
        settles slowly, which is what the recording shows.
        """
        eased = 1.0 - (1.0 - min(1.0, self._sweep)) ** 2.2
        return eased * (1.0 + 2.0 * FLOOD_EDGE) - FLOOD_EDGE

    def _sweep_front(self) -> float | None:
        """How far round the rim the light has reached, or None once settled.

        None means the sweep is over and every pixel is at full gain, which lets
        the steady state skip the gain table entirely. That matters because the
        steady state is nearly all of an utterance and the sweep is the first
        half second of it.
        """
        if self.sweep_seconds <= 0.0 or self._sweep >= 1.0:
            return None
        # Fast off the mark and decelerating, the way light spreads rather than
        # the way an object travels. A linear front reads as a wipe.
        eased = 1.0 - (1.0 - self._sweep) ** 2.5
        trail = self._half_perimeter * SWEEP_TRAIL_FRAC
        return eased * (self._sweep_span + SWEEP_OVERRUN * trail)

    def _target_brightness(self) -> float:
        if self.state == "ready":
            return LINGER_LEVEL
        if self.state == "transcribing":
            return THINK_BASE + THINK_SWING * math.sin(
                self._phase * THINK_HZ * 2 * math.pi)
        if self._speaking and self.state == "recording":
            return SPEAK_FLOOR + (1.0 - SPEAK_FLOOR) * _response(self._level)
        return ARMED_LEVEL

    def _target_colour(self) -> tuple[float, float, float]:
        if self.state == "transcribing":
            return STATE_COLOURS["busy"]
        if self.state == "ready":
            # Back to the listening colour rather than holding amber, so the
            # linger says "waiting for you" instead of "still working".
            return STATE_COLOURS["quiet"]
        if self.state != "recording":
            return self._colour
        # Same hysteresis and hang as the capsule, so the two never disagree
        # about whether you are talking.
        if self._level >= SPEECH_ON:
            self._speaking = True
            self._hang = SPEECH_HANG_FRAMES
        elif self._level <= SPEECH_OFF:
            if self._hang > 0:
                self._hang -= 1
            else:
                self._speaking = False
        return STATE_COLOURS["speaking" if self._speaking else "quiet"]

    def _alpha_profile(self, thickness: int) -> np.ndarray:
        """Alpha at every distance from the screen edge, before any gain."""
        grow = THICK_REST + (1.0 - THICK_REST) * _response(self._level)
        span = max(2.0, thickness * grow)
        t = np.arange(thickness, dtype=np.float32) / span
        body = np.clip(1.0 - t, 0.0, 1.0) ** BODY_GAMMA
        core = np.exp(-t * CORE_FALLOFF)
        a = np.clip(body * BODY_WEIGHT + core * CORE_WEIGHT, 0.0, 1.0)
        a *= self._alpha * self.opacity
        a[t >= 1.0] = 0.0
        return a

    def _colour_table(self, a: np.ndarray, rows: int) -> np.ndarray:
        """Premultiplied BGRA for a flat array of alpha values.

        Premultiplied is what ULW_ALPHA expects: the colour channels are already
        scaled by alpha. Feeding it straight colour makes the band wash out
        toward white as it fades.
        """
        r, g, b = colorsys.hsv_to_rgb(*self._colour)
        lut = np.empty((rows, 4), dtype=np.uint8)
        lut[:, 0] = (a * b * 255.0).astype(np.uint8)
        lut[:, 1] = (a * g * 255.0).astype(np.uint8)
        lut[:, 2] = (a * r * 255.0).astype(np.uint8)
        lut[:, 3] = (a * 255.0).astype(np.uint8)
        return lut

    def _build_lut(self, thickness: int) -> np.ndarray:
        """Premultiplied BGRA for every distance from the edge, this frame."""
        return self._colour_table(self._alpha_profile(thickness), thickness)

    def _build_wash_lut(self, thickness: int) -> np.ndarray:
        """Table for the intro wash: rim falloff plus a flood over the interior.

        The flood term rises and falls across the intro and must reach exactly
        zero at the end, because that is the frame where the rim bands take
        over. Any residue left in the interior shows up as the whole screen
        blinking off.
        """
        rim = self._alpha_profile(thickness)
        p = min(1.0, max(0.0, self._sweep))
        flood = FLOOD_PEAK * math.sin(math.pi * p) ** FLOOD_SHAPE
        flood *= self._alpha * self.opacity
        gains = np.linspace(0.0, 1.0, FLOOD_STEPS, dtype=np.float32)
        table = np.clip((rim[None, :] + flood) * gains[:, None], 0.0, 1.0)
        return self._colour_table(table.reshape(-1), FLOOD_STEPS * thickness)

    def _build_sweep_lut(self, thickness: int) -> np.ndarray:
        """The same table with a gain axis, for while the light is travelling.

        Row g*thickness + d is the colour at distance d under gain step g, so a
        column applies its gain by adding g*thickness to its index. Built only
        during the sweep, which is half a second at the start of an utterance.
        """
        a = self._alpha_profile(thickness)
        gains = np.linspace(0.0, 1.0 + SWEEP_BOOST, SWEEP_GAIN_STEPS,
                            dtype=np.float32)
        table = np.clip(gains[:, None] * a[None, :], 0.0, 1.0)
        return self._colour_table(table.reshape(-1),
                                  SWEEP_GAIN_STEPS * thickness)

    def _shimmer(self) -> dict[int, np.ndarray]:
        """Thickness ripple per band length, or empty while nothing is moving.

        The ripple arrives and leaves on _ripple, which eases. Switching it on
        the speaking flag instead makes the whole band change shape in one
        frame the moment you stop talking, which is the single most obvious
        discontinuity the animation had.
        """
        amp = SHIMMER_AMP * self._ripple * _response(self._level)
        if amp < 0.002:
            return {}
        out: dict[int, np.ndarray] = {}
        for length, axis in self._axes.items():
            wave = self._waves.get(length)
            if wave is None:
                wave = np.empty_like(axis)
                self._waves[length] = wave
            np.sin((axis * SHIMMER_W1 + self._phase * SHIMMER_S1) * 2 * math.pi,
                   out=wave)
            wave *= 0.6
            wave += 0.4 * np.sin(
                (axis * SHIMMER_W2 + self._phase * SHIMMER_S2) * 2 * math.pi)
            wave *= amp
            wave += 1.0
            out[length] = wave
        return out


def _ease_hsv(
    a: tuple[float, float, float], b: tuple[float, float, float], t: float
) -> tuple[float, float, float]:
    """Blend two HSV colours the short way round the wheel.

    Blending in RGB instead passes through muddy grey between opposite hues.
    The green rule above STATE_COLOURS in overlay.py is what keeps the short way
    round from crossing green.
    """
    ah, asat, av = a
    bh, bsat, bv = b
    dh = bh - ah
    if dh > 0.5:
        dh -= 1.0
    elif dh < -0.5:
        dh += 1.0
    return ((ah + dh * t) % 1.0, asat + (bsat - asat) * t, av + (bv - av) * t)
