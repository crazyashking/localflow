"""Put text into whatever window currently has focus.

Two strategies, because neither works everywhere:

  type  - SendInput with KEYEVENTF_UNICODE. Touches nothing global, works in
          terminals and most native apps. Cost grows with text length.
  paste - Put the text on the clipboard and send Ctrl+V. Constant time
          regardless of length, and the only reliable option in some Electron
          apps, but it has to borrow the clipboard.

"auto" types short text and pastes long text. Critically, if the clipboard
currently holds something we cannot faithfully restore (an image, files), auto
refuses to paste and types instead, so dictating never destroys whatever you
had copied.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

import win32clipboard  # from pywin32

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_CONTROL = 0x11
VK_V = 0x56
CF_UNICODETEXT = 13


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class _MOUSEINPUT(ctypes.Structure):
    """Not used directly, but it is the LARGEST member of the INPUT union and
    therefore determines sizeof(INPUT).

    Omitting it makes the union 24 bytes instead of 32, so sizeof(INPUT) comes
    out as 32 rather than 40 on x64, and SendInput rejects every call with
    ERROR_INVALID_PARAMETER (87). Nothing gets typed, with no other symptom.
    """

    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTunion)]


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT

# x64: 40. Guard against silently regressing the layout above.
_EXPECTED_INPUT_SIZE = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
if ctypes.sizeof(_INPUT) != _EXPECTED_INPUT_SIZE:
    raise RuntimeError(
        f"INPUT struct is {ctypes.sizeof(_INPUT)} bytes, expected "
        f"{_EXPECTED_INPUT_SIZE}. SendInput would reject every call."
    )


def _key_event(vk: int = 0, scan: int = 0, flags: int = 0) -> _INPUT:
    return _INPUT(
        type=INPUT_KEYBOARD,
        u=_INPUTunion(ki=_KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=None)),
    )


def _send(events: list[_INPUT]) -> int:
    if not events:
        return 0
    n = len(events)
    arr = (_INPUT * n)(*events)
    sent = user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
    if sent != n:
        raise OSError(f"SendInput sent {sent}/{n} events (err {ctypes.get_last_error()})")
    return sent


# --- strategy: simulated typing -----------------------------------------


def type_text(text: str, chunk: int = 200) -> None:
    """Send text as Unicode key events. Does not touch the clipboard.

    Sent in chunks because a single very large SendInput array can be dropped
    by the OS, and chunking gives the target app time to keep up.
    """
    if not text:
        return
    events: list[_INPUT] = []
    for ch in text:
        code = ord(ch)
        # Characters outside the BMP need a surrogate pair, one event each.
        if code > 0xFFFF:
            code -= 0x10000
            units = [0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF)]
        else:
            units = [code]
        for unit in units:
            events.append(_key_event(scan=unit, flags=KEYEVENTF_UNICODE))
            events.append(_key_event(scan=unit, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))

    for i in range(0, len(events), chunk):
        _send(events[i : i + chunk])
        time.sleep(0.001)


# --- strategy: clipboard paste -------------------------------------------


def _clipboard_open(retries: int = 10, delay: float = 0.02):
    """The clipboard is a shared global lock; other apps hold it briefly."""
    last: Exception | None = None
    for _ in range(retries):
        try:
            win32clipboard.OpenClipboard()
            return
        except Exception as exc:  # pywintypes.error
            last = exc
            time.sleep(delay)
    raise OSError(f"could not open clipboard: {last}")


def clipboard_is_restorable() -> bool:
    """True if the clipboard is empty or holds only text we can put back."""
    try:
        _clipboard_open()
    except OSError:
        return False
    try:
        formats: list[int] = []
        fmt = 0
        while True:
            fmt = win32clipboard.EnumClipboardFormats(fmt)
            if fmt == 0:
                break
            formats.append(fmt)
        if not formats:
            return True
        # Text-only formats are the ones we can faithfully save and restore.
        # 1=CF_TEXT, 7=CF_OEMTEXT, 13=CF_UNICODETEXT, 16=CF_LOCALE.
        return all(f in (1, 7, 13, 16) for f in formats)
    finally:
        win32clipboard.CloseClipboard()


def _clipboard_get_text() -> str | None:
    _clipboard_open()
    try:
        if not win32clipboard.IsClipboardFormatAvailable(CF_UNICODETEXT):
            return None
        return win32clipboard.GetClipboardData(CF_UNICODETEXT)
    finally:
        win32clipboard.CloseClipboard()


def _clipboard_set_text(text: str) -> None:
    _clipboard_open()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(CF_UNICODETEXT, text)
    finally:
        win32clipboard.CloseClipboard()


def paste_text(text: str, restore_delay: float = 0.35) -> None:
    """Set the clipboard, send Ctrl+V, then put the old clipboard back.

    restore_delay is a real correctness knob, not cosmetic. The target app
    reads the clipboard asynchronously when it processes WM_PASTE. Restore too
    early and the app reads the RESTORED value, so the user gets their previous
    clipboard pasted instead of what they just dictated. Electron and other
    slow-pumping apps are the usual victims, so the default is deliberately
    generous.
    """
    if not text:
        return
    previous = _clipboard_get_text()
    _clipboard_set_text(text)
    try:
        _send(
            [
                _key_event(vk=VK_CONTROL),
                _key_event(vk=VK_V),
                _key_event(vk=VK_V, flags=KEYEVENTF_KEYUP),
                _key_event(vk=VK_CONTROL, flags=KEYEVENTF_KEYUP),
            ]
        )
        # The target app reads the clipboard asynchronously, so restoring
        # immediately would race it and paste the old contents instead.
        time.sleep(restore_delay)
    finally:
        if previous is not None:
            try:
                _clipboard_set_text(previous)
            except OSError:
                pass


# --- dispatch --------------------------------------------------------------


def inject(text: str, method: str = "auto", paste_threshold: int = 120) -> str:
    """Insert text at the cursor. Returns the strategy actually used."""
    if not text:
        return "noop"

    if method == "type":
        type_text(text)
        return "type"
    if method == "paste":
        paste_text(text)
        return "paste"

    # auto
    if len(text) >= paste_threshold and clipboard_is_restorable():
        paste_text(text)
        return "paste"
    type_text(text)
    return "type"
