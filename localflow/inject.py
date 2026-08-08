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

import contextlib
import ctypes
import time
from ctypes import wintypes

import pywintypes  # from pywin32
import win32clipboard  # from pywin32

user32 = ctypes.WinDLL("user32", use_last_error=True)

# pywintypes.error derives from Exception, NOT from OSError. Every clipboard
# helper below therefore converts it, so callers have exactly one exception type
# to handle. Skipping that conversion is not cosmetic: _paste_or_type catches
# OSError to fall back to typing, so an unconverted pywintypes.error would sail
# past the fallback, lose the transcription, and leave the dictated text sitting
# on the clipboard because the restore in paste_text's finally would also be
# skipped.
_WIN32_ERRORS = (pywintypes.error, OSError)

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_CONTROL = 0x11
VK_V = 0x56

CF_TEXT = 1
CF_OEMTEXT = 7
CF_LOCALE = 16
CF_UNICODETEXT = 13

# Text and its companions. Windows synthesises the other three from
# CF_UNICODETEXT, so putting that one back restores all four.
TEXT_FORMATS = frozenset({CF_TEXT, CF_OEMTEXT, CF_UNICODETEXT, CF_LOCALE})

# Bookkeeping that OLE-aware applications attach to an ordinary text copy.
# Word, Excel, Explorer, browsers and most .NET apps all add these, and they
# hold no user content: they describe the source object and are rebuilt on the
# next copy. Treating them as content is not a theoretical mistake. It made
# clipboard_is_restorable() return False for essentially every real text copy,
# so "auto" never took the paste path and always fell back to typing.
#
# These are registered formats, so their numeric ids differ per session and
# they have to be matched by name.
IGNORABLE_FORMAT_NAMES = frozenset(
    {
        "DataObject",
        "Ole Private Data",
        "Object Descriptor",
        "Link Source Descriptor",
        "Preferred DropEffect",
    }
)


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class _MOUSEINPUT(ctypes.Structure):
    """Never sent by this module, but it is the LARGEST member of the INPUT
    union and therefore determines sizeof(INPUT).

    Omitting it makes the union 24 bytes instead of 32, so sizeof(INPUT) comes
    out as 32 rather than the 40 Windows requires on x64, and SendInput then
    rejects every call with ERROR_INVALID_PARAMETER (87). Nothing is typed and
    there is no other symptom, which is why the size assertion below exists and
    why bench/test_inject.py injects into a real widget instead of a stub.
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
        except _WIN32_ERRORS as exc:
            last = exc
            time.sleep(delay)
    raise OSError(f"could not open clipboard: {last}")


def _clipboard_close() -> None:
    """Best effort. Never let a close failure mask the real error.

    These run in `finally` blocks, so an exception here would replace whatever
    actually went wrong with a confusing one from the cleanup path.
    """
    with contextlib.suppress(*_WIN32_ERRORS):
        win32clipboard.CloseClipboard()


def _format_name(fmt: int) -> str:
    """Name of a registered clipboard format, or "" for a standard one."""
    try:
        return win32clipboard.GetClipboardFormatName(fmt)
    except Exception:
        return ""


def clipboard_is_restorable() -> bool:
    """True when borrowing the clipboard would cost the user nothing.

    Borrowing means we can only put CF_UNICODETEXT back, so anything richer
    (an image, a file drop, formatted HTML) would be destroyed by pasting.
    This is the guard that stops that from happening.

    The rule is deliberately allow-list shaped: text plus known-harmless OLE
    bookkeeping passes, and any format we do not recognise fails, so an
    unfamiliar payload makes the caller type instead of paste. Erring toward
    the slower path is the right trade when the alternative is eating whatever
    the user had copied.
    """
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
        return all(
            f in TEXT_FORMATS or _format_name(f) in IGNORABLE_FORMAT_NAMES for f in formats
        )
    except _WIN32_ERRORS:
        # This function answers "is it safe to borrow the clipboard". If the
        # question cannot be answered, the safe answer is no. Raising instead
        # would break the contract callers rely on: inject() treats this as a
        # plain bool and has no handler for it.
        return False
    finally:
        _clipboard_close()


def _clipboard_get_text() -> str | None:
    _clipboard_open()
    try:
        if not win32clipboard.IsClipboardFormatAvailable(CF_UNICODETEXT):
            return None
        return win32clipboard.GetClipboardData(CF_UNICODETEXT)
    except _WIN32_ERRORS as exc:
        raise OSError(f"could not read clipboard: {exc}") from exc
    finally:
        _clipboard_close()


def _clipboard_set_text(text: str) -> None:
    _clipboard_open()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(CF_UNICODETEXT, text)
    except _WIN32_ERRORS as exc:
        raise OSError(f"could not write clipboard: {exc}") from exc
    finally:
        _clipboard_close()


def _clipboard_clear() -> None:
    _clipboard_open()
    try:
        win32clipboard.EmptyClipboard()
    except _WIN32_ERRORS as exc:
        raise OSError(f"could not clear clipboard: {exc}") from exc
    finally:
        _clipboard_close()


def paste_text(text: str, restore_delay: float = 0.35) -> None:
    """Set the clipboard, send Ctrl+V, then put the old clipboard back.

    restore_delay is a correctness knob rather than a cosmetic one. The target
    app reads the clipboard asynchronously when it processes WM_PASTE, so
    restoring too early makes it read the RESTORED value and the user gets
    their previous clipboard pasted instead of what they just dictated.
    Electron and other slow-pumping apps are the usual victims, so the default
    is deliberately generous.

    Raises OSError if the clipboard cannot be borrowed. Callers that have a
    typing path available should fall back to it rather than propagate.
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
        time.sleep(restore_delay)
    finally:
        try:
            if previous is not None:
                _clipboard_set_text(previous)
            else:
                # There was no text to put back, which means the clipboard was
                # empty or held something non-text. Either way, leaving the
                # dictation sitting there would silently change what the user's
                # next Ctrl+V does, so clear it instead.
                _clipboard_clear()
        except OSError:
            pass


# --- dispatch --------------------------------------------------------------


def _paste_or_type(text: str) -> str:
    """Paste, falling back to typing if the clipboard cannot be borrowed.

    Another process can hold the clipboard lock for longer than the retries in
    _clipboard_open will wait. Losing a transcription to that would be far
    worse than taking the slower path, so this degrades instead of raising.
    """
    try:
        paste_text(text)
        return "paste"
    except OSError as exc:
        print(f"\n[inject] clipboard unavailable ({exc}); typing instead")
        type_text(text)
        return "type"


def inject(text: str, method: str = "auto", paste_threshold: int = 120) -> str:
    """Insert text at the cursor. Returns the strategy actually used."""
    if not text:
        return "noop"

    if method == "type":
        type_text(text)
        return "type"
    if method == "paste":
        return _paste_or_type(text)

    # auto
    if len(text) >= paste_threshold and clipboard_is_restorable():
        return _paste_or_type(text)
    type_text(text)
    return "type"
