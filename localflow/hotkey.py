"""Global push-to-talk hotkey via a WH_KEYBOARD_LL hook.

Implemented directly on ctypes rather than the `keyboard` package, which is
effectively unmaintained and known to drop events. This needs no administrator
rights.

Two things matter for correctness:

  * The hook procedure runs on the thread that installed it and blocks the
    entire system's input queue while it executes. If it is slow, Windows
    silently unhooks it. So the callbacks here only ever flip a flag or hand
    work to another thread. Never transcribe inside a callback.

  * Events we synthesise ourselves (inject.py sends Ctrl+V) come back through
    this hook flagged as injected. We ignore those, otherwise typing the
    transcription could retrigger the hotkey.
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes
from typing import Callable

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_QUIT = 0x0012
LLKHF_INJECTED = 0x10

HC_ACTION = 0

LRESULT = ctypes.c_ssize_t


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


HOOKPROC = ctypes.CFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

user32.SetWindowsHookExW.argtypes = (ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD)
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
user32.UnhookWindowsHookEx.restype = wintypes.BOOL
user32.GetMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT)
user32.GetMessageW.restype = wintypes.BOOL
user32.PostThreadMessageW.argtypes = (wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
user32.PostThreadMessageW.restype = wintypes.BOOL


# Common virtual-key codes, for settings.json readability.
VK_NAMES = {
    0xA2: "Left Ctrl",
    0xA3: "Right Ctrl",
    0xA0: "Left Shift",
    0xA1: "Right Shift",
    0xA4: "Left Alt",
    0xA5: "Right Alt",
    0x14: "Caps Lock",
    0x91: "Scroll Lock",
    0x7A: "F11",
    0x7B: "F12",
}


def key_name(vk: int) -> str:
    return VK_NAMES.get(vk, f"VK 0x{vk:02X}")


class HotkeyListener:
    """Calls on_press when the key goes down and on_release when it comes up.

    Key repeat is collapsed, so holding the key fires on_press exactly once.
    """

    def __init__(
        self,
        vk: int,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        suppress: bool = True,
        allow_injected: bool = False,
    ):
        self.vk = vk
        self.on_press = on_press
        self.on_release = on_release
        self.suppress = suppress
        # Only the self-test sets this. In normal use, synthesised events must
        # be ignored so that injecting the transcription cannot retrigger the
        # hotkey.
        self.allow_injected = allow_injected
        self.events_seen = 0

        self._hook = None
        self._thread_id: int | None = None
        self._down = False
        self._running = threading.Event()
        # Held so CPython does not garbage-collect the trampoline while
        # Windows still holds a pointer to it.
        self._proc = HOOKPROC(self._hook_proc)

    def _hook_proc(self, ncode: int, wparam: int, lparam: int) -> int:
        if ncode != HC_ACTION:
            return user32.CallNextHookEx(self._hook, ncode, wparam, lparam)

        kb = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
        self.events_seen += 1

        # Ignore anything we synthesised ourselves.
        if (kb.flags & LLKHF_INJECTED) and not self.allow_injected:
            return user32.CallNextHookEx(self._hook, ncode, wparam, lparam)

        if kb.vkCode == self.vk:
            if wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                if not self._down:
                    self._down = True
                    self._safe(self.on_press)
                if self.suppress:
                    return 1
            elif wparam in (WM_KEYUP, WM_SYSKEYUP):
                if self._down:
                    self._down = False
                    self._safe(self.on_release)
                if self.suppress:
                    return 1

        return user32.CallNextHookEx(self._hook, ncode, wparam, lparam)

    @staticmethod
    def _safe(fn: Callable[[], None]) -> None:
        # An exception escaping into the Windows hook chain would be fatal, so
        # callbacks are contained here.
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[hotkey] callback error: {exc!r}")

    def run(self) -> None:
        """Install the hook and pump messages. Blocks until stop() is called."""
        self._thread_id = kernel32.GetCurrentThreadId()
        self._hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, None, 0)
        if not self._hook:
            raise OSError(f"SetWindowsHookExW failed (err {ctypes.get_last_error()})")

        self._running.set()
        msg = wintypes.MSG()
        try:
            while True:
                ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret in (0, -1):  # WM_QUIT, or error
                    break
        finally:
            self._running.clear()
            user32.UnhookWindowsHookEx(self._hook)
            self._hook = None

    def stop(self) -> None:
        if self._thread_id is not None:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)

    @property
    def running(self) -> bool:
        return self._running.is_set()
