"""Verify injection never leaves the clipboard damaged.

Sends no keystrokes, so it is safe to run with anything focused. The Ctrl+V
burst and the typing path are both stubbed out; what is under test is the
save / set / restore logic that decides what the user's clipboard looks like
afterwards, plus the two guards that keep a transcription from being lost or
from clobbering something that cannot be put back.

This script refuses to run if your clipboard currently holds something it
cannot restore, because a clipboard-safety test that destroys your clipboard
would be worse than no test at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pywintypes
import win32clipboard

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import inject  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def put_text_with_ole_bookkeeping(text: str) -> None:
    """Copy text the way an OLE-aware application does.

    Word, Excel, Explorer, browsers and most .NET apps attach DataObject and
    Ole Private Data to an ordinary text copy. Treating those as user content
    made the restorable check reject nearly every real clipboard, so "auto"
    silently typed instead of pasting. This reproduces that exact shape.
    """
    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(inject.CF_UNICODETEXT, text)
        for name in ("DataObject", "Ole Private Data"):
            win32clipboard.SetClipboardData(
                win32clipboard.RegisterClipboardFormat(name), b"\x01\x00\x00\x00"
            )
    finally:
        win32clipboard.CloseClipboard()


def put_non_text_format() -> None:
    """Own the clipboard with a format that cannot be saved and restored.

    A registered private format stands in for the real cases (an image, a file
    drop). Faking the guard's answer instead would test nothing.
    """
    fmt = win32clipboard.RegisterClipboardFormat("LocalFlowTestBinary")
    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(fmt, b"\x00\x01\x02 not text")
    finally:
        win32clipboard.CloseClipboard()


print("clipboard safety checks")

if not inject.clipboard_is_restorable():
    print("\n  SKIPPED: your clipboard currently holds something this test cannot")
    print("  put back (an image or files). Copy some text, then run it again.")
    raise SystemExit(0)

entry_clipboard = inject._clipboard_get_text()
print(f"  (clipboard on entry: {entry_clipboard!r})")

# Neutralise the two things that would touch the outside world. paste_text is
# still exercised in full; only the Ctrl+V burst and the wait after it are
# replaced, so the save / set / restore sequence around them runs for real.
real_send, real_type = inject._send, inject.type_text
pasted: list[str] = []
typed: list[str] = []


def fake_send(events: list) -> int:
    """Stand in for the Ctrl+V burst, and record what was on the clipboard.

    Reading here is the point: it captures the clipboard at the exact moment
    the target app would read it, which is what the restore must not race.
    """
    pasted.append(inject._clipboard_get_text() or "")
    return len(events)


inject._send = fake_send
inject.type_text = lambda text, chunk=200: typed.append(text)

try:
    # 1. Basic round-trip through the helpers everything else depends on.
    inject._clipboard_set_text("sentinel-alpha")
    check("set/get round-trip", inject._clipboard_get_text() == "sentinel-alpha")

    tricky = "cafe naive 123 ok"
    inject._clipboard_set_text(tricky)
    check("unicode round-trip", inject._clipboard_get_text() == tricky)

    # 2. Text copied by a real application, OLE bookkeeping and all, must be
    #    treated as borrowable. Getting this wrong does not lose data, it
    #    quietly disables the paste path everywhere.
    put_text_with_ole_bookkeeping("copied from a real app")
    check("text with OLE bookkeeping is restorable", inject.clipboard_is_restorable() is True)
    check("and the text itself still reads back",
          inject._clipboard_get_text() == "copied from a real app")

    # 3. The real contract: paste_text must leave the prior text exactly as it
    #    found it, and the target must have seen the dictation, not the
    #    restored value. Restoring too early is a real bug that this catches.
    inject._clipboard_set_text("something the user copied")
    pasted.clear()
    inject.paste_text("dictated sentence", restore_delay=0.0)
    check("target saw the dictation", pasted == ["dictated sentence"], f"(saw {pasted})")
    check(
        "prior clipboard restored exactly",
        inject._clipboard_get_text() == "something the user copied",
        f"(now {inject._clipboard_get_text()!r})",
    )

    # 3. With nothing text-shaped to restore, the dictation must NOT be left
    #    behind. Leaving it there silently changes what the next Ctrl+V does.
    inject._clipboard_clear()
    inject.paste_text("dictated sentence", restore_delay=0.0)
    check(
        "dictation not left on an empty clipboard",
        inject._clipboard_get_text() is None,
        f"(now {inject._clipboard_get_text()!r})",
    )

    # 4. A clipboard holding a format we cannot rebuild must be reported
    #    unrestorable. This is the guard that protects images and file drops.
    put_non_text_format()
    check("non-text clipboard reported unrestorable", inject.clipboard_is_restorable() is False)

    # 4b. The boundary of the allow-list. Rich formats carry real user content
    #     that restoring plain text would silently flatten, so they must fail
    #     the check even though CF_UNICODETEXT is sitting right there. Browsers
    #     attach HTML Format to every copy, so this is the common case.
    for rich in ("HTML Format", "Rich Text Format"):
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(inject.CF_UNICODETEXT, "hello")
            win32clipboard.SetClipboardData(win32clipboard.RegisterClipboardFormat(rich), b"x")
        finally:
            win32clipboard.CloseClipboard()
        check(f"{rich} clipboard is not borrowed", inject.clipboard_is_restorable() is False)

    # 5. And auto must act on that answer by typing instead of pasting, even
    #    when the text is long enough that it would normally paste.
    typed.clear()
    pasted.clear()
    long_text = "x" * 500
    used = inject.inject(long_text, method="auto", paste_threshold=10)
    check("auto types rather than clobber a non-text clipboard", used == "type", f"(chose {used})")
    check("auto left the non-text clipboard alone", inject.clipboard_is_restorable() is False)

    # 6. If the clipboard cannot be borrowed at all, a forced paste must fall
    #    back to typing rather than lose the transcription.
    inject._clipboard_set_text("prior")
    typed.clear()

    def refuse(text: str, restore_delay: float = 0.35) -> None:
        raise OSError("could not open clipboard: simulated lock")

    real_paste = inject.paste_text
    inject.paste_text = refuse
    try:
        used = inject.inject("some dictation", method="paste")
    finally:
        inject.paste_text = real_paste
    check("locked clipboard falls back to typing", used == "type", f"(chose {used})")
    check("the text was still delivered", typed == ["some dictation"], f"(typed {typed})")

    # 7. pywin32 raises pywintypes.error, which does NOT derive from OSError.
    #    Every guard in inject.py is written as `except OSError`, so if the raw
    #    win32 calls did not convert, a clipboard failure would sail past the
    #    fallback and the transcription would be lost. These drive the real
    #    failure by making the underlying win32 call raise.
    check(
        "pywintypes.error is not an OSError (the reason this section exists)",
        not issubclass(pywintypes.error, OSError),
        f"(mro {[c.__name__ for c in pywintypes.error.__mro__]})",
    )

    def boom(*args, **kwargs):
        raise pywintypes.error(5, "GetClipboardData", "Access is denied.")

    real_get = win32clipboard.GetClipboardData
    win32clipboard.GetClipboardData = boom
    try:
        raised: Exception | None = None
        try:
            inject._clipboard_get_text()
        except Exception as exc:
            raised = exc
        check(
            "a failing win32 read surfaces as OSError",
            isinstance(raised, OSError),
            f"(got {type(raised).__name__})",
        )

        # The one that actually costs the user something: paste must degrade to
        # typing rather than raise, or the dictation is gone.
        typed.clear()
        used = inject.inject("dictation that must survive", method="paste")
        check("a win32 clipboard failure still falls back to typing", used == "type",
              f"(chose {used})")
        check("and the text was delivered", typed == ["dictation that must survive"],
              f"(typed {typed})")
    finally:
        win32clipboard.GetClipboardData = real_get

    # clipboard_is_restorable answers a yes/no question and inject() has no
    # handler for it throwing, so an unanswerable question must read as "no".
    real_enum = win32clipboard.EnumClipboardFormats
    win32clipboard.EnumClipboardFormats = boom
    try:
        answer = inject.clipboard_is_restorable()
        check("an unreadable clipboard reports unrestorable rather than raising",
              answer is False, f"(returned {answer!r})")
    except Exception as exc:
        check("an unreadable clipboard reports unrestorable rather than raising",
              False, f"(raised {type(exc).__name__})")
    finally:
        win32clipboard.EnumClipboardFormats = real_enum

    # A failing restore must not escape paste_text and must not leave the
    # dictation behind. Only the write is broken here, so the paste still runs.
    inject._clipboard_set_text("user's own text")
    real_set = win32clipboard.SetClipboardData
    calls = {"n": 0}

    def fail_second_write(fmt, data):
        calls["n"] += 1
        if calls["n"] >= 2:  # let the dictation land, break the restore
            raise pywintypes.error(5, "SetClipboardData", "Access is denied.")
        return real_set(fmt, data)

    win32clipboard.SetClipboardData = fail_second_write
    try:
        escaped: Exception | None = None
        try:
            inject.paste_text("dictated sentence", restore_delay=0.0)
        except Exception as exc:
            escaped = exc
        check("a failing restore does not escape paste_text", escaped is None,
              f"(raised {type(escaped).__name__ if escaped else 'nothing'})")
    finally:
        win32clipboard.SetClipboardData = real_set

finally:
    inject._send = real_send
    inject.type_text = real_type
    if entry_clipboard is not None:
        inject._clipboard_set_text(entry_clipboard)
    else:
        inject._clipboard_clear()
    print(f"  (clipboard restored to: {entry_clipboard!r})")

print(f"\n{'all clipboard checks passed' if not failures else f'{failures} check(s) failed'}")
raise SystemExit(1 if failures else 0)
