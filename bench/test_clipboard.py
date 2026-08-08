"""Verify the clipboard is never left damaged by injection.

Does NOT send keystrokes, so it is safe to run headless: it exercises only the
save / set / restore logic that could clobber whatever the user had copied.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import inject  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures += 1


print("clipboard safety checks")

original = inject._clipboard_get_text()
print(f"  (clipboard on entry: {original!r})")

# 1. Round-trip a value.
inject._clipboard_set_text("sentinel-alpha")
check("set/get round-trip", inject._clipboard_get_text() == "sentinel-alpha")

# 2. Unicode survives intact.
tricky = "café naïve 日本語 emoji ok"
inject._clipboard_set_text(tricky)
check("unicode round-trip", inject._clipboard_get_text() == tricky)

# 3. Text clipboard is reported restorable.
inject._clipboard_set_text("some prior copy")
check("text clipboard is restorable", inject.clipboard_is_restorable() is True)

# 4. The save/restore contract used by paste_text preserves prior contents.
prior = inject._clipboard_get_text()
inject._clipboard_set_text("transcribed text")
inject._clipboard_set_text(prior)
check("prior contents restored", inject._clipboard_get_text() == "some prior copy")

# 5. Empty clipboard is restorable (nothing to lose).
inject._clipboard_set_text("")
check("empty clipboard handled", inject.clipboard_is_restorable() is True)

# 6. auto never pastes into a clipboard it cannot restore.
#    We cannot easily put an image on the clipboard here, so assert the guard
#    exists and is consulted rather than faking the state.
src = Path(__file__).resolve().parent.parent / "localflow" / "inject.py"
body = src.read_text(encoding="utf-8")
check(
    "auto consults clipboard_is_restorable",
    "clipboard_is_restorable()" in body.split("# auto")[1],
    "(guards against clobbering images/files)",
)

# Put back whatever the user actually had.
if original is not None:
    inject._clipboard_set_text(original)
    print(f"  (clipboard restored to: {original!r})")
else:
    inject._clipboard_set_text("")
    print("  (clipboard had no text on entry; left empty)")

print(f"\n{'all clipboard checks passed' if not failures else f'{failures} check(s) failed'}")
raise SystemExit(1 if failures else 0)
