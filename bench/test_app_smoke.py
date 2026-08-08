"""Start the real app with the overlay, let it run, then shut it down.

Covers the integration that no other test does: model warm-up, live mic
stream, keyboard hook, and the Tk overlay owning the main thread, all at once,
plus a clean shutdown. Simulates one utterance by driving the recorder
directly, since nobody is here to speak.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow.app import DictationApp  # noqa: E402
from localflow.config import DEFAULTS, Settings  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures += 1


settings = Settings({**DEFAULTS, "save_transcripts": False, "overlay": True})
seen: list[str] = []
app = DictationApp(settings, on_status=lambda s, d: seen.append(s))

print("app smoke test (a small overlay will appear near the bottom of the screen)")
print("  warming up ...")
app.warm_up()
check("model warm and mic open", app.transcriber.ready and app.recorder._stream is not None)
check("overlay created", app.overlay is not None)


def driver() -> None:
    time.sleep(2.5)  # let the hook install and the overlay build

    check("hotkey listener running", app._listener is not None and app._listener.running)

    # Simulate a press and release without a human.
    app._on_press()
    check("overlay shows recording", app.overlay.state == "recording")
    time.sleep(1.2)
    lvl = app.recorder.live_level
    check("live level readable", 0.0 <= lvl <= 1.0, f"(level={lvl:.3f})")
    app._on_release()

    time.sleep(2.5)
    check("returned to ready", app.overlay.state == "ready", f"(state={app.overlay.state})")
    check("status transitions seen", "recording" in seen, f"({seen})")

    app.stop()


thread = threading.Thread(target=driver, daemon=True)
thread.start()
app.start()  # blocks in the Tk loop until stop()
thread.join(timeout=5)

check("shut down cleanly", app._stopping.is_set())

print(f"\n{'app smoke test passed' if not failures else f'{failures} check(s) failed'}")
raise SystemExit(1 if failures else 0)
