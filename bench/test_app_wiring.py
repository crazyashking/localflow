"""Exercise the full app path minus the two hardware-bound pieces.

Feeds a known WAV straight through DictationApp._handle, which is exactly what
the hotkey release callback does. Injection is stubbed so nothing is typed into
whatever window happens to be focused, and transcripts are written to a temp
directory rather than the real one.

Covers: transcribe -> inject dispatch -> transcript write -> status reporting.
Not covered here (needs a human): the keyboard hook and live mic capture.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_whisper.audio import decode_audio  # noqa: E402

from localflow import app as app_module  # noqa: E402
from localflow.app import DictationApp  # noqa: E402
from localflow.config import DEFAULTS, Settings  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures += 1


tmpdir = Path(tempfile.mkdtemp(prefix="localflow-test-"))
settings = Settings({**DEFAULTS, "transcript_dir": str(tmpdir), "save_transcripts": True})

# Stub injection: capture what would have been typed instead of typing it.
injected: list[tuple[str, str]] = []


def fake_inject(text: str, method: str = "auto", paste_threshold: int = 120) -> str:
    used = "paste" if len(text) >= paste_threshold else "type"
    injected.append((text, used))
    return used


app_module.inject.inject = fake_inject  # type: ignore[assignment]

statuses: list[tuple[str, str]] = []
application = DictationApp(settings, on_status=lambda s, d: statuses.append((s, d)))

print("app wiring checks")
print("  loading model ...")
application.transcriber.load()

clip = decode_audio(str(Path(__file__).parent / "samples" / "s1.wav"), sampling_rate=16000)
application._handle(clip)

check("injection was called", len(injected) == 1)
if injected:
    text, method = injected[0]
    check("transcribed text is non-empty", bool(text.strip()), f"({len(text)} chars)")
    check("expected content present", "quick brown fox" in text.lower())
    # Route must follow the configured threshold, not a guess about it.
    expected = "paste" if len(text) >= settings.paste_threshold_chars else "type"
    check(
        "injection route matches threshold",
        method == expected,
        f"({len(text)} chars vs threshold {settings.paste_threshold_chars} -> {method})",
    )

check("utterance counted", application.utterances == 1)

written = list(tmpdir.glob("*.md"))
check("transcript file written", len(written) == 1, f"({written[0].name if written else 'none'})")
if written:
    body = written[0].read_text(encoding="utf-8")
    check("transcript contains the text", "quick brown fox" in body.lower())
    check("transcript records timing", "decode" in body and "x)" in body)

check("final status is ready", statuses and statuses[-1][0] == "ready",
      f"({statuses[-1] if statuses else 'none'})")

# Silence must not produce an utterance, an injection, or a transcript entry.
import numpy as np  # noqa: E402

before = application.utterances
application._handle(np.zeros(16000 * 2, dtype=np.float32))
check("silence injects nothing", len(injected) == 1)
check("silence adds no utterance", application.utterances == before)

print(f"\n{'all wiring checks passed' if not failures else f'{failures} check(s) failed'}")
print(f"(temp transcripts: {tmpdir})")
raise SystemExit(1 if failures else 0)
