"""Append every utterance to a dated transcript file.

One markdown file per day. Both the raw model output and the final text are
recorded, because keeping them side by side is what makes it possible to tell
whether a cleanup rule or a correction map actually helped.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class TranscriptLog:
    def __init__(self, directory: str | Path, enabled: bool = True):
        self.directory = Path(directory)
        self.enabled = enabled
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        text: str,
        *,
        raw_text: str = "",
        profile: str = "general",
        audio_seconds: float = 0.0,
        decode_seconds: float = 0.0,
        method: str = "",
    ) -> Path | None:
        if not self.enabled or not text.strip():
            return None

        now = datetime.now()
        path = self.directory / f"{now:%Y-%m-%d}.md"
        new_file = not path.exists()

        lines: list[str] = []
        if new_file:
            lines.append(f"# LocalFlow transcript, {now:%A %d %B %Y}\n")

        speed = audio_seconds / decode_seconds if decode_seconds > 0 else 0.0
        lines.append(f"\n## {now:%H:%M:%S}\n")
        lines.append(
            f"`{profile}` | {audio_seconds:.1f}s audio | "
            f"{decode_seconds:.2f}s decode ({speed:.0f}x) | {method}\n"
        )
        lines.append(f"\n{text.strip()}\n")
        if raw_text.strip() and raw_text.strip() != text.strip():
            lines.append(f"\n<details><summary>raw</summary>\n\n{raw_text.strip()}\n\n</details>\n")

        with path.open("a", encoding="utf-8") as f:
            f.writelines(lines)
        return path
