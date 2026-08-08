"""The dictation loop.

Hold the hotkey, speak, release. The audio is transcribed on a worker thread
and the result is typed at the cursor and appended to today's transcript.

Threading: the hotkey hook owns the main thread and must never block, so it
only flips recording state and hands the captured audio to a single worker.
One worker (not a pool) keeps utterances in order.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np

from . import asr, audio, hotkey, inject, overlay
from .config import Settings
from .history import TranscriptLog

StatusFn = Callable[[str, str], None]


class DictationApp:
    def __init__(self, settings: Settings, on_status: StatusFn | None = None):
        self.settings = settings
        self.on_status = on_status or (lambda state, detail: None)

        self.recorder = audio.Recorder(
            device=settings.input_device,
            max_seconds=settings.max_seconds,
            min_seconds=settings.min_seconds,
        )
        self.transcriber = asr.Transcriber(
            model_key=settings.model,
            language=settings.language,
        )
        self.log = TranscriptLog(settings.transcript_dir, enabled=settings.save_transcripts)

        self._jobs: queue.Queue[np.ndarray | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._listener: hotkey.HotkeyListener | None = None
        self._stopping = threading.Event()
        self._warned_clipping = False
        self.utterances = 0

        self.overlay: overlay.WaveOverlay | None = None
        if settings.overlay:
            saved = None
            if settings.overlay_x is not None and settings.overlay_y is not None:
                saved = (int(settings.overlay_x), int(settings.overlay_y))
            self.overlay = overlay.WaveOverlay(
                get_level=lambda: self.recorder.live_level,
                position=saved,
                bottom_margin=settings.overlay_bottom_margin,
                opacity=settings.overlay_opacity,
                on_move=self._remember_overlay_position,
            )

    def _remember_overlay_position(self, x: int, y: int) -> None:
        """Persist where the user dragged the overlay to."""
        try:
            self.settings.set(overlay_x=x, overlay_y=y)
            self.settings.save()
        except OSError as exc:
            # Losing the position is a small annoyance; crashing the drag
            # handler would be a real bug, so this stays non-fatal.
            print(f"\n[overlay] could not save position: {exc}")

    # --- setup ------------------------------------------------------------

    def warm_up(self) -> None:
        """Load the model and open the mic before the first hotkey press."""
        self._status("loading", "opening microphone")
        self.recorder.open()

        self._status("loading", "loading model into VRAM")
        t0 = time.perf_counter()
        self.transcriber.load()
        self._status("ready", f"model warm in {time.perf_counter() - t0:.1f}s")

    # --- hotkey callbacks (must return immediately) -----------------------

    def _on_press(self) -> None:
        self.recorder.begin()
        self._status("recording", "listening")

    def _on_release(self) -> None:
        clip = self.recorder.end()

        # Read straight after end(), while the flags still describe THIS clip.
        # The recorder resets them on the next begin(), so checking them later
        # on the worker thread would report the wrong utterance's problems.
        if self.recorder.overflowed:
            print("\n[audio] input overflow: some microphone samples were dropped, "
                  "so this transcription may be inaccurate.\n")
        if self.recorder.hit_cap:
            print(f"\n[audio] you spoke past the {self.settings.max_seconds}s limit, so the "
                  f"rest was not recorded.\n         Raise max_seconds in settings.json "
                  f"if you need longer.\n")

        if clip is None:
            self._status("ready", "too short, ignored")
            return
        self._status("transcribing", f"{len(clip) / audio.SAMPLE_RATE:.1f}s captured")
        self._jobs.put(clip)

    # --- worker -----------------------------------------------------------

    def _run_worker(self) -> None:
        while True:
            clip = self._jobs.get()
            if clip is None:
                break
            try:
                self._handle(clip)
            except Exception as exc:  # noqa: BLE001
                self._status("error", repr(exc))

    def _handle(self, clip: np.ndarray) -> None:
        stats = audio.analyse(clip)
        if stats.is_clipping and not self._warned_clipping:
            self._warned_clipping = True
            print(
                f"\n[audio] input is clipping ({stats.clipped_fraction:.1%} of samples maxed "
                f"out). This degrades accuracy.\n"
                f"        Lower the mic level: Settings > System > Sound > your mic > "
                f"Input volume (try 60-75).\n"
            )

        result = self.transcriber.transcribe(clip, beam_size=self.settings.beam_size)

        if not result.text:
            reason = "silence artifact dropped" if result.dropped_artifact else "no speech detected"
            self._status("ready", reason)
            return

        method = inject.inject(
            result.text,
            method=self.settings.inject_method,
            paste_threshold=self.settings.paste_threshold_chars,
        )
        self.log.append(
            result.text,
            raw_text=result.raw_text,
            profile=self.settings.model,
            audio_seconds=result.audio_seconds,
            decode_seconds=result.decode_seconds,
            method=method,
        )
        self.utterances += 1
        self._status(
            "ready",
            f"{len(result.text)} chars, {result.decode_seconds:.2f}s ({result.speed:.0f}x), {method}",
        )

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Run until stop() is called or the user interrupts.

        The hook's message loop runs on its own thread rather than the main
        one. GetMessageW blocks without ever returning to the interpreter, so
        running it on the main thread means Python never gets to run its signal
        handler and Ctrl+C appears to do nothing, which reads as a frozen app.
        """
        self._worker = threading.Thread(target=self._run_worker, name="localflow-asr", daemon=True)
        self._worker.start()

        self._listener = hotkey.HotkeyListener(
            vk=self.settings.hotkey_vk,
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=self.settings.hotkey_suppress,
        )
        hook_thread = threading.Thread(
            target=self._listener.run, name="localflow-hotkey", daemon=True
        )
        hook_thread.start()

        for _ in range(100):
            if self._listener.running:
                break
            time.sleep(0.02)
        if not self._listener.running:
            raise RuntimeError("keyboard hook failed to install")

        key = hotkey.key_name(self.settings.hotkey_vk)
        self._status("ready", f"hold {key} to dictate")

        if self.overlay is not None:
            # Tk is not thread safe and must own the main thread, so the
            # overlay's animation loop is what keeps the process alive here.
            self.overlay.run()
        else:
            while not self._stopping.is_set() and hook_thread.is_alive():
                time.sleep(0.15)

    def stop(self) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        if self.overlay is not None:
            self.overlay.stop()
        if self._listener is not None:
            self._listener.stop()
        self._jobs.put(None)
        if self._worker is not None:
            self._worker.join(timeout=5)
        self.recorder.close()

    def _status(self, state: str, detail: str = "") -> None:
        if self.overlay is not None:
            self.overlay.set_state(state)
        self.on_status(state, detail)
