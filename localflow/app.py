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
from collections.abc import Callable

import numpy as np

from . import asr, audio, glow, hotkey, inject, overlay, wake
from .config import Settings
from .endpoint import SilenceEndpointer
from .history import TranscriptLog
from .models import SAMPLE_RATE

StatusFn = Callable[[str, str], None]

# How often the endpoint watcher samples the microphone level. Fast enough that
# the measured silence overshoots by at most this much, cheap enough to ignore.
ENDPOINT_POLL_SECONDS = 0.05

# Waking from sleep is detected by watching the clock rather than by asking
# Windows: a thread that sleeps RESUME_POLL_S at a time and comes back much
# later was frozen. See _watch_resume.
RESUME_POLL_S = 1.0
RESUME_GAP_S = 5.0


class DictationApp:
    def __init__(self, settings: Settings, on_status: StatusFn | None = None):
        self.settings = settings
        self.on_status = on_status or (lambda state, detail: None)

        # Built before the recorder, because the recorder needs its feed() as
        # the audio tap.
        self.wake: wake.WakeListener | None = None
        if settings.wake_word:
            self.wake = wake.WakeListener(
                on_wake=self._on_wake,
                phrase=settings.wake_phrase,
                threshold=settings.wake_threshold,
                patience=settings.wake_patience,
                debounce_seconds=settings.wake_debounce_seconds,
            )

        self.recorder = audio.Recorder(
            device=settings.input_device,
            max_seconds=settings.max_seconds,
            min_seconds=settings.min_seconds,
            level_ceiling=settings.mic_level_ceiling,
            on_block=self.wake.feed if self.wake is not None else None,
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

        # Wake word state. _wake_active means the current recording was started
        # by voice, which is what makes the hotkey mean "finish this" instead
        # of "start a new one".
        self._wake_active = threading.Event()
        self._swallow_release = False
        self._endpointer = SilenceEndpointer(
            silence_seconds=settings.wake_endpoint_silence
        )
        self._wake_thread: threading.Thread | None = None

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

        self.glow: glow.BorderGlow | None = None
        if settings.glow:
            self.glow = glow.BorderGlow(
                lambda: self.recorder.live_level,
                opacity=settings.glow_opacity,
                thickness_frac=settings.glow_thickness,
                which_monitors=settings.glow_monitors,
                sweep_from=settings.glow_sweep_from,
                sweep_style=settings.glow_sweep_style,
                sweep_seconds=settings.glow_sweep_seconds,
                linger_seconds=settings.glow_linger_seconds,
                max_seconds=settings.max_seconds,
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

        if self.wake is not None:
            self._status("loading", "loading wake word model")
            try:
                self.wake.load()
            except wake.WakeError as exc:
                # A missing wake word must not take dictation down with it.
                # Push-to-talk is the primary path and works without this.
                print(f"\n[wake] disabled: {exc}\n")
                self.recorder.on_block = None
                self.wake = None

    # --- hotkey callbacks (must return immediately) -----------------------

    def _on_press(self) -> None:
        # One key, two meanings, decided by whether a recording is already
        # running. Pressing it during a wake-word recording finishes that
        # utterance; pressing it while idle is ordinary push-to-talk. Without
        # this the key would start a second recording on top of the first and
        # the wake word would be unusable for anyone who still uses the hotkey.
        if self._wake_active.is_set():
            if self.settings.wake_end_mode in ("both", "hotkey"):
                self._swallow_release = True
                self._finish_wake_utterance("ended by hotkey")
            return

        self.recorder.begin()
        self._status("recording", "listening")

    def _on_release(self) -> None:
        # The key-up that follows a tap used to end a wake utterance is not the
        # end of a push-to-talk hold, so it must not run the normal path. It
        # would call recorder.end() a second time on an already-stopped
        # recorder and report a spurious "too short, ignored".
        if self._swallow_release:
            self._swallow_release = False
            return

        self._close_recording()

    def _close_recording(self, empty_detail: str = "too short, ignored") -> None:
        """Stop the recorder and queue whatever it captured.

        Shared by the hotkey release and by the wake word's endpointer, so both
        routes report the same warnings and cannot drift apart.
        """
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
            self._status("ready", empty_detail)
            return
        self._status("transcribing", f"{len(clip) / SAMPLE_RATE:.1f}s captured")
        self._jobs.put(clip)

    # --- wake word --------------------------------------------------------

    def _on_wake(self) -> None:
        """Called from the detector thread when the phrase is heard.

        Must return immediately: it runs on the thread that scores audio, and
        anything slow here delays the next frame.
        """
        if self._wake_active.is_set() or self.recorder.recording:
            return
        if self.wake is not None:
            # Stop scoring for the duration. Otherwise the user's own dictation
            # could contain the wake phrase and start a second recording inside
            # the first.
            self.wake.mute()

        # preroll=True is the point: the detector only knows the phrase was
        # spoken once it has finished, and people run straight into their
        # sentence, so the recording has to reach back a second.
        self.recorder.begin(preroll=True)
        self._endpointer.reset(time.monotonic())
        self._wake_active.set()

        if self.settings.wake_end_mode == "hotkey":
            detail = f"listening, tap {hotkey.key_name(self.settings.hotkey_vk)} when done"
        else:
            detail = f"listening, {self.settings.wake_endpoint_silence:.1f}s silence ends it"
        self._status("recording", detail)

    def _finish_wake_utterance(self, reason: str) -> None:
        """Close a wake-started recording, whatever ended it."""
        if not self._wake_active.is_set():
            return
        self._wake_active.clear()
        heard = self._endpointer.heard_speech
        self._close_recording(
            empty_detail="nothing said" if not heard else "too short, ignored"
        )
        if self.wake is not None:
            self.wake.unmute()
        if reason:
            print(f"\n[wake] {reason}")

    def _watch_endpoint(self) -> None:
        """Close a wake-started utterance once the user stops talking.

        Runs on its own thread rather than inside the audio callback, for the
        reason audio.py gives: nothing slow belongs on the PortAudio thread.
        Polling a float is not slow, but the endpointer also has to keep
        running through blocks that never arrive, and a callback cannot notice
        the absence of audio.
        """
        while not self._stopping.is_set():
            if (
                self._wake_active.is_set()
                and self.settings.wake_end_mode != "hotkey"
                and self._endpointer.update(self.recorder.live_level, time.monotonic())
            ):
                self._finish_wake_utterance("closed after silence")
            time.sleep(ENDPOINT_POLL_SECONDS)

    # --- worker -----------------------------------------------------------

    def _run_worker(self) -> None:
        while True:
            clip = self._jobs.get()
            if clip is None:
                break
            try:
                self._handle(clip)
            except Exception as exc:
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
            speed=result.speed,
            method=method,
        )
        self.utterances += 1
        self._status(
            "ready",
            f"{len(result.text)} chars, {result.decode_seconds:.2f}s "
            f"({result.speed:.0f}x), {method}",
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

        if self.wake is not None:
            self._wake_thread = threading.Thread(
                target=self.wake.run, name="localflow-wake", daemon=True
            )
            self._wake_thread.start()
            threading.Thread(
                target=self._watch_endpoint, name="localflow-endpoint", daemon=True
            ).start()

        threading.Thread(
            target=self._watch_resume, name="localflow-resume", daemon=True
        ).start()

        if self.glow is not None:
            self.glow.start()
            # The glow builds its windows on its own thread, so a failure shows
            # up shortly after start() rather than out of the call. Report it
            # once and carry on: dictation does not depend on it.
            threading.Thread(
                target=self._watch_glow, name="localflow-glow-watch", daemon=True
            ).start()

        key = hotkey.key_name(self.settings.hotkey_vk)
        if self.wake is not None:
            phrase = wake.phrase_label(self.settings.wake_phrase)
            self._status("ready", f"hold {key}, or say \"{phrase}\"")
        else:
            self._status("ready", f"hold {key} to dictate")

        if self.overlay is not None:
            # Tk is not thread safe and must own the main thread, so the
            # overlay's animation loop is what keeps the process alive here.
            # That loop cannot also watch the hook, hence the separate watchdog.
            threading.Thread(
                target=self._watch_hook, args=(hook_thread,),
                name="localflow-watchdog", daemon=True,
            ).start()
            self.overlay.run()
        else:
            while not self._stopping.is_set() and hook_thread.is_alive():
                time.sleep(0.15)

    def _watch_resume(self) -> None:
        """Reinstall the keyboard hook after the machine wakes up.

        Windows drops a low-level hook across a sleep without saying so, which
        left the app running and looking healthy while the hotkey did nothing.
        Reported as "it stopped working overnight", and impossible to spot from
        the outside because every other part of the app is fine.

        A resume is detected from the clock: this thread sleeps a second at a
        time, so a much larger jump means the process was frozen. That needs no
        window, no message hook and no power-notification API. The overlays
        recover the same way, each in its own loop.
        """
        while not self._stopping.is_set():
            before = time.monotonic()
            time.sleep(RESUME_POLL_S)
            gap = time.monotonic() - before
            if gap < RESUME_GAP_S or self._listener is None:
                continue
            if self._listener.rehook():
                print(f"\n[hotkey] the machine was asleep for {gap:.0f}s, so the "
                      f"keyboard hook was reinstalled.\n"
                      f"         Windows drops low-level hooks across a sleep "
                      f"without reporting it.\n")

    def _watch_glow(self) -> None:
        """Say so once if the edge glow died, then stop watching.

        Silence would be worse than the failure itself: the glow is the only
        part of the app with no other symptom, so without this it would simply
        never appear and look like a setting that did not take.
        """
        while not self._stopping.is_set():
            if self.glow is None:
                return
            if self.glow.failure:
                print(f"\n[glow] the edge glow stopped: {self.glow.failure}\n"
                      f"       Dictation is unaffected. Set \"glow\": false in "
                      f"settings.json to silence this.\n")
                return
            time.sleep(0.5)

    def _watch_hook(self, hook_thread: threading.Thread) -> None:
        """Shut down if the keyboard hook dies.

        Windows silently unhooks a hook procedure that takes too long, and the
        hook thread can also die outright. Without this the overlay would keep
        animating on the main thread and the app would look perfectly healthy
        while no keypress reached it, which is the worst kind of failure to
        debug. The non-overlay branch above gets this for free from its own
        loop condition.
        """
        while not self._stopping.is_set():
            if not hook_thread.is_alive():
                print(
                    "\n[hotkey] the keyboard hook stopped running, so dictation "
                    "can no longer be triggered.\n"
                    "         Windows unhooks a hook procedure that blocks for "
                    "too long. Restart LocalFlow.\n"
                )
                self._status("error", "keyboard hook died, shutting down")
                self.stop()
                return
            time.sleep(0.25)

    def stop(self) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        if self.glow is not None:
            self.glow.stop()
        if self.overlay is not None:
            self.overlay.stop()
        if self._listener is not None:
            self._listener.stop()
        if self.wake is not None:
            self.wake.stop()
        if self._wake_thread is not None:
            self._wake_thread.join(timeout=2)
        self._jobs.put(None)
        if self._worker is not None:
            self._worker.join(timeout=5)
        self.recorder.close()

    def _status(self, state: str, detail: str = "") -> None:
        if self.overlay is not None:
            self.overlay.set_state(state)
        if self.glow is not None:
            self.glow.set_state(state)
        self.on_status(state, detail)
