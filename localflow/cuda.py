"""Register the pip-installed NVIDIA DLL directories with the Windows loader.

This module MUST be imported before ctranslate2 (and therefore before
faster_whisper). CTranslate2 links against cuBLAS and cuDNN at load time, and
Windows will not search site-packages for them on its own. Without this, the
import fails with an opaque "DLL load failed" that says nothing about the real
cause. Handling it explicitly is the single most common Windows faster-whisper
setup failure.

Two mechanisms are needed, and both are required:

  os.add_dll_directory()  covers DLLs loaded with the modern
                          LOAD_LIBRARY_SEARCH_* flags.
  os.environ["PATH"]      covers plain LoadLibrary calls, which ignore
                          add_dll_directory entirely. CTranslate2 resolves
                          cuBLAS lazily on the first GPU compute this way, so
                          without the PATH entry everything imports and the
                          model even loads, then the first real encode fails
                          with "Library cublas64_12.dll is not found".

Mutating os.environ changes only this process's environment block. Nothing is
written to the registry and the system PATH is untouched.

None of this is fatal on its own. A machine with no NVIDIA libraries installed
is a supported CPU install, so init() records why the GPU is unavailable and
returns. models.py reads that to pick the device. require() is the variant for
callers that genuinely cannot continue without a GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Cookies returned by os.add_dll_directory. These must be kept alive: when a
# cookie is garbage collected, Windows removes the directory again.
_handles: list = []
_registered: list[Path] = []
_initialised = False
# Why the GPU libraries cannot be used, or None when they are fine. Set once by
# init() and read by problem() and require().
_problem: str | None = None


def _discover_bin_dirs() -> list[Path]:
    """Find every nvidia/*/bin directory on sys.path.

    Searching sys.path rather than assuming a fixed layout means this keeps
    working if the project is ever run from an installed package or a
    different venv location.
    """
    seen: list[Path] = []
    for root in (Path(p) / "nvidia" for p in sys.path if p):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            bin_dir = child / "bin"
            if bin_dir.is_dir() and bin_dir not in seen:
                seen.append(bin_dir)
    return seen


def init() -> list[Path]:
    """Register NVIDIA DLL directories. Idempotent, and never raises.

    Returns the directories registered, which is an empty list when the GPU
    libraries are absent or incomplete. In that case problem() explains why, and
    the caller decides whether that is fatal. An empty list is the normal state
    of a CPU install, not an error.
    """
    global _initialised, _problem
    if _initialised:
        return list(_registered)

    if sys.platform != "win32":
        # The loader dance is a Windows problem. Everywhere else CTranslate2
        # finds its CUDA libraries through the ordinary search path, so there is
        # nothing to register here and nothing to report.
        _initialised = True
        return []

    bin_dirs = _discover_bin_dirs()
    if not bin_dirs:
        _problem = (
            "No NVIDIA DLL directories found in site-packages.\n"
            "That is expected on a CPU install. To run on the GPU instead:\n"
            "  pip install -r requirements.txt --require-hashes --only-binary=:all:"
        )
        _initialised = True
        return []

    found = {d.parent.name for d in bin_dirs}
    missing = [name for name in ("cublas", "cudnn") if name not in found]
    if missing:
        # A half-installed GPU stack is worse than none at all, because
        # CTranslate2 still reports the device and only fails on the first real
        # encode. Refusing the GPU here sends it down the CPU path instead,
        # which works.
        _problem = (
            f"Missing required NVIDIA libraries: {', '.join(missing)}.\n"
            f"Found only: {', '.join(sorted(found)) or 'nothing'}\n"
            "CTranslate2 needs both cuBLAS and cuDNN 9 to run on the GPU."
        )
        _initialised = True
        return []

    for bin_dir in bin_dirs:
        _handles.append(os.add_dll_directory(str(bin_dir)))

    # Also prepend to this process's PATH, for the reason in the module
    # docstring. Compare entry by entry: a substring test against the whole
    # PATH string would skip a directory whose path happens to appear inside a
    # longer unrelated entry.
    existing = os.environ.get("PATH", "")
    already = {p.strip().rstrip("\\/").lower() for p in existing.split(os.pathsep) if p.strip()}
    parts = [str(d) for d in bin_dirs if str(d).rstrip("\\/").lower() not in already]
    if parts:
        os.environ["PATH"] = os.pathsep.join(parts + ([existing] if existing else []))

    _registered.extend(bin_dirs)
    _initialised = True
    return bin_dirs


def problem() -> str | None:
    """Why the GPU libraries are unusable, or None when they are fine."""
    init()
    return _problem


def require() -> list[Path]:
    """init(), but fatal when the GPU libraries are missing or incomplete.

    For callers that have already committed to the GPU, where falling back
    silently would hide the reason the GPU was not used.
    """
    dirs = init()
    if _problem:
        raise RuntimeError(_problem)
    return dirs


def describe() -> str:
    """Human-readable summary, for the environment gate."""
    dirs = init()
    if _problem:
        indented = _problem.replace("\n", "\n  ")
        return f"no usable NVIDIA libraries:\n  {indented}"
    if not dirs:
        return "not applicable on this platform (Windows loader workaround only)"
    lines = [f"registered {len(dirs)} NVIDIA DLL director{'y' if len(dirs) == 1 else 'ies'}:"]
    for d in dirs:
        n_dll = len(list(d.glob("*.dll")))
        lines.append(f"  {d.parent.name:<12} {n_dll:>3} dll(s)  {d}")
    return "\n".join(lines)
