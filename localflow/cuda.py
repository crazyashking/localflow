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
    """Register NVIDIA DLL directories. Idempotent.

    Returns the list of directories registered. Raises RuntimeError with an
    actionable message if the required libraries are missing.
    """
    global _initialised
    if _initialised:
        return list(_registered)

    if sys.platform != "win32":
        _initialised = True
        return []

    bin_dirs = _discover_bin_dirs()
    if not bin_dirs:
        raise RuntimeError(
            "No NVIDIA DLL directories found in site-packages.\n"
            "Install them into this project's venv with:\n"
            "  pip install -r requirements.txt --require-hashes --only-binary=:all:"
        )

    found = {d.parent.name for d in bin_dirs}
    missing = [name for name in ("cublas", "cudnn") if name not in found]
    if missing:
        raise RuntimeError(
            f"Missing required NVIDIA libraries: {', '.join(missing)}.\n"
            f"Found only: {', '.join(sorted(found)) or 'nothing'}\n"
            "CTranslate2 needs both cuBLAS and cuDNN 9 to run on the GPU."
        )

    for bin_dir in bin_dirs:
        _handles.append(os.add_dll_directory(str(bin_dir)))

    # Also prepend to this process's PATH. add_dll_directory alone is not
    # enough: CTranslate2 loads cuBLAS lazily on first GPU compute through a
    # search path that does not consult it.
    existing = os.environ.get("PATH", "")
    parts = [str(d) for d in bin_dirs if str(d) not in existing]
    if parts:
        os.environ["PATH"] = os.pathsep.join(parts + ([existing] if existing else []))

    _registered.extend(bin_dirs)
    _initialised = True
    return bin_dirs


def describe() -> str:
    """Human-readable summary, for the environment gate and tray diagnostics."""
    dirs = init()
    lines = [f"registered {len(dirs)} NVIDIA DLL director{'y' if len(dirs) == 1 else 'ies'}:"]
    for d in dirs:
        n_dll = len(list(d.glob("*.dll")))
        lines.append(f"  {d.parent.name:<12} {n_dll:>3} dll(s)  {d}")
    return "\n".join(lines)
