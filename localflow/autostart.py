"""Start LocalFlow when Windows starts, via a shortcut in the Startup folder.

Three ways exist to do this on Windows, and the Startup folder is the one that
suits an app the user talks to:

  Startup folder  a .lnk in the per-user Startup directory. No admin rights, no
                  elevation prompt, and the user can see it and delete it in
                  Explorer without knowing what a registry key is. It runs once
                  the desktop shell is up, which is what the keyboard hook and
                  SendInput need anyway.
  Run key         HKCU\\...\\Run. Same timing, but invisible to anyone who does
                  not go looking in regedit.
  Task Scheduler  can run before login and with raised privileges. Both are
                  wrong here: dictation types into the session's focused window,
                  so there is nothing to type into until someone has logged in.

The shortcut points at the interpreter running right now, so a virtualenv is
picked up without being told about it, and at PROJECT_ROOT as the working
directory so settings.json and models/ resolve exactly as they do by hand.

sync() runs on every launch and rewrites a shortcut whose target has drifted.
That is what makes moving or rebuilding the venv safe: the old shortcut would
otherwise keep pointing at an interpreter that no longer exists, and the failure
shows up as nothing happening at login, with no error anyone ever sees.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import PROJECT_ROOT

SHORTCUT_NAME = "LocalFlow.lnk"

# Minimised, so logging in does not throw a console window over whatever the
# user is doing. The window stays on the taskbar, which is still how the app is
# read and how it is quit.
_SW_SHOWMINNOACTIVE = 7


def startup_dir() -> Path:
    """The current user's Startup folder, from the shell rather than guessed.

    Building this path from APPDATA assumes an English install and a Start Menu
    that has never been redirected by group policy. The shell knows.
    """
    from win32com.client import Dispatch  # from pywin32

    return Path(Dispatch("WScript.Shell").SpecialFolders("Startup"))


def shortcut_path() -> Path:
    return startup_dir() / SHORTCUT_NAME


def _target() -> tuple[str, str, str]:
    """(executable, arguments, working directory) the shortcut should hold."""
    # python.exe rather than pythonw.exe on purpose. The console is the app's
    # status display, and closing it is the documented way to quit; pythonw
    # would leave a process with no window and no way to stop it short of Task
    # Manager.
    return sys.executable, "-m localflow", str(PROJECT_ROOT)


def install() -> Path:
    """Create or refresh the Startup shortcut. Returns where it was written."""
    from win32com.client import Dispatch  # from pywin32

    exe, args, workdir = _target()
    path = shortcut_path()
    link = Dispatch("WScript.Shell").CreateShortcut(str(path))
    link.TargetPath = exe
    link.Arguments = args
    link.WorkingDirectory = workdir
    link.WindowStyle = _SW_SHOWMINNOACTIVE
    link.Description = "LocalFlow offline dictation"
    link.Save()
    return path


def remove() -> bool:
    """Delete the Startup shortcut. True if one was there to delete."""
    path = shortcut_path()
    if not path.exists():
        return False
    path.unlink()
    return True


def is_current() -> bool:
    """Whether an installed shortcut already points at this checkout and venv."""
    from win32com.client import Dispatch  # from pywin32

    path = shortcut_path()
    if not path.exists():
        return False
    exe, args, workdir = _target()
    link = Dispatch("WScript.Shell").CreateShortcut(str(path))
    # Case-insensitive because Windows paths are, and the shell hands back
    # whatever casing it stored.
    return (
        link.TargetPath.lower() == exe.lower()
        and link.Arguments.strip() == args
        and link.WorkingDirectory.lower().rstrip("\\") == workdir.lower().rstrip("\\")
    )


def sync(enabled: bool) -> str:
    """Make the Startup folder agree with the setting. Returns a status line.

    Never raises. A machine where the shortcut cannot be written (a locked-down
    Startup folder, COM unavailable) is still a machine where dictation itself
    works, so this reports the problem and gets out of the way.
    """
    try:
        if enabled:
            if is_current():
                return "starts with Windows"
            install()
            return f"starts with Windows   installed in {startup_dir()}"
        if remove():
            return "removed from Windows startup"
        return ""
    except Exception as exc:
        return f"could not be changed ({exc})"
